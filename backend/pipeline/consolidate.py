"""Memory consolidation ("sleep") algorithm.

This is the core of the hippocampal model. The flow:

  1. Pull all active (non-superseded) episodes for the session.
  2. Single-link cluster them by cosine similarity >= SIM_THRESHOLD.
  3. Sort clusters by novelty vs existing consolidated memories; take the
     top MAX_CLUSTERS_PER_PASS, capped by total episode budget.
  4. For each cluster, call the LLM with the consolidation prompt.
  5. If a near-duplicate consolidated memory already exists, merge into it.
  6. Otherwise add the new memory, and if supersedes_episodes is true (or
     forced by the post-process rule), soft-delete the source episodes.

The brain analogy: hippocampal traces are replayed during slow-wave sleep;
overlapping traces get generalised into a cortical engram (consolidated
memory) and the source traces lose their retrieval weight.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any

import numpy as np

from .. import config
from ..agents.prompts import render_consolidation_prompt
from ..embeddings import Embedder
from ..llm import OllamaClient
from ..stores.consolidated_store import ConsolidatedStore
from ..stores.episode_store import EpisodeStore


log = logging.getLogger(__name__)


# --- union-find ---


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _cluster_by_similarity(
    embeddings: np.ndarray, threshold: float
) -> list[list[int]]:
    """Single-link clustering on cosine similarity >= threshold.
    Returns a list of clusters (each a list of original indices)."""
    n = embeddings.shape[0]
    if n == 0:
        return []
    # Vectors are L2-normalized by our Embedder, so cosine == dot product.
    sims = embeddings @ embeddings.T
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    return list(groups.values())


# --- helpers ---


def _format_cluster_for_prompt(episodes: list[dict]) -> str:
    lines: list[str] = []
    for e in episodes:
        turn = e["metadata"].get("turn_index", "?")
        role = e["metadata"].get("role", "?")
        text = e["text"].strip()
        lines.append(f"[turn {turn} | {role}]\n{text}")
    return "\n\n---\n\n".join(lines)


def _cluster_novelty(
    cluster: list[dict],
    embeddings: np.ndarray,
    existing_memories: list[dict],
) -> float:
    """Higher = more novel vs what's already consolidated."""
    if not existing_memories:
        return 1.0
    # Mean embedding of the cluster.
    idx = [e["_idx"] for e in cluster]
    centroid = embeddings[idx].mean(axis=0, keepdims=True)
    # Normalise centroid.
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    # Compare to existing consolidated embeddings (we don't have them cached;
    # approximate by max sim to any existing memory's stored embedding is
    # available via query() — but to keep this cheap we just use cluster size
    # and recency as a proxy here. The consolidator will run a real dedup
    # pass later against the actual memory vectors.)
    return float(len(cluster))


# --- main entry ---


def consolidate(
    session_id: str,
    episode_store: EpisodeStore,
    consolidated_store: ConsolidatedStore,
    embedder: Embedder,
    llm: OllamaClient,
) -> dict:
    """Run one consolidation pass. Returns a serialisable report dict."""
    t0 = time.time()
    report: dict[str, Any] = {
        "session_id": session_id,
        "clusters_processed": 0,
        "memories_created": 0,
        "episodes_marked_superseded": 0,
        "details": [],
    }

    # 1. Pull all active episodes.
    episodes = episode_store.list_active(session_id)
    if len(episodes) < config.MIN_CLUSTER_SIZE:
        report["skipped"] = "too few active episodes"
        report["duration_ms"] = int((time.time() - t0) * 1000)
        return report

    # 2. Build embedding matrix.
    # Episode embeddings aren't stored on the Episode dict (Chroma holds them);
    # we need to fetch. We re-embed the texts: cheaper than a second roundtrip
    # and gives us up-to-date vectors. (For a real product you'd cache them.)
    texts = [e["text"] for e in episodes]
    vectors = embedder.encode(texts)  # (N, D) L2-normalized
    # Tag original indices onto each episode so we can map back.
    for i, e in enumerate(episodes):
        e["_idx"] = i
        e["_vec"] = vectors[i]

    # 3. Single-link clustering.
    clusters = _cluster_by_similarity(vectors, config.SIM_THRESHOLD)
    clusters = [c for c in clusters if len(c) >= config.MIN_CLUSTER_SIZE]
    if not clusters:
        report["skipped"] = "no clusters above threshold"
        report["duration_ms"] = int((time.time() - t0) * 1000)
        return report

    # Map index -> episode dict.
    idx_to_episode = {i: episodes[i] for i in range(len(episodes))}
    cluster_episode_lists = [
        [idx_to_episode[i] for i in c] for c in clusters
    ]

    # 4. Sort by novelty (here: cluster size as a proxy), take top.
    cluster_episode_lists.sort(key=len, reverse=True)

    # Cap by total episode budget.
    budget_left = config.MAX_EPISODES_PER_PASS
    selected: list[list[dict]] = []
    for cl in cluster_episode_lists:
        if len(selected) >= config.MAX_CLUSTERS_PER_PASS:
            break
        if budget_left <= 0:
            break
        if len(cl) > budget_left:
            # Take the most recent subset of the cluster.
            cl = sorted(cl, key=lambda e: int(e["metadata"].get("turn_index", 0)))[
                -budget_left:
            ]
        selected.append(cl)
        budget_left -= len(cl)

    # 5. For each cluster, call LLM and persist.
    for cluster in selected:
        episode_objs = cluster
        source_ids = [e["id"] for e in episode_objs]
        cluster_text = _format_cluster_for_prompt(episode_objs)
        prompt = render_consolidation_prompt(
            episodes_text=cluster_text, n=len(episode_objs)
        )

        # Force JSON-leaning system message: just use the prompt directly.
        try:
            result = llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You produce only valid JSON. No prose, no markdown "
                            "fences. Just the JSON object."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as e:
            log.warning("Consolidation LLM call failed: %s", e)
            report["details"].append(
                {
                    "cluster_size": len(cluster),
                    "error": str(e),
                }
            )
            continue

        summary = (result.get("summary") or "").strip()
        if not summary:
            log.warning("Consolidator returned empty summary; skipping cluster")
            continue

        # Coerce types defensively.
        try:
            confidence = float(result.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        try:
            supersedes_raw = bool(result.get("supersedes_episodes", False))
        except Exception:
            supersedes_raw = False
        try:
            preserved_facts = list(result.get("preserved_facts", []) or [])
        except Exception:
            preserved_facts = []

        # 6. Post-process: force supersedes for big high-confidence clusters.
        force_supersede = (
            len(cluster) >= config.FORCE_SUPERSEDE_MIN_SUPPORT
            and confidence >= config.FORCE_SUPERSEDE_MIN_CONFIDENCE
        )
        supersedes = supersedes_raw or force_supersede

        # Embed the new summary.
        vec = embedder.encode_one(summary)

        # 7. Dedup pass.
        existing = consolidated_store.find_near_duplicate(
            session_id, vec, config.CONSOLIDATED_DEDUP_THRESHOLD
        )
        if existing is not None:
            # Merge into the existing memory: union source ids, bump support.
            new_ids = [eid for eid in source_ids]
            consolidated_store.merge_into(
                existing["id"],
                new_ids,
                # Keep whichever text is longer; ties to the new one.
                summary if len(summary) > len(existing["text"]) else existing["text"],
            )
            if supersedes:
                episode_store.mark_superseded(source_ids, existing["id"])
                report["episodes_marked_superseded"] += len(source_ids)
            report["details"].append(
                {
                    "cluster_size": len(cluster),
                    "merged_into_existing": existing["id"],
                    "confidence": confidence,
                    "supersedes": supersedes,
                    "forced": supersedes_raw != supersedes,
                }
            )
            report["clusters_processed"] += 1
            continue

        # 8. Add new consolidated memory.
        new_id = consolidated_store.add(
            session_id=session_id,
            text=summary,
            embedding=vec,
            source_episode_ids=source_ids,
            confidence=confidence,
            support_count=len(cluster),
            preserved_facts=preserved_facts,
        )

        # 9. Soft-delete source episodes if supersedes.
        if supersedes:
            episode_store.mark_superseded(source_ids, new_id)
            report["episodes_marked_superseded"] += len(source_ids)

        report["memories_created"] += 1
        report["clusters_processed"] += 1
        report["details"].append(
            {
                "cluster_size": len(cluster),
                "memory_id": new_id,
                "confidence": confidence,
                "supersedes": supersedes,
                "forced": supersedes_raw != supersedes,
                "summary_preview": summary[:160],
            }
        )
        log.info(
            "Consolidated %d episodes into memory %s (confidence=%.2f, supersedes=%s)",
            len(cluster),
            new_id,
            confidence,
            supersedes,
        )

    report["duration_ms"] = int((time.time() - t0) * 1000)
    return report
