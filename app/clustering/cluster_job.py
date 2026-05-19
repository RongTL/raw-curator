"""Orchestrate clustering: burst -> phash dedupe -> CLIP HDBSCAN."""

from __future__ import annotations

import logging
from collections import defaultdict

from rich.console import Console
from sqlalchemy import select, update

from app.clustering.clip_cluster import cluster_embeddings
from app.clustering.exif_burst import burst_groups
from app.clustering.recommend import rank
from app.config import settings
from app.db import session_scope
from app.models import Cluster, ClusterMember, Photo, PhotoEmbedding

log = logging.getLogger(__name__)
console = Console()


def _create_cluster(sess, kind: str, members: list[Photo]) -> Cluster:
    cluster = Cluster(kind=kind, size=len(members))
    sess.add(cluster)
    sess.flush()
    ranked = rank(members)
    for r, photo in enumerate(ranked):
        sess.add(ClusterMember(cluster_id=cluster.id, photo_hash=photo.hash, rank=r))
        if r == 0:
            sess.execute(
                update(Photo)
                .where(Photo.hash == photo.hash)
                .values(cluster_id=cluster.id, is_recommended=1)
            )
        else:
            sess.execute(
                update(Photo).where(Photo.hash == photo.hash).values(cluster_id=cluster.id)
            )
    return cluster


def run_clustering() -> None:
    with session_scope() as sess:
        photos = list(sess.execute(select(Photo)).scalars().all())
        if not photos:
            console.print("[yellow]No photos to cluster.[/yellow]")
            return

        console.print(f"[cyan]Clustering {len(photos)} photo(s).[/cyan]")
        bursts = burst_groups(photos, window_seconds=settings.burst_seconds)
        console.print(f"  burst groups (>=2): {len(bursts)}")
        for group in bursts:
            _create_cluster(sess, "burst", group)

        # CLIP HDBSCAN over photos not already clustered.
        rows = sess.execute(
            select(PhotoEmbedding.photo_hash, PhotoEmbedding.vec)
        ).all()
        if not rows:
            console.print("[yellow]No CLIP embeddings — skipping HDBSCAN.[/yellow]")
            return

        hashes = [h for h, _ in rows]
        vecs = [v for _, v in rows]
        labels = cluster_embeddings(hashes, vecs)
        groups: dict[int, list[str]] = defaultdict(list)
        for h, lbl in labels.items():
            if lbl >= 0:
                groups[lbl].append(h)
        console.print(f"  CLIP clusters: {len(groups)}")

        by_hash = {p.hash: p for p in photos}
        for _lbl, hashes_in_cluster in groups.items():
            members = [by_hash[h] for h in hashes_in_cluster if by_hash[h].cluster_id is None]
            if len(members) >= 2:
                _create_cluster(sess, "clip-hdbscan", members)

    console.print("[green]Clustering complete.[/green]")
