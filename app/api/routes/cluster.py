from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.routes._review_data import enhanced_by_hash, face_counts_by_hash
from app.api.serializers import photo_summary
from app.db import session_scope
from app.models import Cluster, ClusterMember, Decision, Photo

router = APIRouter()

UNCLUSTERED_ID = -1


@router.get("/")
def list_clusters() -> list[dict[str, Any]]:
    """Every cluster + a synthetic 'unclustered' bucket for photos with no cluster_id."""
    out: list[dict[str, Any]] = []
    with session_scope() as sess:
        face_counts = face_counts_by_hash(sess)
        enhanced = enhanced_by_hash(sess)

        def summary(p: Photo, d: Decision | None, rank: int | None = None) -> dict[str, Any]:
            q_after, degraded = enhanced.get(p.hash, (None, False))
            return photo_summary(
                p,
                d,
                rank=rank,
                q_after=q_after,
                degraded=degraded,
                n_faces=face_counts.get(p.hash, 0),
            )

        clusters = (
            sess.execute(select(Cluster).order_by(Cluster.size.desc(), Cluster.id)).scalars().all()
        )
        for c in clusters:
            rows = sess.execute(
                select(ClusterMember, Photo, Decision)
                .join(Photo, Photo.hash == ClusterMember.photo_hash)
                .outerjoin(Decision, Decision.photo_hash == Photo.hash)
                .where(ClusterMember.cluster_id == c.id)
                .order_by(ClusterMember.rank)
            ).all()
            out.append(
                {
                    "id": c.id,
                    "kind": c.kind,
                    "label": c.label,
                    "size": c.size,
                    "photos": [summary(p, d, rank=m.rank) for m, p, d in rows],
                }
            )
        unclustered = sess.execute(
            select(Photo, Decision)
            .outerjoin(Decision, Decision.photo_hash == Photo.hash)
            .where(Photo.cluster_id.is_(None))
            .order_by(Photo.technical_score.desc().nulls_last())
        ).all()
        if unclustered:
            out.append(
                {
                    "id": UNCLUSTERED_ID,
                    "kind": "unclustered",
                    "label": None,
                    "size": len(unclustered),
                    "photos": [summary(p, d) for p, d in unclustered],
                }
            )
    return out


@router.get("/{cluster_id}")
def get_cluster(cluster_id: int) -> dict[str, Any]:
    with session_scope() as sess:
        cluster = sess.get(Cluster, cluster_id)
        if not cluster:
            raise HTTPException(status_code=404, detail="cluster not found")
        members = (
            sess.execute(
                select(ClusterMember)
                .where(ClusterMember.cluster_id == cluster_id)
                .order_by(ClusterMember.rank)
            )
            .scalars()
            .all()
        )
        return {
            "id": cluster.id,
            "kind": cluster.kind,
            "size": cluster.size,
            "members": [{"hash": m.photo_hash, "rank": m.rank, "score": m.score} for m in members],
        }
