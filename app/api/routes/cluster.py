from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.routes._urls import cache_url
from app.db import session_scope
from app.models import Cluster, ClusterMember, Decision, Photo

router = APIRouter()


def _photo_payload(p: Photo, d: Decision | None, rank: int | None) -> dict:
    return {
        "hash": p.hash,
        "filename": Path(p.source_path).name if p.source_path else None,
        "file_kind": p.file_kind,
        "thumb_url": cache_url(p.thumb_path),
        "captured_at": p.captured_at.isoformat() if p.captured_at else None,
        "camera_body": p.camera_body,
        "blur_var": p.blur_var,
        "aesthetic_score": p.aesthetic_score,
        "technical_score": p.technical_score,
        "cluster_id": p.cluster_id,
        "is_recommended": bool(p.is_recommended),
        "rank": rank,
        "decision": (
            {
                "selected": d.selected,
                "stars": d.stars,
                "favorite": bool(d.favorite),
                "applied": bool(d.applied),
                "action": d.action,
            }
            if d
            else None
        ),
    }


@router.get("/")
def list_clusters() -> list[dict]:
    """Every cluster + a synthetic 'unclustered' bucket for photos with no cluster_id."""
    out: list[dict] = []
    with session_scope() as sess:
        clusters = sess.execute(
            select(Cluster).order_by(Cluster.size.desc(), Cluster.id)
        ).scalars().all()
        for c in clusters:
            rows = sess.execute(
                select(ClusterMember, Photo, Decision)
                .join(Photo, Photo.hash == ClusterMember.photo_hash)
                .outerjoin(Decision, Decision.photo_hash == Photo.hash)
                .where(ClusterMember.cluster_id == c.id)
                .order_by(ClusterMember.rank)
            ).all()
            photos = [_photo_payload(p, d, m.rank) for m, p, d in rows]
            out.append(
                {
                    "id": c.id,
                    "kind": c.kind,
                    "label": c.label,
                    "size": c.size,
                    "photos": photos,
                }
            )
        unclustered_rows = sess.execute(
            select(Photo, Decision)
            .outerjoin(Decision, Decision.photo_hash == Photo.hash)
            .where(Photo.cluster_id.is_(None))
            .order_by(Photo.technical_score.desc().nulls_last())
        ).all()
        if unclustered_rows:
            out.append(
                {
                    "id": -1,
                    "kind": "unclustered",
                    "label": None,
                    "size": len(unclustered_rows),
                    "photos": [_photo_payload(p, d, None) for p, d in unclustered_rows],
                }
            )
    return out


@router.get("/{cluster_id}")
def get_cluster(cluster_id: int) -> dict:
    with session_scope() as sess:
        cluster = sess.get(Cluster, cluster_id)
        if not cluster:
            raise HTTPException(status_code=404, detail="cluster not found")
        members = sess.execute(
            select(ClusterMember).where(ClusterMember.cluster_id == cluster_id).order_by(
                ClusterMember.rank
            )
        ).scalars().all()
        return {
            "id": cluster.id,
            "kind": cluster.kind,
            "size": cluster.size,
            "members": [
                {
                    "hash": m.photo_hash,
                    "rank": m.rank,
                    "score": m.score,
                }
                for m in members
            ],
        }
