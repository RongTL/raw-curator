from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.db import session_scope
from app.models import Cluster, ClusterMember, Photo

router = APIRouter()


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
