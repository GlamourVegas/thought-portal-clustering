from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from collections import defaultdict
from datetime import datetime, timezone
import os, math, igraph as ig, leidenalg, httpx
import torch
from huggingface_hub import login
from sentence_transformers import CrossEncoder

app = FastAPI()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
API_SECRET   = os.environ["API_SECRET"]

SB = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

# Hugging Face authenticatie
hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)

# Cross-encoder model — laadt één keer bij server start (~10s)
print("Loading cross-encoder model...")
cross_model = CrossEncoder(
    "cross-encoder/ms-marco-multilingual-MiniLM-L6-v2",
    max_length=256
)
print("Cross-encoder loaded.")


class ClusterRequest(BaseModel):
    user_id: str

class CompareRequest(BaseModel):
    user_id: str
    thought_id: str
    thought_text: str
    threshold: float = 0.5


@app.get("/health")
def health():
    return {"status": "alive", "service": "thought-portal-clustering"}


@app.post("/compare")
async def compare_thought(
    req: CompareRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 1. Haal alle bestaande thoughts op voor deze user
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/thoughts",
            headers=SB,
            params={
                "user_id": f"eq.{req.user_id}",
                "id":      f"neq.{req.thought_id}",
                "select":  "id,raw_thought",
                "limit":   "2000"
            }
        )
        existing = r.json()

    if not existing:
        return {"status": "ok", "synapses_created": 0, "message": "No existing thoughts to compare"}

    # 2. Bouw paren voor cross-encoder
    pairs = [
        [req.thought_text, t["raw_thought"]]
        for t in existing
    ]

    # 3. Batch scoring
    scores = cross_model.predict(
        pairs,
        batch_size=64,
        show_progress_bar=False
    )

    # 4. Normaliseer scores naar 0-1 via sigmoid
    scores_norm = torch.sigmoid(
        torch.tensor(scores)
    ).numpy().tolist()

    # 5. Filter op threshold en bouw synapses
    synapses = []
    for i, score in enumerate(scores_norm):
        if score >= req.threshold:
            synapses.append({
                "user_id":      req.user_id,
                "thought_a_id": req.thought_id,
                "thought_b_id": existing[i]["id"],
                "strength":     round(float(score), 4),
                "method":       "cross-encoder"
            })

    # 6. Batch INSERT naar Supabase
    if synapses:
        async with httpx.AsyncClient(timeout=60.0) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/synapses",
                headers={**SB, "Prefer": "resolution=ignore-duplicates"},
                json=synapses
            )

    return {
        "status":           "ok",
        "compared":         len(existing),
        "synapses_created": len(synapses),
        "avg_score":        round(sum(scores_norm) / len(scores_norm), 4),
        "max_score":        round(max(scores_norm), 4),
        "threshold":        req.threshold
    }


@app.post("/cluster")
async def run_clustering(
    req: ClusterRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    async with httpx.AsyncClient(timeout=60.0) as client:

        # 1. Haal synapses op
        r1 = await client.get(
            f"{SUPABASE_URL}/rest/v1/synapses",
            headers=SB,
            params={
                "user_id": f"eq.{req.user_id}",
                "select":  "thought_a_id,thought_b_id,strength",
                "limit":   "500000"
            }
        )
        synapses = r1.json()

        # 2. Haal thoughts op
        r2 = await client.get(
            f"{SUPABASE_URL}/rest/v1/thoughts",
            headers=SB,
            params={
                "user_id": f"eq.{req.user_id}",
                "select":  "id,valence_score,activation_score,agency_score,created_at",
                "limit":   "500000"
            }
        )
        thoughts = r2.json()

    if not thoughts:
        return {"status": "no_thoughts", "clusters": 0}

    # 3. Bouw igraph
    ids    = [t["id"] for t in thoughts]
    id_idx = {tid: i for i, tid in enumerate(ids)}

    g = ig.Graph(n=len(ids))
    g.vs["name"] = ids

    edges, weights = [], []
    for s in synapses:
        a = id_idx.get(s["thought_a_id"])
        b = id_idx.get(s["thought_b_id"])
        if a is not None and b is not None:
            edges.append((a, b))
            weights.append(float(s["strength"]))

    g.add_edges(edges)
    if weights:
        g.es["weight"] = weights

    # 4. Leiden community detection
    n = len(thoughts)
    resolution = 1.0
    if n > 2000: resolution = 1.2
    if n > 5000: resolution = 1.5

    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        weights="weight" if weights else None,
        seed=42,
        n_iterations=10
    )

    # 5. Sorteer clusters op grootte
    communities = sorted(
        enumerate(partition),
        key=lambda x: len(x[1]),
        reverse=True
    )

    cluster_map = {}
    for new_cid, (_, members) in enumerate(communities, 1):
        for idx in members:
            cluster_map[ids[idx]] = new_cid

    # 6. Groepeer thoughts per cluster
    cluster_thoughts = defaultdict(list)
    for t in thoughts:
        cid = cluster_map.get(t["id"])
        if cid:
            cluster_thoughts[cid].append(t)

    # 7. Bereken circuit_strength per cluster
    circuit_strengths = {}
    for cid, members in cluster_thoughts.items():
        strengths, latest = [], None
        for t in members:
            v  = float(t.get("valence_score")    or 0)
            a  = float(t.get("activation_score") or 0.5)
            ag = float(t.get("agency_score")     or 0.5)
            intensity = math.sqrt(v**2 + a**2 + (1 - ag)**2)
            strengths.append(intensity)
            if t.get("created_at"):
                d = datetime.fromisoformat(t["created_at"].replace("Z", "+00:00"))
                if latest is None or d > latest:
                    latest = d

        avg_intensity = sum(strengths) / len(strengths)
        days_since    = (datetime.now(timezone.utc) - latest).days if latest else 30
        recency       = 1 / (1 + days_since / 30)
        circuit_strengths[cid] = round(len(members) * avg_intensity * recency, 4)

    # 8. Schrijf terug naar Supabase
    async with httpx.AsyncClient(timeout=120.0) as client:

        # Update thoughts.cluster_id
        for tid, cid in cluster_map.items():
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/thoughts",
                headers=SB,
                params={"id": f"eq.{tid}"},
                json={"cluster_id": cid}
            )

        # Upsert cluster_labels
        for cid, members in cluster_thoughts.items():
            avg_v = round(
                sum(float(t.get("valence_score") or 0) for t in members) / len(members), 4
            )
            await client.post(
                f"{SUPABASE_URL}/rest/v1/cluster_labels",
                headers={**SB, "Prefer": "resolution=merge-duplicates"},
                json={
                    "cluster_id":       cid,
                    "user_id":          req.user_id,
                    "thought_count":    len(members),
                    "circuit_strength": circuit_strengths[cid],
                    "avg_valence":      avg_v
                }
            )

    return {
        "status":   "ok",
        "thoughts": n,
        "clusters": len(cluster_thoughts)
    }
