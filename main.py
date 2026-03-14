"""
Thought Portal — Synapse Detection & Clustering Service v3.0
=============================================================
Method: Three-way Composite Similarity
  score = α · embedding_cosine + β · dimension_cosine(22 dims, no valence) + γ · valence_alignment

Model:  paraphrase-multilingual-MiniLM-L12-v2 (~470MB, 50+ talen)
Host:   Railway (CPU)

Waarom drieweg:
  - Embedding (α=0.70): poortwachter — dwingt thematische overlap af
  - Valence alignment (γ=0.15): splitter — scheidt positief van negatief
  - Dimensies zonder valence (β=0.15): tiebreaker — psychologische bonus

Wiskundig bewijs: max score zonder embedding-overlap = 0.15 + 0.15 = 0.30
  → na calibratie ~0.17 → ver onder threshold 0.55
  → superclusters op basis van alleen emotie zijn ONMOGELIJK
"""

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from collections import defaultdict
from datetime import datetime, timezone
import os, math, igraph as ig, leidenalg, httpx
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Thought Portal Clustering", version="3.0.0")

# ── Environment ──────────────────────────────────────────────
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_KEY"]
API_SECRET   = os.environ["API_SECRET"]

SB = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
}

# ── Tuning Parameters (override via Railway env vars) ────────
ALPHA          = float(os.environ.get("SYN_ALPHA",     "0.70"))
BETA           = float(os.environ.get("SYN_BETA",      "0.15"))
GAMMA          = float(os.environ.get("SYN_GAMMA",     "0.15"))
EDGE_THRESHOLD = float(os.environ.get("SYN_THRESHOLD", "0.55"))
CAL_MIDPOINT   = float(os.environ.get("SYN_CAL_MID",   "0.50"))
CAL_STEEPNESS  = float(os.environ.get("SYN_CAL_STEEP",  "8"))
LEIDEN_RES_OVERRIDE = os.environ.get("LEIDEN_RESOLUTION")

# ── Valence: separate dimension (the splitter) ───────────────
VALENCE_COL = "valence_score"

# ── 22 Psychological Dimensions (without valence) ────────────
DIMENSION_COLS = [
    "activation_score",
    "agency_score",
    "temporal_score",
    "certainty_score",
    "depth_score",
    "rumination_signal",
    "counterfactual",
    "abstractness",
    "approach_avoidance",
    "intrinsic_extrinsic",
    "urgency",
    "energy_direction_atomic",
    "social_orientation",
    "social_comparison",
    "self_disclosure",
    "self_compassion",
    "values_alignment",
    "somatic_awareness",
    "attribution_internal",
    "attribution_stable",
    "goal_relevance",
    "agency_direction",
]

ALL_DIM_COLS = [VALENCE_COL] + DIMENSION_COLS

# ── Model laden ──────────────────────────────────────────────
print("⏳ Loading bi-encoder: paraphrase-multilingual-MiniLM-L12-v2 ...")
model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMB_DIM = model.get_sentence_embedding_dimension()
print(f"✅ Bi-encoder loaded — embedding dim: {EMB_DIM}")
print(f"✅ Scoring: α={ALPHA} (emb) + β={BETA} (dim/22) + γ={GAMMA} (valence)")


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def calibrate(raw: float) -> float:
    x = CAL_STEEPNESS * (raw - CAL_MIDPOINT)
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def auto_resolution(n_thoughts: int, n_edges: int) -> float:
    if LEIDEN_RES_OVERRIDE is not None:
        return float(LEIDEN_RES_OVERRIDE)
    if n_thoughts < 10:
        base = 0.5
    elif n_thoughts < 50:
        base = 1.0
    elif n_thoughts < 200:
        base = 1.5
    elif n_thoughts < 500:
        base = 2.0
    elif n_thoughts < 1000:
        base = 2.5
    else:
        base = 3.0
    if n_thoughts > 1:
        avg_edges = n_edges / n_thoughts
        if avg_edges > 50:
            base += 0.5
        elif avg_edges > 30:
            base += 0.25
    return round(base, 2)


def get_valence(thought: dict) -> float:
    return float(thought.get(VALENCE_COL) or 0.0)


def dim_vector(thought: dict) -> np.ndarray:
    return np.array([float(thought.get(c) or 0.0) for c in DIMENSION_COLS],
                    dtype=np.float32)


def valence_alignment(va: float, vb: float) -> float:
    """1.0 = identical valence, 0.0 = opposite. Formula: 1 - |va - vb| / 2"""
    return 1.0 - abs(va - vb) / 2.0


def dim_cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_composite_scores(
    new_emb: np.ndarray,
    new_valence: float,
    new_dims: np.ndarray,
    emb_matrix: np.ndarray,
    valences: np.ndarray,
    dim_vectors: np.ndarray,
) -> np.ndarray:
    n = len(valences)
    emb_sims = emb_matrix @ new_emb

    new_norm = np.linalg.norm(new_dims)
    if new_norm > 1e-9 and len(DIMENSION_COLS) > 0:
        new_d = new_dims / new_norm
        norms = np.linalg.norm(dim_vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        dim_normed = dim_vectors / norms
        dim_sims = dim_normed @ new_d
    else:
        dim_sims = np.zeros(n, dtype=np.float32)

    val_aligns = 1.0 - np.abs(valences - new_valence) / 2.0

    raw = ALPHA * emb_sims + BETA * dim_sims + GAMMA * val_aligns
    calibrated = np.array([calibrate(float(s)) for s in raw], dtype=np.float32)
    return calibrated


# ══════════════════════════════════════════════════════════════
#  REQUEST MODELS
# ══════════════════════════════════════════════════════════════

class ClusterRequest(BaseModel):
    user_id: str

class CompareRequest(BaseModel):
    user_id:      str
    thought_id:   str
    thought_text: str
    threshold:    float = 0.55


# ══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status":  "alive",
        "version": "3.0.0",
        "model":   "paraphrase-multilingual-MiniLM-L12-v2",
        "method":  "three-way-composite",
        "config": {
            "alpha": ALPHA, "beta": BETA, "gamma": GAMMA,
            "threshold":   EDGE_THRESHOLD,
            "calibration": {"midpoint": CAL_MIDPOINT, "steepness": CAL_STEEPNESS},
            "leiden_resolution": LEIDEN_RES_OVERRIDE or "auto",
            "dimension_count":   len(DIMENSION_COLS),
            "valence_col":       VALENCE_COL,
        }
    }


@app.post("/compare")
async def compare_thought(
    req: CompareRequest,
    x_api_key: str = Header(None),
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    new_emb = model.encode(req.thought_text, normalize_embeddings=True)
    new_emb_list = new_emb.tolist()
    dim_select = ",".join(ALL_DIM_COLS)

    async with httpx.AsyncClient(timeout=60.0) as client:
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/thoughts",
            headers=SB,
            params={"id": f"eq.{req.thought_id}"},
            json={"embedding_vector": new_emb_list},
        )
        r0 = await client.get(
            f"{SUPABASE_URL}/rest/v1/thoughts",
            headers=SB,
            params={"id": f"eq.{req.thought_id}", "select": dim_select, "limit": "1"},
        )
        new_data = r0.json()
        r1 = await client.get(
            f"{SUPABASE_URL}/rest/v1/thoughts",
            headers=SB,
            params={
                "user_id": f"eq.{req.user_id}", "id": f"neq.{req.thought_id}",
                "embedding_vector": "not.is.null",
                "select": f"id,embedding_vector,{dim_select}", "limit": "2000",
            },
        )
        existing = r1.json()

    if not existing:
        return {"status": "ok", "synapses_created": 0, "compared": 0}

    nd = new_data[0] if new_data else {}
    new_valence = get_valence(nd)
    new_dims    = dim_vector(nd)
    emb_matrix  = np.array([t["embedding_vector"] for t in existing], dtype=np.float32)
    valences    = np.array([get_valence(t) for t in existing], dtype=np.float32)
    dim_matrix  = np.array([dim_vector(t) for t in existing], dtype=np.float32)

    scores = compute_composite_scores(new_emb, new_valence, new_dims,
                                      emb_matrix, valences, dim_matrix)

    threshold = req.threshold or EDGE_THRESHOLD
    synapses = []
    for i, score in enumerate(scores):
        if score >= threshold:
            synapses.append({
                "user_id": req.user_id, "thought_a_id": req.thought_id,
                "thought_b_id": existing[i]["id"],
                "similarity_score": round(float(score), 4), "method": "composite-v3",
            })

    if synapses:
        async with httpx.AsyncClient(timeout=60.0) as client:
            await client.post(
                f"{SUPABASE_URL}/rest/v1/synapses",
                headers={**SB, "Prefer": "resolution=ignore-duplicates"},
                json=synapses,
            )

    return {
        "status": "ok", "compared": len(existing),
        "synapses_created": len(synapses),
        "avg_score": round(float(np.mean(scores)), 4),
        "max_score": round(float(np.max(scores)), 4),
        "threshold": threshold, "method": "three-way-composite-v3",
    }


@app.post("/debug_scores")
async def debug_scores(
    req: CompareRequest,
    x_api_key: str = Header(None),
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    new_emb = model.encode(req.thought_text, normalize_embeddings=True)
    dim_select = ",".join(ALL_DIM_COLS)

    async with httpx.AsyncClient(timeout=60.0) as client:
        r0 = await client.get(
            f"{SUPABASE_URL}/rest/v1/thoughts",
            headers=SB,
            params={"id": f"eq.{req.thought_id}", "select": dim_select, "limit": "1"},
        )
        new_data = r0.json()
        r1 = await client.get(
            f"{SUPABASE_URL}/rest/v1/thoughts",
            headers=SB,
            params={
                "user_id": f"eq.{req.user_id}", "id": f"neq.{req.thought_id}",
                "embedding_vector": "not.is.null",
                "select": f"id,raw_thought,embedding_vector,{dim_select}", "limit": "30",
            },
        )
        existing = r1.json()

    if not existing:
        return {"message": "No existing thoughts with embeddings found"}

    nd = new_data[0] if new_data else {}
    new_valence = get_valence(nd)
    new_dims    = dim_vector(nd)
    emb_matrix  = np.array([t["embedding_vector"] for t in existing], dtype=np.float32)
    valences    = np.array([get_valence(t) for t in existing], dtype=np.float32)
    dim_matrix  = np.array([dim_vector(t) for t in existing], dtype=np.float32)

    scores = compute_composite_scores(new_emb, new_valence, new_dims,
                                      emb_matrix, valences, dim_matrix)

    emb_sims = emb_matrix @ new_emb
    new_d_norm = np.linalg.norm(new_dims)
    samples = []
    for i, t in enumerate(existing):
        e_sim = float(emb_sims[i])
        d_sim = dim_cosine(new_dims, dim_vector(t)) if new_d_norm > 1e-9 else 0.0
        v_align = valence_alignment(new_valence, get_valence(t))
        raw = ALPHA * e_sim + BETA * d_sim + GAMMA * v_align
        samples.append({
            "thought": t["raw_thought"][:100],
            "composite": round(float(scores[i]), 4),
            "emb_cosine": round(e_sim, 4),
            "dim_cosine_22": round(d_sim, 4),
            "val_align": round(v_align, 4),
            "valence": round(get_valence(t), 2),
            "raw_composite": round(raw, 4),
        })

    samples.sort(key=lambda x: x["composite"], reverse=True)

    return {
        "config": {
            "alpha": ALPHA, "beta": BETA, "gamma": GAMMA,
            "calibration": {"midpoint": CAL_MIDPOINT, "steepness": CAL_STEEPNESS},
            "dimensions_used": len(DIMENSION_COLS),
            "new_thought_valence": round(new_valence, 2),
        },
        "stats": {
            "min": round(float(np.min(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "avg": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
        },
        "samples": samples,
    }


@app.post("/backfill")
async def backfill_synapses(
    req: ClusterRequest,
    x_api_key: str = Header(None),
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    dim_select = ",".join(ALL_DIM_COLS)

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/thoughts",
            headers=SB,
            params={
                "user_id": f"eq.{req.user_id}",
                "select": f"id,raw_thought,embedding_vector,{dim_select}",
                "order": "created_at.asc", "limit": "10000",
            },
        )
        thoughts = r.json()

    if not thoughts:
        return {"status": "no_thoughts"}

    to_embed = [t for t in thoughts if not t.get("embedding_vector")]
    if to_embed:
        texts = [t["raw_thought"] for t in to_embed]
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=64,
                                  show_progress_bar=False)
        async with httpx.AsyncClient(timeout=120.0) as client:
            for i, t in enumerate(to_embed):
                emb_list = embeddings[i].tolist()
                t["embedding_vector"] = emb_list
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/thoughts",
                    headers=SB,
                    params={"id": f"eq.{t['id']}"},
                    json={"embedding_vector": emb_list},
                )
        print(f"Embedded {len(to_embed)} thoughts")

    total_synapses = 0
    threshold = EDGE_THRESHOLD

    for i in range(1, len(thoughts)):
        t = thoughts[i]
        preceding = [p for p in thoughts[:i] if p.get("embedding_vector")]
        if not preceding:
            continue

        new_emb     = np.array(t["embedding_vector"], dtype=np.float32)
        new_valence = get_valence(t)
        new_dims    = dim_vector(t)
        emb_matrix  = np.array([p["embedding_vector"] for p in preceding], dtype=np.float32)
        valences    = np.array([get_valence(p) for p in preceding], dtype=np.float32)
        dim_matrix  = np.array([dim_vector(p) for p in preceding], dtype=np.float32)

        scores = compute_composite_scores(new_emb, new_valence, new_dims,
                                          emb_matrix, valences, dim_matrix)

        synapses = []
        for j, score in enumerate(scores):
            if score >= threshold:
                synapses.append({
                    "user_id": req.user_id, "thought_a_id": t["id"],
                    "thought_b_id": preceding[j]["id"],
                    "similarity_score": round(float(score), 4), "method": "composite-v3",
                })

        if synapses:
            async with httpx.AsyncClient(timeout=60.0) as client:
                await client.post(
                    f"{SUPABASE_URL}/rest/v1/synapses",
                    headers={**SB, "Prefer": "resolution=ignore-duplicates"},
                    json=synapses,
                )
            total_synapses += len(synapses)

        if (i + 1) % 50 == 0:
            print(f"Backfill: {i+1}/{len(thoughts)} — {total_synapses} synapses")

    print(f"Backfill: {len(thoughts)}/{len(thoughts)} — {total_synapses} synapses")

    return {
        "status": "ok", "thoughts": len(thoughts),
        "newly_embedded": len(to_embed), "synapses_created": total_synapses,
        "threshold": threshold, "method": "three-way-composite-v3",
    }


@app.post("/cleanup")
async def cleanup_old_synapses(
    req: ClusterRequest,
    x_api_key: str = Header(None),
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/synapses",
            headers=SB,
            params={
                "user_id": f"eq.{req.user_id}", "method": "eq.composite-v3",
                "select": "id", "limit": "1",
            },
        )
        check = r.json()

    if not check:
        return {"status": "aborted", "reason": "Geen v3 synapses. Draai eerst /backfill."}

    async with httpx.AsyncClient(timeout=120.0) as client:
        for method in ["cosine", "cross-encoder", "composite"]:
            await client.delete(
                f"{SUPABASE_URL}/rest/v1/synapses",
                headers=SB,
                params={"user_id": f"eq.{req.user_id}", "method": f"eq.{method}"},
            )
        await client.delete(
            f"{SUPABASE_URL}/rest/v1/synapses",
            headers=SB,
            params={"user_id": f"eq.{req.user_id}", "method": "is.null"},
        )

    return {"status": "ok", "message": "Oude synapses verwijderd. Draai nu /cluster."}


@app.post("/cluster")
async def run_clustering(
    req: ClusterRequest,
    x_api_key: str = Header(None),
):
    if x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    async with httpx.AsyncClient(timeout=60.0) as client:
        r1 = await client.get(
            f"{SUPABASE_URL}/rest/v1/synapses",
            headers=SB,
            params={
                "user_id": f"eq.{req.user_id}",
                "select": "thought_a_id,thought_b_id,similarity_score",
                "limit": "500000",
            },
        )
        synapses = r1.json()
        r2 = await client.get(
            f"{SUPABASE_URL}/rest/v1/thoughts",
            headers=SB,
            params={
                "user_id": f"eq.{req.user_id}",
                "select": "id,valence_score,activation_score,agency_score,created_at",
                "limit": "500000",
            },
        )
        thoughts = r2.json()

    if not thoughts:
        return {"status": "no_thoughts", "clusters": 0}

    n = len(thoughts)
    if n == 1:
        async with httpx.AsyncClient(timeout=60.0) as client:
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/thoughts",
                headers=SB,
                params={"id": f"eq.{thoughts[0]['id']}"},
                json={"cluster_id": 1},
            )
        return {"status": "ok", "thoughts": 1, "clusters": 1, "resolution": 0}

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
            weights.append(float(s["similarity_score"]))

    g.add_edges(edges)
    if weights:
        g.es["weight"] = weights

    if not edges:
        cluster_map = {tid: i + 1 for i, tid in enumerate(ids)}
        async with httpx.AsyncClient(timeout=120.0) as client:
            for tid, cid in cluster_map.items():
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/thoughts",
                    headers=SB,
                    params={"id": f"eq.{tid}"},
                    json={"cluster_id": cid},
                )
        return {"status": "ok", "thoughts": n, "clusters": n,
                "resolution": 0, "note": "no synapses found"}

    resolution = auto_resolution(n, len(edges))

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight" if weights else None,
        resolution_parameter=resolution,
        seed=42,
        n_iterations=10,
    )

    communities = sorted(enumerate(partition), key=lambda x: len(x[1]), reverse=True)

    cluster_map = {}
    for new_cid, (_, members) in enumerate(communities, 1):
        for idx in members:
            cluster_map[ids[idx]] = new_cid

    cluster_thoughts = defaultdict(list)
    for t in thoughts:
        cid = cluster_map.get(t["id"])
        if cid:
            cluster_thoughts[cid].append(t)

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
        days_since = (datetime.now(timezone.utc) - latest).days if latest else 30
        recency = 1 / (1 + days_since / 30)
        circuit_strengths[cid] = round(len(members) * avg_intensity * recency, 4)

    async with httpx.AsyncClient(timeout=120.0) as client:
        for tid, cid in cluster_map.items():
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/thoughts",
                headers=SB,
                params={"id": f"eq.{tid}"},
                json={"cluster_id": cid},
            )
        for cid, members in cluster_thoughts.items():
            avg_v = round(
                sum(float(t.get("valence_score") or 0) for t in members) / len(members), 4
            )
            await client.post(
                f"{SUPABASE_URL}/rest/v1/cluster_labels",
                headers={**SB, "Prefer": "resolution=merge-duplicates"},
                json={
                    "cluster_id": cid, "user_id": req.user_id,
                    "thought_count": len(members),
                    "circuit_strength": circuit_strengths[cid],
                    "avg_valence": avg_v,
                },
            )

    return {
        "status": "ok", "thoughts": n,
        "clusters": len(cluster_thoughts),
        "singletons": sum(1 for m in cluster_thoughts.values() if len(m) == 1),
        "method": "leiden-RBConfiguration",
        "resolution": resolution,
        "resolution_mode": "override" if LEIDEN_RES_OVERRIDE else "auto",
        "scoring": "three-way-composite-v3",
    }
