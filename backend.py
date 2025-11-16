

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from pydantic import BaseModel

from contextlib import asynccontextmanager

# ----------------------------
# App Initialization
# ----------------------------

MODEL_CHECKPOINT = "vae_checkpoints/vqvae_epoch_20.pt"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vqvae_model, pca_components, pca_mean, latent_vectors, latent_indices

    # Load model
    checkpoint = torch.load(MODEL_CHECKPOINT, map_location=device)

    from vq_vae import VQVAE
    vqvae_model = VQVAE(
        embedding_dim=checkpoint["config"]["embedding_dim"],
        num_embeddings=checkpoint["config"]["codebook_size"],
        commitment_cost=checkpoint["config"]["beta"],
        decay=checkpoint["config"]["ema_decay"],
    ).to(device)
    vqvae_model.encoder.load_state_dict(checkpoint["encoder"])
    vqvae_model.vq_vae.load_state_dict(checkpoint["quantizer"])
    vqvae_model.decoder.load_state_dict(checkpoint["decoder"])
    vqvae_model.eval()

    # Load PCA & latents
    try:
        pca_components = np.load("./pca_components.npy")
        pca_mean = np.load("./pca_mean.npy")
        latent_vectors = np.load("./latent_vectors.npy")
        latent_indices = np.load("./latent_indices.npy")
    except Exception as e:
        print("Warning: PCA or latent data missing:", e)

    yield  # startup complete

    # (Optional cleanup) — currently nothing to clean

app = FastAPI(lifespan=lifespan)

# CORS for frontend (React / Three.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Latent + PCA globals (loaded once at startup)
# ----------------------------
vqvae_model = None
pca_components = None   # shape (2, D)
pca_mean = None         # shape (D,)
latent_vectors = None   # shape (N, D)
latent_indices = None   # shape (N, 16, 16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Request Models
# ----------------------------
class DecodeRequest(BaseModel):
    latent_vector: list  # flattened latent vector, length D

class PCARequest(BaseModel):
    x: float
    y: float


# ----------------------------
# Health check
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ----------------------------
# Return PCA scatter data
# ----------------------------
@app.get("/pca")
def get_pca():
    if latent_vectors is None or pca_components is None:
        return {"error": "PCA data not loaded"}
    # project all stored latent vectors into 2D for visualization
    centered = latent_vectors - pca_mean
    xy = centered @ pca_components.T
    return {"xy": xy.tolist()}

# ----------------------------
# Decode a latent vector into an image
# ----------------------------
@app.post("/decode")
def decode(req: DecodeRequest):
    lv = np.array(req.latent_vector)
    lv = lv.reshape(16, 16)
    code = torch.tensor(lv, dtype=torch.long, device=device)

    with torch.no_grad():
        if (not vqvae_model):
            raise Exception
        img = vqvae_model.decode_code(code).squeeze(0).cpu().numpy()
    img = np.transpose(img, (1,2,0))  # HWC

    return {"image": img.tolist()}

# ----------------------------
# Convert clicked PCA coordinates back to latent vector
# ----------------------------
@app.post("/pca_to_latent")
def pca_to_latent(req: PCARequest):
    if pca_components is None:
        return {"error": "PCA not loaded"}

    xy = np.array([req.x, req.y])
    latent = pca_mean + xy @ pca_components  # inverse mapping
    if latent_vectors is None:
        raise Exception
    latent = np.clip(np.round(latent), 0, latent_vectors.max())  # enforce codebook range

    return {"latent_vector": latent.tolist()}

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)