# download_models.py
# Run this once on the host (with internet) to fetch model weights into ./model/
# so the Docker image can be built and run fully offline.
from huggingface_hub import snapshot_download

MODELS = {
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
    "sam2.1-hiera-large": "facebook/sam2.1-hiera-large",
    "sam3": "facebook/sam3",
}

for local_name, repo_id in MODELS.items():
    print(f"Downloading {repo_id} -> model/{local_name}")
    snapshot_download(repo_id=repo_id, local_dir=f"model/{local_name}")

print("Done.")
