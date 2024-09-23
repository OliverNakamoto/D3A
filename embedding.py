import os
import torch
import clip
from PIL import Image
from tqdm import tqdm

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def compute_embeddings(image_folder):
    embeddings = []
    image_paths = []
    for filename in tqdm(os.listdir(image_folder)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image_path = os.path.join(image_folder, filename)
        try:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = model.encode_image(image)
            embeddings.append(embedding.cpu())
            image_paths.append(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    embeddings = torch.vstack(embeddings)
    return embeddings, image_paths

# Compute embeddings for curated and uncurated datasets
curated_embeddings, curated_paths = compute_embeddings('curated_dataset')
uncurated_embeddings, uncurated_paths = compute_embeddings('uncurated_dataset')

# Save embeddings for later use
torch.save({
    'embeddings': curated_embeddings,
    'paths': curated_paths
}, 'curated_embeddings.pt')

torch.save({
    'embeddings': uncurated_embeddings,
    'paths': uncurated_paths
}, 'uncurated_embeddings.pt')
