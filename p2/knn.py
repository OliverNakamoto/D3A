import torch
import faiss
import numpy as np
import os

# Load embeddings
curated_data = torch.load('curated_embeddings.pt', weights_only=True)
uncurated_data = torch.load('uncurated_embeddings.pt', weights_only=True)

curated_embeddings = curated_data['embeddings'].numpy()
uncurated_embeddings = uncurated_data['embeddings'].numpy()

curated_paths = curated_data['paths']
uncurated_paths = uncurated_data['paths']

# Build FAISS index
dimension = curated_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(uncurated_embeddings)

# For each curated image, find the 4 nearest uncurated images
k = 4  # Number of nearest neighbors
_, indices = index.search(curated_embeddings, k)

# Collect nearest neighbor image paths
retrieved_image_paths = set()
for idx_list in indices:
    for idx in idx_list:
        retrieved_image_paths.add(uncurated_paths[idx])

# Save retrieved images to a new folder
os.makedirs('retrieved_dataset', exist_ok=True)
for path in retrieved_image_paths:
    filename = os.path.basename(path)
    dest_path = os.path.join('retrieved_dataset', filename)
    if not os.path.exists(dest_path):
        os.link(path, dest_path)  # Create a hard link to save space

print(f"Retrieved {len(retrieved_image_paths)} images.")

import shutil

# Create final dataset directory
os.makedirs('final_dataset', exist_ok=True)

# Copy curated images
for path in curated_paths:
    filename = os.path.basename(path)
    dest_path = os.path.join('final_dataset', filename)
    if not os.path.exists(dest_path):
        shutil.copy(path, dest_path)

# Copy retrieved images
for path in retrieved_image_paths:
    filename = os.path.basename(path)
    dest_path = os.path.join('final_dataset', filename)
    if not os.path.exists(dest_path):
        shutil.copy(path, dest_path)

print(f"Final dataset contains {len(os.listdir('final_dataset'))} images.")
