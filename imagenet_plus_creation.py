import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from stable_diffusion import StableDiffusionModel
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import polynomial_kernel
import clip

# Constants
IMAGE_NET_CLASSES = [...]  # List of ImageNet 1k classes
IMAGES_PER_CLASS = 100
FID_THRESHOLD = 50.0
CLIP_THRESHOLD = 0.3
IS_THRESHOLD = 2.5
IMAGE_SIZE = 256
BATCH_SIZE = 8

# Initialize Stable Diffusion model
sd_model = StableDiffusionModel()

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Directories for saving images
output_dir = "generated_dataset"
os.makedirs(output_dir, exist_ok=True)

# Helper functions
def generate_images(class_name, num_images):
    images = []
    for _ in range(num_images):
        prompt = f"A photo of a {class_name}"
        image = sd_model.generate_image(prompt)
        images.append(image)
    return images

def calculate_fid_score(real_images, generated_images):
    real_features = extract_features(real_images)
    generated_features = extract_features(generated_images)
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def extract_features(images):
    images = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(images).cpu().numpy()
    return features

def calculate_clip_score(image, class_name):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = clip.tokenize([f"A photo of a {class_name}"]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
    return (image_features @ text_features.T).item()

def calculate_is_score(images):
    # Assuming inception model is already loaded and preprocessed
    inception_features = extract_inception_features(images)
    kl_div = inception_features * (np.log(inception_features) - np.log(inception_features.mean(axis=0)))
    is_score = np.exp(kl_div.sum(axis=1).mean())
    return is_score

def extract_inception_features(images):
    # Placeholder for inception feature extraction
    return np.random.rand(len(images), 1000)

# Generation and evaluation
for class_name in IMAGE_NET_CLASSES:
    print(f"Generating images for class: {class_name}")
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    generated_images = generate_images(class_name, IMAGES_PER_CLASS)
    fid_scores, clip_scores, is_scores = [], [], []
    
    for image in generated_images:
        image_path = os.path.join(class_dir, f"{len(fid_scores)}.png")
        image.save(image_path)
        
        fid_score = calculate_fid_score(real_images, [image])
        clip_score = calculate_clip_score(image, class_name)
        is_score = calculate_is_score([image])
        
        fid_scores.append(fid_score)
        clip_scores.append(clip_score)
        is_scores.append(is_score)
    
    selected_images = [
        image for image, fid, clip, is_ in zip(generated_images, fid_scores, clip_scores, is_scores)
        if fid <= FID_THRESHOLD and clip >= CLIP_THRESHOLD and is_ >= IS_THRESHOLD
    ]
    
    for i, image in enumerate(selected_images):
        image.save(os.path.join(class_dir, f"selected_{i}.png"))

print("Image generation and evaluation complete.")
