# ImageNet+ : A Synthetic Version of ImageNet-1k

## Introduction

In this document, we will describe the process of creating a dataset using a diffusion model to generate images belonging to different classes. Specifically, we will discuss the use of Stable Diffusion to create a parallel version of the ImageNet 1k dataset, containing only images generated using Stable Diffusion. We will also explain the evaluation of these generated images using FID score, CLIP score, and IS score to select only the images that pass a specified threshold.

## Step 1: Generating Images Using Stable Diffusion

### 1.1 What is Stable Diffusion?

Stable Diffusion is a powerful image generation model that uses the principles of diffusion processes to create high-quality images from text prompts.

### 1.2 Setting Up Stable Diffusion

To generate images, first, set up the Stable Diffusion environment:
- Install necessary libraries and dependencies.
- Download pre-trained Stable Diffusion models.
- Configure the model for image generation.

### 1.3 Generating Images for Different Classes

For each class in the ImageNet 1k dataset:
- Define a text prompt that describes the class.
- Use Stable Diffusion to generate multiple images for each prompt.
- Save the generated images in a structured format, corresponding to their respective classes.

## Step 2: Creating a Parallel Version of ImageNet 1k

### 2.1 Overview

The goal is to create a parallel version of ImageNet 1k, where each original image is replaced by an image generated using Stable Diffusion.

### 2.2 Mapping Classes

Ensure that the generated images are correctly mapped to the respective classes:
- Maintain a directory structure similar to ImageNet 1k.
- Store images in folders named after their classes.

### 2.3 Ensuring Diversity and Quality

Generate multiple images per class and select the most diverse and high-quality images:
- Use random seeds and variations in prompts to increase diversity.
- Pre-filter images based on visual inspection and basic quality checks.

## Step 3: Evaluating Generated Images

### 3.1 Evaluation Metrics

To ensure the quality of the generated dataset, we use three primary evaluation metrics:
- **FID Score** (Frechet Inception Distance): Measures the similarity between the generated images and real images.
- **CLIP Score**: Evaluates how well the generated images align with their corresponding text descriptions.
- **IS Score** (Inception Score): Measures the diversity and quality of the generated images.

### 3.2 Calculating FID Score

- Compute the FID score by comparing the distribution of features from the generated images with those from the original ImageNet images.
- Use a pre-trained Inception model to extract features.

### 3.3 Calculating CLIP Score

- Use the CLIP model to evaluate the correspondence between generated images and their text descriptions.
- Generate scores based on the alignment of image-text pairs.

### 3.4 Calculating IS Score

- Use the Inception model to classify generated images and compute the IS score based on the predicted class probabilities.
- Ensure a high IS score to confirm that the images are both diverse and high-quality.

## Step 4: Selecting Images Based on Thresholds

### 4.1 Defining Thresholds

Set thresholds for FID score, CLIP score, and IS score to filter out low-quality images:
- FID score threshold: Lower values indicate better quality.
- CLIP score threshold: Higher values indicate better alignment with descriptions.
- IS score threshold: Higher values indicate better diversity and quality.

### 4.2 Filtering Images

- Evaluate each generated image using the three metrics.
- Select images that pass all the thresholds for inclusion in the final dataset.

### 4.3 Finalizing the Dataset

- Compile the selected images into the final dataset.
- Ensure the dataset structure matches that of ImageNet 1k for consistency.

## Conclusion

By following these steps, we can create a high-quality, parallel version of the ImageNet 1k dataset using images generated with Stable Diffusion. This dataset can be used for various machine learning tasks, ensuring the images meet strict quality standards through rigorous evaluation.
