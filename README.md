# AI-Powered Medical Image Restoration using Diffusion Models

***

A cutting-edge diffusion-based image restoration system that enhances degraded chest X-ray images to improve diagnostic accuracy. This project leverages generative AI and UNet architecture to recover high-quality medical images from inputs affected by noise, blur, and imaging artifacts. 

**Live Notebook:** [Google Colab Demo](https://colab.research.google.com/drive/1KSnNWeA_sHc-IRAfB-M_XARPKJd0dn5j?usp=sharing)

## Problem Statement

Chest X-ray images are frequently degraded by noise, blur, low contrast, and imaging artifacts, which can obscure critical clinical details and negatively impact diagnosis. Conventional image enhancement methods struggle to handle diverse and complex degradations effectively.

This project addresses these challenges by developing a **diffusion-based image restoration model** that can:
- Recover high-quality chest X-ray images from degraded inputs
- Preserve critical diagnostic features
- Handle multiple types of degradation simultaneously
- Achieve superior performance metrics (PSNR: 32.98 dB, SSIM: 0.9225)

## Real-World Applications

- **Medical Diagnostics**: Enhancing low-quality X-rays for accurate diagnosis
- **Legacy Archive Restoration**: Improving old medical imaging records
- **Telemedicine**: Compensating for poor imaging equipment in remote areas
- **Emergency Medicine**: Quick enhancement of suboptimal emergency room scans
- **Research**: Creating clean datasets from degraded historical medical images

## Key Features

- **Diffusion-Based Architecture**: State-of-the-art generative AI approach for image restoration
- **UNet Backbone**: Encoder-decoder architecture with skip connections for detail preservation 
- **Iterative Denoising**: Progressive noise removal through learned reverse diffusion process 
- **Multi-Degradation Handling**: Addresses noise, blur, low contrast, and artifacts simultaneously
- **High Performance**: Achieves PSNR of 32.98 dB and SSIM of 0.9225
- **GPU-Accelerated**: Optimized for training on Google Colab with GPU support

## Architecture Overview

### Diffusion Process

The model uses a **UNet-based diffusion architecture** to progressively remove noise from images through learned reverse diffusion: 

```
Input Image → Noise Addition (Forward Diffusion) → UNet Denoising Network → 
Noise Prediction → Restored Image Output
```

### Model Components

1. **Encoder**: Downsamples the image while extracting hierarchical features
2. **Bottleneck**: Processes compressed representation at lowest resolution
3. **Decoder**: Upsamples while reconstructing fine details
4. **Skip Connections**: Preserves spatial information across network depth 

### Diffusion-Based Restoration

The restoration process utilizes:
- **Forward Diffusion**: Gradually adds noise to learn the noise distribution
- **Reverse Diffusion**: Iteratively removes noise to restore clean images
- **UNet Denoising**: Predicts and removes noise at each timestep

This approach enables the model to learn intricate details and effectively enhance image clarity during restoration. 

## Dataset

**Source**: Chest X-Ray Images (Pneumonia) - Paul Mooney (Kaggle)

### Dataset Statistics
- **Total Images**: 5,863 chest X-ray images in JPEG format 
- **Splits**:
  - Training set
  - Validation set
  - Testing set
- **Organization**: Separate folders for Pneumonia and Normal images in each subset 

### Preprocessing Pipeline
1. Image loading and normalization
2. Degradation simulation (blur, noise, artifacts)
3. Tensor conversion and augmentation
4. Batch preparation for training

## 🛠️ Technology Stack

### Core Technologies
- **Python** – Core programming language for implementation
- **PyTorch** – Building and training the neural network model
- **HuggingFace Diffusers** – Implementing diffusion model and noise scheduler 

### Deep Learning Framework
- **UNet2DModel** – Backbone architecture for denoising and image restoration 
- **Torchvision** – Image preprocessing and transformations 

### Image Processing
- **OpenCV (cv2)** – Simulating blur and imaging artifacts 
- **Pillow (PIL)** – Loading and handling X-ray images 
- **NumPy** – Numerical computations and array operations 

### Visualization & Evaluation
- **Matplotlib** – Visualization of images and training curves
- **Scikit-image** – Image quality evaluation using PSNR and SSIM metrics 

### Infrastructure
- **Kaggle API** – Downloading the chest X-ray dataset 
- **Google Colab** – GPU-enabled environment for training and experimentation

## Getting Started

### Prerequisites
- Python 3.8+
- GPU recommended (NVIDIA CUDA-compatible)
- Google Colab account (for notebook version)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-image-restoration.git
cd medical-image-restoration

# Install dependencies
pip install torch torchvision
pip install diffusers transformers accelerate
pip install opencv-python pillow numpy matplotlib scikit-image
pip install kaggle

# Download dataset (requires Kaggle API credentials)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip
```

### Running the Notebook

**Google Colab**
1. Open the [Colab notebook](https://colab.research.google.com/drive/1KSnNWeA_sHc-IRAfB-M_XARPKJd0dn5j?usp=sharing)
2. Connect to GPU runtime (Runtime → Change runtime type → GPU)
3. Run cells sequentially


## Training Process

### Training Configuration
- **Epochs**: 15 epochs shown in training curve 
- **Loss Function**: MSE (Mean Squared Error) for noise prediction
- **Optimizer**: Adam optimizer
- **Learning Rate**: Adaptive with scheduler
- **Batch Processing**: GPU-optimized batching

### Training Progress

The model demonstrates consistent convergence with decreasing loss over epochs: 
- **Training Loss**: Decreases from ~0.012 to ~0.004
- **Validation Loss**: Closely tracks training loss, indicating good generalization

## Performance Metrics

### Quantitative Results
- **PSNR (Peak Signal-to-Noise Ratio)**: 32.98 dB 
  - Measures pixel-level reconstruction accuracy
  - Higher is better (typical range: 20-40 dB)
  
- **SSIM (Structural Similarity Index)**: 0.9225 
  - Measures perceptual image quality
  - Range: 0-1 (1 = perfect similarity)

### Qualitative Comparison
The model produces visually superior results compared to degraded inputs, closely matching ground truth quality: 
- **Degraded**: Noisy, blurred input images
- **Restored**: Clean, enhanced outputs
- **Ground Truth**: Original high-quality reference images

## How It Works

### 1. Forward Diffusion (Training)
```python
# Add noise progressively to clean images
noisy_image = clean_image + noise_schedule(timestep) * noise
```

### 2. Noise Prediction (UNet)
```python
# UNet predicts the noise added at each timestep
predicted_noise = unet_model(noisy_image, timestep)
```

### 3. Reverse Diffusion (Inference)
```python
# Iteratively remove predicted noise
for t in reversed(timesteps):
    noise_pred = model(noisy_image, t)
    noisy_image = denoise_step(noisy_image, noise_pred, t)
restored_image = noisy_image
```

### 4. Quality Evaluation
```python
# Compute metrics
psnr = peak_signal_noise_ratio(restored, ground_truth)
ssim = structural_similarity(restored, ground_truth)
```

## Visualizations

The notebook includes comprehensive visualizations:
- **Training/Validation Loss Curves**: Monitor convergence 
- **Side-by-Side Comparisons**: Degraded vs Restored vs Ground Truth 
- **Metric Tracking**: PSNR and SSIM progression
- **Denoising Steps**: Intermediate outputs during restoration

## Experimental Setup

### Degradation Simulation
To create training pairs, the following degradations are applied to clean X-rays:
- **Gaussian Noise**: Simulates sensor noise
- **Motion Blur**: Mimics patient movement
- **Low Contrast**: Replicates poor imaging conditions
- **Compression Artifacts**: JPEG compression effects

### Data Augmentation
- Random rotations
- Horizontal flips
- Intensity variations
- Crop and resize

## Key Concepts Explained

### Diffusion Models
Diffusion models work by:
1. **Learning to destroy**: Add noise progressively (forward process)
2. **Learning to restore**: Remove noise step-by-step (reverse process)
3. **Generative capability**: Can generate new samples by starting from pure noise

### UNet Architecture
- **Encoder Path**: Downsamples image, captures context
- **Decoder Path**: Upsamples features, reconstructs details 
- **Skip Connections**: Preserves fine details lost during downsampling 

### Why Diffusion for Medical Imaging?
- Superior detail preservation compared to GANs
- More stable training than VAEs
- Better handling of complex degradations
- Proven success in image-to-image translation tasks

## Future Enhancements

### Model Improvements
- **Conditional Diffusion**: Guide restoration based on pathology type
- **Attention Mechanisms**: Focus on diagnostically important regions
- **Multi-Scale Processing**: Handle various image resolutions
- **Ensemble Approaches**: Combine multiple denoising strategies

### Dataset Expansion
- Multi-modality support (CT, MRI, ultrasound)
- Larger diverse datasets for better generalization
- Real degraded images vs simulated degradations

### Deployment
- Web-based interface for radiologists
- Real-time processing pipeline
- DICOM format support
- Integration with PACS systems
- Mobile application for field use

### Evaluation
- Clinical validation with radiologists
- Diagnostic accuracy comparison studies
- Blind quality assessment tests
- Edge case analysis

## Research Background

This project builds upon recent advances in:
- **Denoising Diffusion Probabilistic Models (DDPM)**
- **Score-based Generative Models**
- **Image-to-Image Translation with Diffusion**
- **Medical Image Enhancement using Deep Learning**

## Limitations & Considerations

### Current Limitations
- Training requires significant GPU resources
- Inference time longer than traditional methods
- Primarily tested on chest X-rays (generalization to other modalities untested)
- Simulated degradations may not fully match real-world artifacts


## Acknowledgments

- **Dataset**: Paul Mooney for the Chest X-Ray Images (Pneumonia) dataset on Kaggle 
- **Framework**: HuggingFace team for the Diffusers library
- **Architecture**: UNet architecture researchers
- **Community**: PyTorch and medical imaging research communities


## Visual Results

### Architecture Diagram
<img width="960" height="530" alt="image" src="https://github.com/user-attachments/assets/4cab43a7-5dea-41ee-9733-d46e08602b6c" />

### Training Progress
<img width="1533" height="821" alt="image" src="https://github.com/user-attachments/assets/8bf9e421-f2b8-4012-b408-8d41984cad37" />

### Restoration Results
<img width="1700" height="633" alt="image" src="https://github.com/user-attachments/assets/89552e31-e500-4869-8558-87557be4d976" />


**Metrics**: PSNR: 32.98 dB | SSIM: 0.9225  
***
