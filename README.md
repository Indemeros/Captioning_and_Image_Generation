# Multimodal Generative AI - Streamlit App

A professional Streamlit application for image captioning and text-to-image generation using CNN encoders and Transformer decoders.

## Features

### 1. Image Captioning
- Upload any image
- Get AI-generated captions using ResNet-18 + Transformer architecture
- Real-time inference

### 2. Text-to-Image Generation
- Enter text prompts
- Generate high-quality images using Stable Diffusion
- Adjustable inference steps and guidance scale
- Download generated images

## Installation

### Prerequisites
- Python 3.8-3.10
- CUDA-capable GPU (recommended, but CPU works too)
- At least 8GB RAM (16GB recommended)

### Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download model files** (if you have trained models):
   - Place `best_caption_model.pth` in the app directory
   - Place `word_to_idx.pkl` and `idx_to_word.pkl` in the app directory

3. **HuggingFace Token** (for Stable Diffusion):
   - Create an account at https://huggingface.co
   - Accept the license for Stable Diffusion v1.4
   - Login via CLI: `huggingface-cli login`

## Running the App

### Local Development
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Cloud Deployment

#### Option 1: Streamlit Cloud
1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Deploy from your repository
4. Add HuggingFace token in Secrets

#### Option 2: Heroku
```bash
heroku create your-app-name
git push heroku main
```

## Usage Guide

### Image Captioning Tab
1. Click on "📸 Image Captioning" tab
2. Upload an image (PNG, JPG, or JPEG)
3. Click "🔮 Generate Caption"
4. View the AI-generated caption

### Text-to-Image Tab
1. Click on "🎨 Text-to-Image" tab
2. Enter a descriptive text prompt
3. Adjust inference steps (10-100) and guidance scale (1-20)
4. Click "🎨 Generate Image"
5. Download the generated image if desired

## Model Architecture

### Image Captioning
- **Encoder**: ResNet-18 pretrained on ImageNet
- **Decoder**: 6-layer Transformer with:
  - 8 attention heads
  - 512 embedding dimensions
  - Positional encoding
  - Cross-attention mechanism

### Text-to-Image
- **Model**: Stable Diffusion v1.4
- **Architecture**: Latent Diffusion Model
- **Resolution**: 512x512 pixels

## File Structure
```
.
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── best_caption_model.pth     # Trained captioning model (optional)
├── word_to_idx.pkl            # Vocabulary mapping (optional)
└── idx_to_word.pkl            # Reverse vocabulary mapping (optional)
```

## Hardware Requirements

### Minimum
- CPU with 8GB RAM
- Can run both models (slowly)

### Recommended
- NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- 16GB system RAM
- Significantly faster inference

### Optimal
- NVIDIA GPU with 12GB+ VRAM (RTX 3080 or better)
- 32GB system RAM
- Real-time generation

## Troubleshooting

### Out of Memory Errors
- Reduce inference steps for text-to-image
- Use CPU instead of GPU (slower but works)
- Close other applications

### Model Loading Issues
- Ensure you have internet connection for first-time download
- Check HuggingFace authentication
- Verify CUDA installation if using GPU

### Slow Performance
- First generation is always slower (model loading)
- Subsequent generations are faster (cached)
- Consider using GPU for better performance

## Performance Tips

1. **First Run**: Models are downloaded and cached (~4GB for Stable Diffusion)
2. **GPU Acceleration**: Automatic if CUDA is available
3. **Caching**: Models are cached in memory for faster inference
4. **Batch Processing**: Not implemented, process one at a time

## Metrics & Evaluation

The training notebook includes evaluation using:
- **BLEU**: Measures n-gram overlap
- **METEOR**: Considers synonyms and stemming
- **CLIPScore**: Measures semantic similarity

## Future Enhancements

- Batch image processing
- Multiple caption generation with beam search
- Advanced text-to-image parameters (seed, negative prompts)
- Image editing capabilities
- Video captioning
- Multi-language support

## Credits

- **Datasets**: Flickr8k
- **Models**: ResNet-18 (torchvision), Stable Diffusion (CompVis)
- **Libraries**: PyTorch, HuggingFace Transformers & Diffusers, Streamlit

## License

This project is for educational purposes as part of a Master-level Deep Learning course.

## Contact

For questions or issues, please refer to the course materials or contact your instructor.

---

**Note**: Make sure to place your trained model files (`best_caption_model.pth`, `word_to_idx.pkl`, `idx_to_word.pkl`) in the same directory as `app.py` before running for best results. The app will work without them but with limited captioning capability.
