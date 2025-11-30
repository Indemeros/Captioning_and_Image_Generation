import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline
import pickle
import os
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Multimodal GenAI Demo",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #616161;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem;
    }
    .caption-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="main-header">🎨 Multimodal Generative AI Demo</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">CNN + Transformer Image Captioning & Text-to-Image Generation</p>', unsafe_allow_html=True)

# Transformer Decoder Model
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, 
                 max_seq_length=50, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = self._create_positional_encoding(max_seq_length, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_seq_length, embed_dim):
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, image_features, captions, tgt_mask=None):
        batch_size = captions.size(0)
        seq_length = captions.size(1)
        
        # Embed captions and add positional encoding
        embedded = self.embedding(captions) * np.sqrt(self.embedding.embedding_dim)
        pos_encoding = self.positional_encoding[:, :seq_length, :].to(embedded.device)
        embedded = self.dropout(embedded + pos_encoding)
        
        # Decode
        output = self.transformer_decoder(
            tgt=embedded,
            memory=image_features,
            tgt_mask=tgt_mask
        )
        
        output = self.fc_out(output)
        return output
    
    def generate_caption(self, image_features, word_to_idx, idx_to_word, 
                        max_length=50, device='cuda'):
        self.eval()
        with torch.no_grad():
            # Start with <start> token
            start_token = word_to_idx.get('<start>', 1)
            end_token = word_to_idx.get('<end>', 2)
            
            generated = [start_token]
            
            for _ in range(max_length):
                captions_tensor = torch.LongTensor(generated).unsqueeze(0).to(device)
                
                # Generate causal mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    captions_tensor.size(1)
                ).to(device)
                
                output = self.forward(image_features, captions_tensor, tgt_mask)
                
                # Get the last token prediction
                next_token_logits = output[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()
                
                if next_token == end_token:
                    break
                
                generated.append(next_token)
            
            # Convert indices to words
            caption = []
            for idx in generated[1:]:  # Skip <start> token
                word = idx_to_word.get(idx, '<unk>')
                if word not in ['<start>', '<end>', '<pad>']:
                    caption.append(word)
            
            return ' '.join(caption)


# Image Captioning Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6):
        super(ImageCaptioningModel, self).__init__()
        
        # CNN Encoder (ResNet-18)
        resnet = models.resnet18(pretrained=True)
        self.cnn_encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # Adaptive pooling to get fixed-size features
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Project CNN features to embedding dimension
        self.feature_projection = nn.Linear(512 * 7 * 7, embed_dim)
        
        # Transformer Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
    
    def forward(self, images, captions, tgt_mask=None):
        # Extract CNN features
        with torch.no_grad():
            cnn_features = self.cnn_encoder(images)
        
        # Adaptive pooling
        cnn_features = self.adaptive_pool(cnn_features)
        
        # Flatten and project
        batch_size = cnn_features.size(0)
        cnn_features = cnn_features.view(batch_size, -1)
        image_features = self.feature_projection(cnn_features)
        image_features = image_features.unsqueeze(1)  # Add sequence dimension
        
        # Decode
        output = self.decoder(image_features, captions, tgt_mask)
        return output
    
    def generate_caption(self, images, word_to_idx, idx_to_word, device='cuda'):
        self.eval()
        with torch.no_grad():
            # Extract CNN features
            cnn_features = self.cnn_encoder(images)
            cnn_features = self.adaptive_pool(cnn_features)
            
            batch_size = cnn_features.size(0)
            cnn_features = cnn_features.view(batch_size, -1)
            image_features = self.feature_projection(cnn_features)
            image_features = image_features.unsqueeze(1)
            
            # Generate caption
            caption = self.decoder.generate_caption(
                image_features, word_to_idx, idx_to_word, device=device
            )
            
            return caption


@st.cache_resource
def load_captioning_model():
    """Load the trained image captioning model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        if os.path.exists('word_to_idx.pkl') and os.path.exists('idx_to_word.pkl'):
            with open('word_to_idx.pkl', 'rb') as f:
                word_to_idx = pickle.load(f)
            with open('idx_to_word.pkl', 'rb') as f:
                idx_to_word = pickle.load(f)
        else:
            st.warning("Vocabulary files not found. Using dummy vocabulary.")
            # Create dummy vocabulary for demo purposes
            word_to_idx = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
            idx_to_word = {0: '<pad>', 1: '<start>', 2: '<end>', 3: '<unk>'}
        
        vocab_size = len(word_to_idx)
        
        # Initialize model
        model = ImageCaptioningModel(
            vocab_size=vocab_size,
            embed_dim=512,
            num_heads=8,
            num_layers=6
        ).to(device)
        
        # Load trained weights if available
        if os.path.exists('best_caption_model.pth'):
            model.load_state_dict(torch.load('best_caption_model.pth', map_location=device))
            st.success("✅ Captioning model loaded successfully!")
        else:
            st.warning("⚠️ Trained model not found. Using untrained model for demo.")
        
        model.eval()
        return model, word_to_idx, idx_to_word, device
    
    except Exception as e:
        st.error(f"Error loading captioning model: {str(e)}")
        return None, None, None, None


@st.cache_resource
def load_text_to_image_model():
    """Load the pretrained text-to-image generation model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Stable Diffusion (use a smaller version for efficiency)
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        
        st.success("✅ Text-to-Image model loaded successfully!")
        return pipe, device
    
    except Exception as e:
        st.error(f"Error loading text-to-image model: {str(e)}")
        st.info("💡 Tip: Make sure you have enough GPU memory and have accepted the model license on HuggingFace.")
        return None, None


def preprocess_image(image):
    """Preprocess image for the captioning model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application demonstrates:
    
    1. **Image Captioning**: Upload an image and get an AI-generated caption
       - Uses ResNet-18 CNN encoder
       - Transformer decoder with attention
    
    2. **Text-to-Image**: Enter a text prompt and generate an image
       - Uses Stable Diffusion model
       - High-quality image synthesis
    """)
    
    st.header("⚙️ Settings")
    device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.info(f"Running on: **{device_info}**")
    
    if st.button("🔄 Reload Models"):
        st.cache_resource.clear()
        st.rerun()

# Main content
tab1, tab2 = st.tabs(["📸 Image Captioning", "🎨 Text-to-Image"])

# Tab 1: Image Captioning
with tab1:
    st.header("Upload an Image for Captioning")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg'],
        key="caption_uploader"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔮 Generate Caption", key="gen_caption"):
                with st.spinner("Generating caption..."):
                    model, word_to_idx, idx_to_word, device = load_captioning_model()
                    
                    if model is not None:
                        # Preprocess image
                        image_tensor = preprocess_image(image).to(device)
                        
                        # Generate caption
                        caption = model.generate_caption(
                            image_tensor, 
                            word_to_idx, 
                            idx_to_word, 
                            device=device
                        )
                        
                        # Display result
                        with col2:
                            st.markdown("### Generated Caption")
                            st.markdown(f'<div class="caption-box"><p style="font-size: 1.2rem; font-weight: bold;">{caption}</p></div>', 
                                      unsafe_allow_html=True)
                    else:
                        st.error("Failed to load the captioning model.")

# Tab 2: Text-to-Image Generation
with tab2:
    st.header("Generate Image from Text")
    
    text_prompt = st.text_area(
        "Enter your prompt:", 
        placeholder="Example: A beautiful sunset over mountains with a lake in the foreground",
        height=100
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        num_inference_steps = st.slider(
            "Inference Steps", 
            min_value=10, 
            max_value=100, 
            value=50,
            help="More steps = better quality but slower"
        )
        
        guidance_scale = st.slider(
            "Guidance Scale", 
            min_value=1.0, 
            max_value=20.0, 
            value=7.5,
            step=0.5,
            help="How closely to follow the prompt"
        )
    
    if st.button("🎨 Generate Image", key="gen_image"):
        if text_prompt:
            with st.spinner("Generating image... This may take a minute..."):
                pipe, device = load_text_to_image_model()
                
                if pipe is not None:
                    try:
                        # Generate image
                        image = pipe(
                            text_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale
                        ).images[0]
                        
                        # Display result
                        with col2:
                            st.markdown("### Generated Image")
                            st.image(image, use_container_width=True)
                            
                            # Download button
                            buf = BytesIO()
                            image.save(buf, format="PNG")
                            byte_im = buf.getvalue()
                            
                            st.download_button(
                                label="⬇️ Download Image",
                                data=byte_im,
                                file_name="generated_image.png",
                                mime="image/png"
                            )
                    
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")
                else:
                    st.error("Failed to load the text-to-image model.")
        else:
            st.warning("Please enter a text prompt first!")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #616161;'>
        <p>🎓 Master-Level Project: Multimodal Generative AI</p>
        <p>Built with PyTorch, Transformers, and Stable Diffusion</p>
    </div>
""", unsafe_allow_html=True)
