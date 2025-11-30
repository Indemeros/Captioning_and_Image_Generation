# app.py

import streamlit as st
from PIL import Image
import torch
from transformers import AutoTokenizer, VisionEncoderDecoderModel
from diffusers import StableDiffusionPipeline
import io
import os

# --- Configuration ---
st.set_page_config(
    page_title="Multimodal GenAI: Image Captioning & Text-to-Image",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧠 Multimodal Generative AI Project")
st.markdown("---")

# ----------------------------------------------------
# 1. Image Captioning Model (CNN Encoder + Transformer Decoder)
#    (Trained on Flickr8k Dataset)
# ----------------------------------------------------

# IMPORTANT: Replace these paths with your actual trained model files from the Colab notebook.
# This function uses a placeholder structure based on your architecture (CNN+Transformer).
CAPTIONING_MODEL_PATH = "path/to/your/flickr8k_captioning_model.pth"
CAPTIONING_VOCAB_PATH = "path/to/your/flickr8k_tokenizer_vocab.pkl"

@st.cache_resource
def load_captioning_model_and_tokenizer():
    """
    Loads the custom Image Captioning model (ResNet/MobileNet + Transformer).
    
    NOTE: Since the actual weights are not provided, this function uses a dummy
    Hugging Face model structure as a placeholder for a functional demo.
    YOU MUST REPLACE THIS WITH YOUR CUSTOM IMPLEMENTATION.
    """
    try:
        # Placeholder for your custom Flickr8k-trained model
        st.info(f"Loading custom CNN-Transformer model (Expected path: {CAPTIONING_MODEL_PATH})...")
        
        # --- YOUR CUSTOM MODEL LOADING LOGIC GOES HERE ---
        # 1. Load the Vocabulary/Tokenizer (e.g., using pickle for the word-to-index mapping)
        # 2. Instantiate your CNN-Transformer Model class (e.g., from model.py)
        # 3. Load the state dict: model.load_state_dict(torch.load(CAPTIONING_MODEL_PATH))
        
        # For a truly runnable demo without your weights, we'll use a pre-trained
        # image-to-text model from HF as a functional stand-in for demonstration purposes.
        # Replace this with your actual model class and weight loading:
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        return model, tokenizer
    
    except Exception as e:
        st.error(f"Error loading custom captioning model. Please ensure files are at: {CAPTIONING_MODEL_PATH}. Using a dummy object for demo.")
        st.warning("To run this, you need to replace the placeholder logic with your model's loading code or ensure a fallback model is available.")
        return None, None


@st.cache_data(show_spinner=False)
def generate_caption(image_bytes, model, tokenizer):
    """
    Generates a caption for the uploaded image using the loaded model.
    """
    if model is None or tokenizer is None:
        return "Model not loaded. Please check model paths."

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # --- YOUR CUSTOM PREPROCESSING AND INFERENCE LOGIC GOES HERE ---
        # 1. Preprocess the image (resize, normalize, convert to tensor).
        # 2. Run inference on your model (e.g., using beam search or greedy decoding).
        # 3. Decode the generated token IDs back into a sentence using your vocabulary.

        # Placeholder inference using the fallback Hugging Face model:
        pixel_values = tokenizer.image_processor(image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values, max_length=50, num_beams=5)
        caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return caption
    
    except Exception as e:
        st.error(f"Caption generation error: {e}")
        return "Failed to generate caption."


# ----------------------------------------------------
# 2. Text-to-Image Generation (Pretrained Generator Integration)
# ----------------------------------------------------

# Recommended Text-to-Image model for integration (Stable Diffusion)
T2I_MODEL = "runwayml/stable-diffusion-v1-5"

@st.cache_resource
def load_text_to_image_model():
    """
    Loads the pretrained Text-to-Image pipeline (Hugging Face Diffusers).
    """
    st.info(f"Loading pretrained Text-to-Image model: {T2I_MODEL}...")
    try:
        # You may need to specify a device if running on a powerful machine: 
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        
        pipeline = StableDiffusionPipeline.from_pretrained(T2I_MODEL, torch_dtype=torch.float32)
        pipeline = pipeline.to(device)
        return pipeline
    except Exception as e:
        st.error(f"Error loading Text-to-Image model: {e}")
        return None

# --- Main Streamlit App Layout ---

# Load models outside of the main function using cache
caption_model, caption_tokenizer = load_captioning_model_and_tokenizer()
t2i_pipeline = load_text_to_image_model()


tab1, tab2 = st.tabs(["🖼️ Image Captioning", "✍️ Text-to-Image Generation"])

with tab1:
    st.header("Task 2: Image Captioning with CNN-Transformer")
    st.markdown("Upload an image to see the caption generated by your model trained on the **Flickr8k** dataset.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image_data = uploaded_file.getvalue()
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
    with col2:
        if uploaded_file is not None:
            if st.button("Generate Caption", key="caption_button", use_container_width=True):
                with st.spinner('Generating caption using CNN-Transformer model...'):
                    # The core task: run your custom-trained model
                    caption = generate_caption(image_data, caption_model, caption_tokenizer)
                    
                    st.subheader("Generated Caption:")
                    st.success(caption)
                    st.info("The Image Captioning model combines a CNN Encoder (e.g., ResNet) to extract visual features with a Transformer Decoder to generate the sequence of words.")
        else:
            st.warning("Please upload an image to start the captioning process.")


with tab2:
    st.header("Task 3: Integration of Text-to-Image Synthesis")
    st.markdown("Provide a text prompt to generate a new image using a pretrained generative model.")
    st.markdown("---")
    
    if t2i_pipeline is None:
        st.error("Text-to-Image pipeline failed to load. Please check installation.")
    else:
        prompt = st.text_input(
            "Enter your prompt (e.g., 'A golden retriever wearing sunglasses on a tropical beach')",
            key="t2i_prompt",
            value="A futuristic city in a glass dome, digital art",
        )

        if st.button("Generate Image", key="t2i_button", use_container_width=True):
            if prompt:
                with st.spinner("Synthesizing image... this may take a moment."):
                    # The core task: run the integrated T2I model
                    try:
                        # Use the pipeline to generate the image
                        generated_image = t2i_pipeline(prompt).images[0]
                        st.subheader("Generated Image:")
                        st.image(generated_image, caption=prompt, use_column_width=True)
                        st.success("Image synthesis complete!")
                    except Exception as e:
                        st.error(f"Image generation failed: {e}")
            else:
                st.warning("Please enter a prompt to generate an image.")
                
st.markdown("---")
st.caption("Multimodal Generative AI Project - Deep Learning Expert")
