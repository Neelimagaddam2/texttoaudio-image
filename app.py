import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline
import matplotlib.pyplot as plt
from gtts import gTTS
import soundfile as sf
import io
import IPython.display as ipd

# Configuration class
class CFG:
    device = "cuda"
    seed = 42
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9

# Load Stable Diffusion model
@st.cache_resource
def load_image_model():
    model = StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id, torch_dtype=torch.float16, revision="fp16"
    )
    model = model.to(CFG.device)
    return model

image_gen_model = load_image_model()

# Function to generate image
def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]
    image = image.resize(CFG.image_gen_size)
    return image

# Function to generate audio
def generate_audio(text):
    tts = gTTS(text=text, lang="en")
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# Streamlit App Configuration
st.title("AI Media Generation App")

# Tabs for different functionalities
tab1, tab2 = st.tabs(["Generate Image", "Generate Audio"])

# Tab 1: Image Generation
with tab1:
    st.header("Image Generation")

    # Prompt input from user
    prompt = st.text_input("Enter a prompt to generate an image:", "A futuristic cityscape at sunset")

    if st.button("Generate Image"):
        if prompt.strip():
            with st.spinner("Generating image..."):
                image = generate_image(prompt, image_gen_model)
            st.image(image, caption="Generated Image", use_column_width=True)
        else:
            st.warning("Please enter a prompt to generate an image.")

# Tab 2: Audio Generation
with tab2:
    st.header("Audio Generation")

    # Text input from user
    text_input = st.text_area("Enter the text you want to convert to audio:", "Hello, Streamlit!")

    if st.button("Generate Audio"):
        if text_input.strip():
            with st.spinner("Generating audio..."):
                audio_buffer = generate_audio(text_input)
            st.audio(audio_buffer, format="audio/mp3")
        else:
            st.warning("Please enter some text to convert to audio.")

# Footer
st.write("---")
st.write("Developed with ❤️ using Streamlit and AI Models")
