import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import re

# Define the model and device
model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

# Function to process the prompt and generate images
def generate_images(prompt):
    subjects = re.findall(r"\{([^}]+)\}", prompt)
    if subjects:
        images = []
        for subject in subjects:
            modified_prompt = prompt.replace("{" + subject + "}", subject)
            image = pipe(modified_prompt).images[0]
            images.append(image)
        return images
    else:
        image = pipe(prompt).images[0]
        return [image]

# Streamlit app
st.title("Text-to-Image Generator")

# Text input
prompt = st.text_input("Enter your prompt:")

# Generate button
if st.button("Generate Images"):
    with st.spinner("Generating..."):
        images = generate_images(prompt)

    # Display images with download buttons
    for i, image in enumerate(images):
        st.image(image, caption=f"Image {i+1}", use_column_width=True)
        btn = st.download_button(
            label="Download Image",
            data=image.tobytes(),
            file_name=f"image_{i+1}.png",
            mime="image/png",
        )
