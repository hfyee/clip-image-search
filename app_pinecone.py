'''
Use CLIP for image and text embeddings into Vectors
Ref: https://www.pinecone.io/learn/clip-image-search/

NB.
Using pinecone (Pinecone SDK version 6) instead of langchain_pinecone package for vector store management
'''
# Install dependencies
#!pip install --quiet pinecone transformers torch Pillow matplotlib python-dotenv gdown

import os
import torch
import requests
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.image as mpimg
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from pinecone import Pinecone
import streamlit as st

# Access secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Check for API keys in environment variables
if not openai_api_key or not pinecone_api_key:
    st.warning("Missing API keys.")
    st.stop()

# Set environment variables   
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['PINECONE_API_KEY'] = pinecone_api_key

# --- Functions for data preprocessing ---

def get_image(image_URL):
   image = Image.open(image_URL).convert("RGB")
   return image

def get_model_info(model_ID, device):
    # Save the model to device
    model = CLIPModel.from_pretrained(model_ID).to(device)
    # Get the processor
    processor = CLIPProcessor.from_pretrained(model_ID)
    # Get the tokenizer
    tokenizer = CLIPTokenizer.from_pretrained(model_ID)
    # Return model, processor & tokenizer
    return model, processor, tokenizer

# Text caption embedding
def get_single_text_embedding(text):
    # tokenize and move tensors to the same device as the model
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # get model output and pull the pooled tensor of shape (1, dim)
    outputs = model.get_text_features(**inputs)
    # `outputs` is a BaseModelOutputWithPooling; embeddings live in `.pooler_output`
    text_embeddings = outputs.pooler_output
    # move to CPU and convert to numpy (numpy arrays live on CPU only)
    embedding_as_np = text_embeddings.detach().cpu().numpy()
    return embedding_as_np

def get_all_text_embeddings(df, text_col):
    df["text_embeddings"] = df[str(text_col)].apply(get_single_text_embedding)
    return df

# Image embedding
def get_single_image_embedding(image):
    # preprocess the image and move tensors to the same device as the model
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # get model output and pull the pooled tensor of shape (1, dim)
    outputs = model.get_image_features(**inputs)
    # `outputs` is a BaseModelOutputWithPooling; embeddings live in `.pooler_output`
    image_embeddings = outputs.pooler_output
    # move to CPU and convert to numpy (numpy arrays live on CPU only)
    embedding_as_np = image_embeddings.detach().cpu().numpy()
    return embedding_as_np

def get_all_image_embeddings(df, image_col):
    df["image_embeddings"] = df[str(image_col)].apply(get_single_image_embedding)
    return df

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the model ID
model_ID = "openai/clip-vit-base-patch32"
# Get model, processor & tokenizer
model, processor, tokenizer = get_model_info(model_ID, device)

# Pinecone's managed vector database
# Connect to the index
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
my_index = pc.Index(name='clip-image-search')

# --- UI PAGE CONFIG ---

st.title("🖼️ CLIP Image Search")

with st.spinner("Data preprocessing..."):
    image_data_df = pd.read_csv('bike_captions.csv')
    print(image_data_df.head())
    image_data_df["image"] = image_data_df["image_url"].apply(get_image)
    print(image_data_df.head())

with st.spinner("Vector embedding..."):
    # Apply text embedding to the dataset
    image_data_df = get_all_text_embeddings(image_data_df, "caption")
    print(image_data_df.head())

    # Apply image embedding to the dataset
    image_data_df = get_all_image_embeddings(image_data_df, "image")
    print(image_data_df.head())

    image_data_df["vector_id"] = image_data_df.index
    image_data_df["vector_id"] = image_data_df["vector_id"].apply(str)

    # Get all the metadata
    final_metadata = []

    for index in range(len(image_data_df)):
        final_metadata.append({
            'ID':  index,
            'caption': image_data_df.iloc[index].caption,
            'image': image_data_df.iloc[index].image_url
        })

    image_IDs = image_data_df.vector_id.tolist()
    image_embeddings = [arr.tolist() for arr in image_data_df.image_embeddings.tolist()]

with st.spinner("Upserting data to Pinecone..."):
    # Pinecone v3 expects a list of dicts.
    data_to_upsert = []

    for img_id, emb, meta in zip(image_IDs, image_data_df.image_embeddings, final_metadata):
        # .flatten() ensures [[...]] becomes [...]
        # .tolist() converts the numpy array to a standard Python list of floats
        flat_emb = emb.flatten().tolist() 
    
        data_to_upsert.append({
            "id": img_id,
            "values": flat_emb,
            "metadata": meta
        })

    # Upload the final data
    my_index.upsert(vectors = data_to_upsert)
    # Check index size for each namespace
    print(my_index.describe_index_stats())

# Text-to-image search
query = st.text_input("Search bike images by text:", placeholder="e.g., Tern bike, tangerine color")

if st.button("Run Task"):
    if not query:
        st.error("Please enter a short bike description.")
    else:
        with st.spinner("Searching from vector database..."):
            query_embedding = get_single_text_embedding(query).flatten().tolist()
            results = my_index.query(vector=query_embedding, top_k=2, include_metadata=True)
    
        st.divider()
        st.markdown("### ✨ Results:")

        for match in results['matches']:
            st.write(f"{match['metadata']['caption']}")
            img = mpimg.imread(match['metadata']['image'])
            st.image(img)

        # Image-to-image search