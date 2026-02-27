import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.manifold import TSNE
from scipy.spatial import distance
import chromadb
import time

# Load the dataset - use ALL reviews
reviews_df = pd.read_csv("data/womens_clothing_e-commerce_reviews.csv")
reviews = reviews_df['Review Text'].dropna().tolist()
print(f"Loaded {len(reviews)} reviews.")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Batch embed in chunks of 100 to avoid rate limits
def get_embeddings_batch(texts, model="text-embedding-3-small", batch_size=100):
    """Embed a list of texts in batches using the OpenAI embeddings API.

    :param texts: list of strings to embed
    :param model: OpenAI embedding model to use
    :param batch_size: number of texts per API call
    :return: list of embedding vectors
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}...")
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend([item.embedding for item in response.data])
        time.sleep(2)  # pause between batches to respect rate limits
    return all_embeddings


embeddings = get_embeddings_batch(reviews)
print(f"Generated {len(embeddings)} embeddings.")


