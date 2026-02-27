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


# Dimensionality reduction & visualization
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = np.array(tsne.fit_transform(np.array(embeddings)))

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
plt.title("2D Visualization of Customer Reviews")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.savefig("data/review_embeddings_2d.png", dpi=150)
plt.show()
print("Plot saved to data/review_embeddings_2d.png")


# Feedback categorization
topics = ['quality', 'fit', 'style', 'comfort', 'price', 'delivery']
print("Embedding topic labels...")
topic_embeddings = get_embeddings_batch(topics)


def find_closest_topic(review_embedding, topic_embeddings, topics):
    """Find the closest topic label to a given review embedding.

    :param review_embedding: embedding vector of the review
    :param topic_embeddings: list of topic embedding vectors
    :param topics: list of topic label strings
    :return: closest topic label
    """
    distances = [distance.cosine(review_embedding, topic_emb) for topic_emb in topic_embeddings]
    return topics[np.argmin(distances)]


categorized = [find_closest_topic(emb, topic_embeddings, topics) for emb in embeddings]

print("\nSample categorizations:")
for review, category in zip(reviews[:5], categorized[:5]):
    print(f"  Category: {category}")
    print(f"  Review:   {review[:80]}...")
    print()


