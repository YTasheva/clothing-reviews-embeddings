<h1 align="center">Women's Clothing Reviews — Embedding Analysis</h1>

A Python project that uses OpenAI text embeddings to analyze women's clothing e-commerce reviews. It covers semantic visualization, topic categorization, and similarity search using ChromaDB.

## Features

- **Text Embeddings** — Embeds all reviews using OpenAI's `text-embedding-3-small` model in batches
- **2D Visualization** — Reduces embeddings to 2D with t-SNE and plots semantic clusters
- **Topic Categorization** — Classifies each review into topics like quality, fit, style, comfort, price, delivery
- **Similarity Search** — Finds the most similar reviews to a given input using ChromaDB vector search

## Project Structure

```
clothing-reviews-embeddings/
├── README.md
├── requirements.txt
├── main.py
├── data/
│   └── clothing_reviews.csv
└── .env
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/clothing-reviews-embeddings.git
cd clothing-reviews-embeddings
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Edit `.env`:

```
OPENAI_API_KEY=api_key_here
```

### 4. Add your data

Place in the `data/` folder -> `clothing_reviews.csv`
The required column is:

| Column | Description |
|--------|-------------|
| `Review Text` | Raw customer review text |

### 5. Run

```bash
python main.py
```

## How It Works

### Step 1 — Batch Embeddings
Reviews are embedded in chunks of 100 using a single API call per batch, with a short pause between batches to respect rate limits. This processes the full dataset (900–1000 reviews) efficiently.

```python
def get_embeddings_batch(texts, model="text-embedding-3-small", batch_size=100):
    ...
```

### Step 2 — Dimensionality Reduction & Visualization
t-SNE reduces the 1536-dimensional embeddings to 2D for plotting, revealing natural semantic clusters in the review data.

### Step 3 — Topic Categorization
Topic labels (`quality`, `fit`, `style`, `comfort`, `price`, `delivery`) are embedded and compared to each review embedding using cosine distance. Each review is assigned its closest topic.

### Step 4 — Similarity Search (ChromaDB)
All review embeddings are stored in a ChromaDB collection. Given any input review, the 3 most semantically similar reviews are retrieved.

```python
most_similar_reviews = find_similar_reviews(
    "Absolutely wonderful - silky and sexy and comfortable",
    collection
)
```

## Output

- A 2D scatter plot of review embeddings
- Console output showing topic categorization for sample reviews
- `most_similar_reviews` — a list of the 3 most similar reviews to the input

## Requirements

- Python 3.8+
- OpenAI API key with access to `text-embedding-3-small`

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Author

- GitHub - [Yuliya Tasheva](https://github.com/YTasheva)

> [Website](https://yuliya-tasheva.co.uk/) &nbsp;&middot;&nbsp;
>  [LinkedIn](https://www.linkedin.com/in/yuliya-stella-tasheva/) &nbsp;&middot;&nbsp;
> [Email](info@yuliya-tasheva.co.uk) &nbsp;&middot;&nbsp;
