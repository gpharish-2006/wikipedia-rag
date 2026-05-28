# Wikipedia-RAG

A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that answers questions using **Wikipedia** content.

## Features

- **Dynamic Knowledge Base**: Automatically fetches and indexes specific Wikipedia pages.

- **Persistent Caching**: Saves the vector index locally (`./wiki`) to skip re-processing on subsequent runs.


## How It Works

The application follows a standard **RAG (Retrieval-Augmented Generation)** pipeline:

1.  **Data Ingestion**  
    On the first run, the `WikipediaReader` fetches raw text from the specified articles (e.g., "Artificial intelligence", "Machine learning").

2.  **Vector Embedding**  
    The text is split into chunks and converted into numerical vectors using the embedding models via the Hugging Face Inference API.

3.  **Local Persistence**  
    These vectors are saved to a local `./wiki` directory. On subsequent runs, the app loads this index instantly instead of re-downloading data.

4.  **Semantic Retrieval**  
    When you ask a question, the system converts your query into a vector and finds the **top 3 most relevant chunks** from the stored Wikipedia data.

5.  **AI Generation**  
    The retrieved context and your question are sent to a query model, which synthesizes a natural language answer based *only* on the provided Wikipedia content.   


## Quick Start

### 1. Prerequisites

- Python 3.9+

- A **Hugging Face Account Access Token** in .env file.


### 2. Installation
```bash
pip install uv
```

## Initialize the project

```bash
git clone https://github.com/gpharish-2006/wikipedia-rag.git

cd wikipedia-rag

uv init
```

## Create a virtual environment
```bash
uv venv
```

## Activate the virtual environment
```bash
source .venv/bin/activate  # Linux/macOS

.venv\Scripts\activate   # Windows
```

## Install dependencies
```bash
uv add -r requirements.txt
```

# Running the Application
## Run directly
```bash
uv run streamlit run main.py
```