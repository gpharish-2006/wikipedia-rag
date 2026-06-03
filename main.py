import os,wikipedia
import streamlit as st

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document,
    Settings,
)


INDEX_DIR = 'wiki'
PAGES = [
    'Artificial intelligence',
    'Machine learning',
    'Natural language processing',
    'Deep learning',
    'Reinforcement learning',
]

Settings.llm = Ollama(
    model="qwen2.5:3b",
    request_timeout=300,
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)

@st.cache_resource
def get_index():
    if os.path.exists(INDEX_DIR):
        print("Loading saved index...")
        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage)
    
    docs = []
    print("Creating new index...")

    for topic in PAGES:
        try:
            cont = wikipedia.page(topic,auto_suggest=False).content
            docs.append(Document(text=cont))
        
        except Exception as e:
            print(f"Failed to fetch {topic}")
            print(e)
    print("Total Documents:", len(docs))

    if len(docs) == 0:
        raise Exception("No Wikipedia documents loaded.")
    
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=INDEX_DIR)

    print("Index saved.")
    return index


@st.cache_resource
def get_query_engine():
    index = get_index()
    return index.as_query_engine(similarity_top_k=3)


def main():
    st.title('Wiki-RAG', text_alignment="center")
    ques = st.text_input('Ask Me any question..')

    if ques:
        try:
            with st.spinner('Loading..'):
                qa = get_query_engine()
                if qa:
                    res = qa.query(ques)

                    st.subheader('Answer:')
                    st.write(res.response)

                    st.subheader('Sources:')
                    for i, source in enumerate(res.source_nodes,start=1):
                        with st.expander(f"Source {i}"):
                            st.write(source.node.text[:500])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.code(f"Error Type: {type(e).__name__}")

if __name__ == '__main__':
    main()