import os
from dotenv import load_dotenv
import streamlit as st

from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface_api import HuggingFaceInferenceAPIEmbedding
# from llama_index.core import Settings


load_dotenv()

# Settings.llm = HuggingFaceInferenceAPI(
#     model_name="google/flan-t5-small",
#     token=os.environ['HF_TOKEN'],
#     provider="auto"
# )

INDEX_DIR = 'wiki'
PAGES = [
    'Artificial intelligence',
    'Machine learning',
    'Natural language processing',
    'Deep learning',
    # 'Reinforcement learning',
    # 'Supervised learning',
    # 'Unsupervised learning',
    # 'Neural networks',
    # 'Computer vision',
    # 'Transformers (machine learning model)',
    # 'Convolutional neural networks',
    # 'Recurrent neural networks',
    # 'Generative adversarial networks',
    # 'Transformer models',
    # 'Support vector machines',
    # 'Decision trees',
    # 'Random forests',
    # 'Gradient boosting',
    # 'K-nearest neighbors',
    # 'Feature engineering',
]

@st.cache_resource
def get_index():
    if os.path.exists(INDEX_DIR):
        storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        return load_index_from_storage(storage)
    
    docs = WikipediaReader().load_data(pages=PAGES, auto_suggest=False)
    embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        token=os.environ['HF_TOKEN']
    )
    
    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    return index

@st.cache_resource
def get_query_engine():
    index = get_index()
    
    llm = HuggingFaceInferenceAPI(
        model_name="google/flan-t5-small",
        token=os.environ['HF_TOKEN'],
        provider="auto"
    )
    
    # Settings.llm = llm
    return index.as_query_engine(llm=llm, similarity_top_k=3)   

def main():
    st.title('Wiki-RAG',text_alignment="center")
    ques = st.text_input('Ask Me any question..')

    if ques:
        with st.spinner('Thinking..'):
            qa = get_query_engine()
            if qa:
                res = qa.query(ques)
                st.subheader('Answer:')
                st.write(res.response)
                st.subheader('Sources:')
                for src in res.source_nodes:
                    st.write(src.node.get_content()[:500] + "...")

if __name__ == '__main__':
    main()