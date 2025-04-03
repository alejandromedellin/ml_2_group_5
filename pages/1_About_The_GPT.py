import streamlit as st

st.image('assets/logoparamonegro.png', width=200)
st.markdown("---")
st.title("About the GPT")
st.header("📄 Technical Details")
st.subheader("📋 Overview")


st.markdown("""**Páramo GPT** is an enterprise-level **Retrieval-Augmented Generation (RAG)** application designed to enhance how the company interacts with its supplier payment data.
            This RAG application makes API calls to Google's **Gemini 2.0 Flash** LLM, and uses supplier payment data PDF documents (provided by 
            Páramo) as a knowledge base. This enterprise RAG uses a Google Gemini Embedding Model to create embedding vectors of the prompt made to Páramo GPT. 
            For the model to be able to find the most similar vectors compared to that of the embedding vector of the prompt, it uses **FAISS** vector index library (as a pseudo vector database).
            All the different components of the RAG application (i.e., the backend) are connected to each other using LangChain.
            This RAG application was deployed into production using StreamlitCloud.""")

st.markdown("""##### Backend:""")

st.markdown("""
- **Large Language Model:** Gemini 2.0 Flash  
- **Embedding Model:** Google Gemini Embedding Model - *models/embedding-001*  
- **Vector Store/Database:** FAISS  
- **Orchestration Framework:** LangChain
""")

st.markdown("""##### Frontend:""")
st.markdown("- **Frontend Framework:** StreamlitCloud")

st.write("")

st.subheader("🗄️ FAISS vs. Chroma as a Vector Store/Database")
st.markdown("""
For our vector store/database, we chose **FAISS** instead of **Chroma** due to **cloud deployment compatibility issues** with StreamlitCloud:

- **FAISS** is a **vector index library** designed for **efficient similarity search** on high-dimensional vectors (like embeddings)  
- It’s not a full vector database like Chroma, but it's **lightweight, fast, and ideal for in-memory use**
- **Chroma**, on the other hand, depends on **SQLite ≥ 3.35.0**, which is **not supported on Streamlit Cloud**, leading to deployment errors
- While Chroma is more suited for **production-level applications** with persistence and metadata, **FAISS is perfect for lightweight, local or cloud-compatible RAG applications**
- Since our project is a **small-scale enterprise RAG prototype** with no need for persistent storage or cloud-scale database features, FAISS was the **most reliable and practical choice**
""")


#✅ FAISS works smoothly on **Streamlit Cloud**  
#❌ Chroma does not, due to **SQLite limitations**""")

#instead of Chroma due to cloud compatibility issues with Streamlit Cloud
