import streamlit as st
import os
import pandas as pd
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API key
os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_API_KEY']

# Directory containing PDF files
pdf_dir = "files"

# Streamlit UI setup
st.image('assets/logoparamonegro.png', width=200)
st.markdown("---")
st.title("🤖 Páramo GPT - Supplier Payments")

# --- Load supplier list from local CSV --- #
def load_supplier_list_from_csv(filepath):
    try:
        df = pd.read_csv(filepath, encoding="latin1")
        return df['name'].dropna().unique().tolist()
    except Exception as e:
        st.warning(f"⚠️ Failed to load supplier list: {e}")
        return []

supplier_list = load_supplier_list_from_csv("support_files/suppliers.csv")
supplier_names = ", ".join(supplier_list)

# --- Extract structured payment data from known layout --- #
def extract_structured_table(docs):
    data = []
    column_names = [
        "NombreID",
        "ID del Banco Nombre delBanco",
        "Número deCuenta Tipo de Cuenta",
        "Monto",
        "NúmerodeFactura",
        "Estado PrenotificaciónVencida",
        "Adenda",
        "Estado deBeneficiario",
        "Código de Motivo deDevolución/Descripción"
    ]

    for doc in docs:
        lines = doc.page_content.splitlines()
        for line in lines:
            parts = re.split(r"\s{2,}", line.strip())  # split on 2+ spaces
            if len(parts) == len(column_names):
                row = dict(zip(column_names, parts))
                data.append(row)

    df = pd.DataFrame(data)

    # Clean the Monto column to numeric
    if "Monto" in df.columns:
        df["Monto"] = df["Monto"].str.replace("[^\d,\.]", "", regex=True)
        df["Monto"] = df["Monto"].str.replace(",", "")
        df["Monto"] = pd.to_numeric(df["Monto"], errors='coerce')

    return df

# --- Load and split PDFs --- #
def load_and_split_pdfs(directory):
    docs = []
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            docs.extend(splitter.split_documents(raw_docs))

    return docs

# --- Process PDFs and create vector store --- #
with st.spinner("🔄 Processing PDFs..."):
    documents = load_and_split_pdfs(pdf_dir)
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(documents, embedding)
    retriever = db.as_retriever(search_kwargs={"k": 20})
    payments_df = extract_structured_table(documents)

# --- RAG setup --- #
prompt = ChatPromptTemplate.from_template(f"""
You are a helpful assistant specialized in answering questions about payments made to suppliers. 
All provided context comes from recent supplier payment PDFs. All payment information is in Colombian Pesos (COP).
All payments contained in the PDFs have been executed. Answer clearly, concisely, and accurately.

The known supplier names include:
{supplier_names}

Use this list to recognize supplier names consistently and match payments across the documents, even if they appear multiple times.

Context:

{{context}}

Question: {{question}}
""")

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Show extracted data table --- #
if not payments_df.empty:
    st.markdown("### 📊 Extracted Payment Records")
    st.dataframe(payments_df)

# --- Question input --- #
user_input = st.text_input("💬 Ask any question about payments to suppliers")

def is_global_question(question):
    return any(keyword in question.lower() for keyword in [
        "list all suppliers", "show all suppliers", "proveedores", "todos los proveedores"
    ])

# --- Process question and show response --- #
if user_input:
    with st.spinner("🤔 Analyzing..."):
        if is_global_question(user_input):
            supplier_names_sorted = sorted(payments_df["NombreID"].dropna().unique().tolist())
            response = "\n".join(supplier_names_sorted)
        else:
            response = chain.invoke(user_input)

    st.markdown("### ✅ Answer:")
    st.write(response)