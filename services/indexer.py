import os

from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, GOOGLE_API_KEY, VECTORSTORE_PATH


def load_documents():

    documents = []

    csv_loader = CSVLoader(
        file_path="data/catalogo.csv",
        source_column="nome",
        encoding="utf-8",
    )

    documents.extend(csv_loader.load())

    txt_loader = TextLoader(file_path="data/loja_info.txt", encoding="utf-8")

    documents.extend(txt_loader.load())

    print(f"📂 {len(documents)} documentos carregados")

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(documents)

    print(f"✂️  {len(chunks)} chunks criados")

    return chunks


def build_vectorstore(chunks):

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, location=GOOGLE_API_KEY
    )

    if os.path.exists(VECTORSTORE_PATH):
        print("Carregando VectorStore existentes...")
        return FAISS.load_local(
            VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
        )

    print("Criando VectorStore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"VectorStore salvo em: {VECTORSTORE_PATH}/")

    return vectorstore


def get_retriever():

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, location=GOOGLE_API_KEY
    )

    if os.path.exists(VECTORSTORE_PATH):
        print("Carregando VectorStore existente...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
    else:
        documents = load_documents()
        chunks = split_documents(documents)
        vectorstore = build_vectorstore(chunks)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    print("retriever pronto!\n")

    return retriever
