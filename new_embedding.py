from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import requests

# Define your Azure OpenAI embedding class
class AzureOpenAIEmbeddings(Embeddings):
    def __init__(self, endpoint: str, api_key: str):
        self.endpoint = endpoint
        self.headers = {
            "Content-Type": "application/json",
            "api-key": api_key
        }

    def embed_documents(self, texts):
        payload = {
            "input": texts
        }
        response = requests.post(self.endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Set your Azure OpenAI endpoint and subscription key
subscription_key = ""CQRRajGnwjez3oSE7I4b8YlRvCopN1TlpqWz8PDayedXJRQAOCjBJQQJ99BEACHYHv6XJ3w3AAABACOGnSZJ"
api_version = "2024-12-01-preview""  # Replace with your actual key
endpoint = "https://hackathon-2025-ltim.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15"

# Initialize the embedding model
embedding_model = AzureOpenAIEmbeddings(endpoint=endpoint, api_key=subscription_key)

# Define your categories and PDF paths
categories = {
    "category1": ["path/to/pdf1.pdf", "path/to/pdf2.pdf"],
    "category2": ["path/to/pdf3.pdf", "path/to/pdf4.pdf"]
}

# Process each category
for category, pdf_paths in categories.items():
    all_docs = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(all_docs)

    # Create Chroma index for this category
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=f"./chroma_index_{category}"
    )

    # Save to disk
    vectorstore.persist()
