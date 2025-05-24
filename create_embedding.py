from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
 
# Use an open-source embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
 
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
