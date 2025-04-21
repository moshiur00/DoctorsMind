from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
import json

# This script creates a knowledge base from PDF files by extracting text, chunking it, and storing it in a vector database.
# It uses Langchain and Pinecone for document processing and vector storage.

# Step 1: Load raw PDF(s)
DATA_PATH="data/"
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
print("Length of PDF pages: ", len(documents))


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=40)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings 

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model=get_embedding_model()

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # e.g., "us-west1-gcp"

# Initialize Pinecone
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

# Define Pinecone index name
INDEX_NAME = "medical-knowledge-index"

# Store embeddings in Pinecone
def store_embeddings_in_pinecone(chunks, embedding_model, batch_size=100):
    # Check if the index exists, if not, create it

    if 'medical-knowledge-index' not in pc.list_indexes().names():
        pc.create_index(
            name='medical-knowledge-index',
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    # Connect to the index
    index = pc.Index(INDEX_NAME)



    # Convert chunks to embeddings and upsert into Pinecone
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.embed_query(chunk.page_content)
        
        # Ensure metadata is serialized to a JSON string if it's a dictionary
        metadata = chunk.metadata
        if isinstance(metadata, dict):
            metadata = json.dumps(metadata)  # Serialize dictionary to JSON string
        
        vectors.append((f"doc-{i}", embedding, {"metadata": metadata}))

    # Upsert vectors in batches
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(batch)
        print(f"Upserted batch {i // batch_size + 1} of size {len(batch)}")

    print(f"Stored {len(vectors)} embeddings in Pinecone.")

store_embeddings_in_pinecone(text_chunks, embedding_model)
