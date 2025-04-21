import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv


load_dotenv()
# Set environment variables
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
INDEX_NAME = "medical-knowledge-index"

# Initialize the LLM
def load_llm(repo_id, token):
    return HuggingFaceEndpoint(
        endpoint_url=f"https://api-inference.huggingface.co/models/{repo_id}",
        huggingfacehub_api_token=token,
        temperature=0.5,
        model_kwargs={"max_length": 512}
    )

llm = load_llm(HUGGINGFACE_REPO_ID, HUGGINGFACE_TOKEN)
print("LLM Loaded:", llm)

# Define custom prompt
CUSTOM_PROMPT_TEMPLATE = """
You have great knowledge in medicine. You are assisting a doctor to take informed decision. You will not give decision rather than help the doctor to decision on some cases. 
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Check if the index exists; if not, create it
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Ensure this matches your embedding model's output dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Adjust as needed
    )

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the vector store
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embedding_model
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

# Create the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

def generate_response(user_query):
    response = qa_chain.invoke({'query': user_query})
    return response

