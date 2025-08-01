from dotenv import load_dotenv
import os

from langchain_pinecone import PineconeVectorStore
from src.helpers import load_pdf, filter_docs, split_text, download_embeddings                                      
from pinecone import Pinecone
from pinecone import ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# access environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# prepare the documents for indexing
extracted_docs = load_pdf(data='data/')
filtered_docs = filter_docs(extracted_docs)
doc_chunks = split_text(filtered_docs)
embeddings = download_embeddings("sentence-transformers/all-MiniLM-L6-v2")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create a serverless index if it does not exist
index_name="medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name= index_name,
        dimension= 384,
        metric= "cosine",
        spec= ServerlessSpec(cloud="aws", region="us-east-1")

) 
# Connect to the index
index = pc.Index(index_name) 

# Create a vector store from the documents
docsearch = PineconeVectorStore.from_documents(
    documents=doc_chunks,
    embedding=embeddings,
    index_name=index_name
)  
