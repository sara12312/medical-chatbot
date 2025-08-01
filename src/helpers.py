from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
import torch 
# extract text from pdf file
def load_pdf(data):
    loader= DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls= PyPDFLoader # type: ignore
        )
        
    documents = loader.load()
    return documents   


# include only relevant content from the documents
def filter_docs(docs: List[Document]) -> List[Document]:
    filtered_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source") # Extract the 'source' from the original metadata
        filtered_docs.append(
            Document(
                page_content=doc.page_content,   # Keep the original page content
                metadata={"source": src}         # Only keep 'source' in metadata
            )
        )
    return filtered_docs    



#split documents into smaller chunks
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(docs)



def download_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )
    