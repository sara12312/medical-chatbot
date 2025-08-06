from flask import Flask, render_template, request, jsonify
from src.helpers import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# load enviroment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# download embeddings
embeddings = download_embeddings("sentence-transformers/all-MiniLM-L6-v2")

# load existing index
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name= index_name,
    embedding= embeddings
)

# create retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 3} 
    )

# initialize the LLM model
model = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)

# create chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# create the chain for question answering
question_answer_chain= create_stuff_documents_chain(model, prompt)

# create the retrieval chain
rag_chain= create_retrieval_chain(retriever, question_answer_chain)

# initialize Flask app
app= Flask(__name__)

# route for homepage
@app.route('/')
def index():
    return render_template('chat.html')

# route for handling chat messages
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get response from RAG chain
        response = rag_chain.invoke({"input": user_message})
        
        # Extract the answer from the response
        bot_response = response.get('answer', 'I apologize, but I could not generate a response at this time.')
        
        
        return jsonify({
            'response': bot_response,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Error in chat route: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request. Please try again.',
            'status': 'error'
        }), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
