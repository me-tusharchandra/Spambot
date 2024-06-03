from dotenv import load_dotenv
load_dotenv()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from transformers import pipeline
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load the text file
loader = TextLoader("transcript.txt")
documents = loader.load()

# Split the text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings and index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embeddings)

# Initialize the summarization and Q&A pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def summarize_transcript(transcript, max_length=150):
    prompt = f"Summarize the following text with a maximum length of {max_length} characters:\n\n{transcript}"
    summary = summarizer(prompt, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def get_qa_response(question, context):
    prompt = f"Generate a response to the question based on the provided context. The response should be derived strictly from the information in the given text, be relevant and accurate, and maintain a tone of helpfulness and friendliness.\n\nQuestion: {question}\n\nContext:\n{context}"
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("SpamBot: Your everyday assistant!")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

option = st.selectbox("Select mode:", ["Q&A", "Summarization"])

if option == "Q&A":
    input = st.text_input("Input: ", key="input")
    submit = st.button("Ask the question")
    clear_chat = st.button("Clear Chat")

    if clear_chat:
        st.session_state['chat_history'] = []

    if submit and input:
        # Retrieve relevant information from the text data
        docs = vectorstore.similarity_search(input, k=3)
        
        # Combine the relevant information
        context = "\n".join([doc.page_content for doc in docs])
        
        # Get the response from the Q&A pipeline
        response = get_qa_response(input, context)
        st.session_state['chat_history'].append(("You", input))
        st.subheader("The Response is")
        st.write(response)
        st.session_state['chat_history'].append(("Bot", response))

elif option == "Summarization":
    if documents:
        transcript = documents[0].page_content
        summary = summarize_transcript(transcript)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("No transcript available for summarization.")

st.subheader("Chat History")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")