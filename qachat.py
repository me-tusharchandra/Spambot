from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from transformers import pipeline
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

genai.configure(api_key="AIzaSyC6pjCSuzeSdm8jMB6O0JHahEtkPzlc5pc")

# Load the text file
loader = TextLoader("transcript.txt")
documents = loader.load()

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Function to summarize the transcript
def summarize_transcript(transcript, max_length=150):
    summary = summarizer(transcript, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Split the text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings and index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embeddings)

# Initialize the LLM
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    # Retrieve relevant information from the text data
    docs = vectorstore.similarity_search(question, k=3)
    
    # Combine the relevant information
    context = "\n".join([doc.page_content for doc in docs])
    
    # Generate a response based on the context
    response = chat.send_message(f"Generate responses to user queries based on the provided text, which contains a detailed account of daily activities. Responses should be: 1. Derived strictly from the information in the given text. 2. Relevant and accurate. 3. Polite, with a tone of helpfulness and friendliness. In a situation where you cannot provide an answer, consider responding with the following suggestions: - 'Please provide more context.' - 'You may want to try summarization if some information seems missing.' Confidence: 90%\n\nQuestion: {question}\n\nContext:\n{context}")
    
    return response

# Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("SpamBot: Your everyday assistant!")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

option = st.selectbox("Select mode:", ["Q&A", "Summarization"])

if option == "Q&A":
    input = st.text_input("Input: ", key="input")
    submit = st.button("Ask the question")

    if submit and input:
        response = get_gemini_response(input)
        st.session_state['chat_history'].append(("You", input))
        st.subheader("The Response is")
        for chunk in response:
            st.write(chunk.text)
            st.session_state['chat_history'].append(("Bot", chunk.text))

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
