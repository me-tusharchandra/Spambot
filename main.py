from dotenv import load_dotenv
load_dotenv()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load the Google API key from the .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Load the text file
loader = TextLoader("transcript.txt")
documents = loader.load()

# Split the text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings and index
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embeddings)

# Initialize the LLM
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def generate_response(task, content, max_length=150):
    if task == "summarization":
        prompt = f"Summarize the following text with a maximum length of {max_length} characters:\n\n{content}"
    else:
        docs = vectorstore.similarity_search(content, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"You are an AI assistant created to help users with their queries based on the provided context. Your responses should adhere to the following guidelines: 1. If the query can be fully answered using the context, provide a direct and relevant response derived strictly from the information in the given text. 2. If the query cannot be fully answered using the context, try to provide a helpful and polite response. This could include: - Acknowledging the parts of the query you can address using the context. - Suggesting additional information or clarification needed to fully answer the query. - Recommending that the user try summarization if the context seems incomplete or missing information. 3. Maintain a friendly, respectful, and helpful tone in your responses. 4. If you cannot provide any meaningful response based on the context, politely inform the user that the query cannot be answered with the given information. 5. Avoid speculating or providing information not grounded in the provided context. Confidence: 90%\n\nQuestion: {content}\n\nContext:\n{context}"
    
    response = chat.send_message(prompt)
    return response

def extract_text_from_response(response):
    """Extract text content from the response structure."""
    try:
        # Directly access candidates attribute
        candidates = response.candidates
        text_parts = [part.text for candidate in candidates for part in candidate.content.parts]
        return " ".join(text_parts)
    except Exception as e:
        st.error(f"Error extracting text from response: {e}")
        return ""

# Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("SpamBot: Your everyday assistant!")
st.write("Welcome to SpamBot! This AI assistant is designed to help you with your queries and provide summaries of your day-to-day activities given in text (transcript.txt). You can choose between two modes: Q&A and Summarization.")
st.write("---")
st.write("Current transcript:")
st.write("Today I have the following tasks to complete:")
st.write("- Go for a walk with my dog.")
st.write("- Get some groceries.")

st.write("Total 13 teams are participating in today's hackathon.")

st.write("GenAi Hackathon is being conducted at the Rishihood University in Sonipat. The prize pool is 1 Lakh rupees.")
st.write("---")
st.write("Since the model is grounded, please ask questions strictly from the transcript.txt or consider modifying the text as per your needs.")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

option = st.selectbox("Select mode:", ["Q&A", "Summarization"])

if option == "Q&A":
    with st.form(key="qa_form"):
        input = st.text_input("Input: ", key="input")
        submit = st.form_submit_button("Ask the question")

        if submit and input:
            response = generate_response("qa", input)
            response_text = extract_text_from_response(response)
            st.session_state['chat_history'].append(("You", input))
            st.session_state['chat_history'].append(("Bot", response_text))
            st.subheader("The Response is")
            st.write(response_text)
            st.markdown("---")  # Line separator

elif option == "Summarization":
    if documents:
        transcript = documents[0].page_content
        response = generate_response("summarization", transcript)
        summary = extract_text_from_response(response)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("No transcript available for summarization.")

st.subheader("Chat History")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
    st.markdown("---")