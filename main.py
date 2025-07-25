import os
import streamlit as st
import pickle
import time
import nltk

# Ensure all necessary NLTK tokenizers and taggers are available
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

from langchain_openai import ChatOpenAI
# Import the new stuff documents chain creator
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain # We'll need this to combine the stuff chain with retrieval (if you want to switch back later)

# Removed: from langchain.chains.summarize import load_summarize_chain # No longer directly using this for stuff
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate # Use ChatPromptTemplate for ChatModels

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

st.title("RockyBot: News Article Summarizer 📝 (Powered by OpenRouter)")
st.sidebar.title("Upload News Article File")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a .html news article", type=["html"])
process_file_clicked = st.sidebar.button("Summarize Article")

# main_placeholder for status messages
main_placeholder = st.empty()

# --- OpenRouter Configuration ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL_NAME = "mistralai/mistral-7b-instruct-v0.2" # Adjust as needed

# Initialize the ChatModel
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    model_name=OPENROUTER_MODEL_NAME,
    temperature=0.5,
    max_tokens=1000
)

# Define a prompt template for summarization using ChatPromptTemplate
# This is crucial for ChatModels
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarizer. Summarize the following content concisely and accurately."),
    ("user", "Summarize the following text:\n\n{context}")
])

# Removed LLMChain creation here


if process_file_clicked:
    if uploaded_file is None:
        st.sidebar.error("❌ Please upload a valid .html file.")
    else:
        temp_file_path = os.path.join("temp_uploaded_file.html")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = UnstructuredFileLoader(temp_file_path)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        if not data:
            st.error("❌ No content found in the uploaded HTML file.")
            os.remove(temp_file_path)
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=100
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        if not docs:
            st.error("❌ No valid documents were extracted after splitting.")
            os.remove(temp_file_path)
            st.stop()

        main_placeholder.text("Creating summarization chain...⏳")
        try:
            # THIS IS THE KEY CHANGE FOR MODERN LANGCHAIN WITH CHATMODELS:
            # Use create_stuff_documents_chain to create the core summarization logic
            stuff_documents_chain = create_stuff_documents_chain(llm, summarize_prompt)

            # For summarization, we directly invoke the stuff_documents_chain
            # with the list of documents
            # If you later need to combine it with retrieval, you'd use create_retrieval_chain
            # and pass it a retriever and this stuff_documents_chain
            # But for pure summarization, we just run this chain directly.
            chain = stuff_documents_chain # The chain to run is now the stuff_documents_chain itself

        except Exception as e:
            st.error(f"Failed to create summarization chain: {e}")
            st.info("Ensure Langchain libraries are up to date: `pip install --upgrade langchain langchain-openai langchain-community langchain-core`")
            os.remove(temp_file_path)
            st.stop()

        main_placeholder.text("Generating Summary...Please wait...⏳⏳⏳")
        start_time = time.time()
        try:
            # Invoke the chain.
            # create_stuff_documents_chain expects 'context' as the input key for documents
            summary_output = chain.invoke({"context": docs})

            end_time = time.time()
            time_taken = round(end_time - start_time, 2)

            st.header("Article Summary")
            st.write(summary_output)
            st.info(f"Summary generated in {time_taken} seconds using {OPENROUTER_MODEL_NAME}.")

        except Exception as e:
            st.error(f"An error occurred during summarization: {e}")
            st.info(f"Please check your OpenRouter API key, model selection ({OPENROUTER_MODEL_NAME}), and that the model is suitable for summarization.")
            st.info("For very long articles, consider increasing `max_tokens` or exploring 'map_reduce' or 'refine' strategies with `load_summarize_chain` (which might have different prompt requirements).")

        os.remove(temp_file_path)

if not process_file_clicked:
    main_placeholder.info("Upload an HTML news article and click 'Summarize Article' to get a summary.")