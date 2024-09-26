import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_fireworks import ChatFireworks
from langchain_fireworks import FireworksEmbeddings
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
fireworks_api_key = os.getenv('FIREWORKS_API_KEY')

# Initialize session state to store history and input field state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []  # List to store question-answer pairs
if 'input_text' not in st.session_state:
    st.session_state.input_text = ''  # Store the question input field value

# CREATING THE UI
st.header('ChattyBuddy')
st.title('My Documents')

# File upload widget
file = st.file_uploader('Upload your pdf or image and shoot out your questions!!')

# EXTRACT THE TEXT FROM PDF/IMAGE
if file:
    typefile = file.type
    st.write(f"File type: {typefile}")
    text=''
    if typefile == 'application/pdf':
        pdf_reader = PdfReader(file)
        for eachpage in pdf_reader.pages:
            text += eachpage.extract_text()
        st.text_area('Extracted Text', text, height=300)

    elif typefile == 'image/jpeg':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        image_reader = Image.open(file)
        text = pytesseract.image_to_string(image_reader)
        st.text_area('Extracted Text', text, height=300)

    # Break text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings
    embeddings = FireworksEmbeddings(api_key=fireworks_api_key, model="nomic-ai/nomic-embed-text-v1.5")
    vector_store = FAISS.from_texts(chunks, embeddings)



    def submit_question():
        question=st.session_state.input_text
        if question:
            # Search for matching chunks in the vector store
            match = vector_store.similarity_search(question)

            # Initialize the language model
            llm = ChatFireworks(
                api_key=fireworks_api_key,
                model="accounts/fireworks/models/llama-v3-70b-instruct",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )

            # Use the QA chain to get an answer
            chain = load_qa_chain(llm, chain_type='stuff')
            answer = chain.run(question=question, input_documents=match)

            # Store question and answer in session state
            st.session_state.qa_history.append({'question': question, 'answer': answer})

            # Clear the input field by resetting the session state variable for input
            st.session_state.input_text = ''


    # Use session state for the question input field
    st.text_input('Ask me anything about the document!', value=st.session_state.input_text, key='input_text',
                             on_change=submit_question())



    # Display all previous questions and answers from session state
    if st.session_state.qa_history:
        st.write("### Question-Answer History:")
        for i, qa_pair in enumerate(st.session_state.qa_history, 1):
            st.write(f"Question: {qa_pair['question']}")
            st.write(f"Answer: {qa_pair['answer']}")
            st.write("---")