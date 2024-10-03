import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain_fireworks import ChatFireworks
from langchain_fireworks import FireworksEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain.docstore.document import Document as dc
import pytesseract
from docx import Document
from PIL import Image
from dotenv import load_dotenv
import os
import time



# Load environment variables
load_dotenv()
fireworks_api_key = os.getenv('FIREWORKS_API_KEY')

# Initialize session state to store history and input field state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []  # List to store question-answer pairs
if 'input_text' not in st.session_state:
    st.session_state.input_text = ''  # Store the question input field value

# CREATING THE UI
with st.sidebar:
    st.header('ChattyBuddy')
# File upload widget
    file = st.file_uploader('')
    st.write('Upload your file and start shooting your questions!!')



# EXTRACT THE TEXT FROM PDF/IMAGE
if file:
    typefile = file.type
    st.write(f"File type: {typefile}")
    text=''

    # For pdf file

    if typefile == 'application/pdf':
        pdf_reader = PdfReader(file)
        for eachpage in pdf_reader.pages:
            text += eachpage.extract_text()


    # For image file

    elif typefile == 'image/jpeg':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        image_reader = Image.open(file)
        text = pytesseract.image_to_string(image_reader)


    # For docx file

    elif typefile == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc=Document(file)
        for para in doc.paragraphs:
            text+=para.text+ ' '

    # Initializing the llm

    llm = ChatFireworks(
        api_key=fireworks_api_key,
        model="accounts/fireworks/models/llama-v3-70b-instruct",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    summary, QnA = st.tabs(['summary','QnA'])

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)


    # For Summarization purpose

    with summary:

        # prompt = f"Write a concise summary of the following:\\n\\n{text}. Remember to include what it is about and the critical details mentioned. " \
        #          f"Just provide the summary,do not mention any extra word apart from the summary."
        # result=llm.invoke(prompt)
        # st.write(result.content)

        # Creating document object from the text
        docs = [dc(page_content=text)]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summarys = chain.run(docs)
        st.write(summarys)




    # For QnA purpose

    with QnA:
        # st.write('QnA')

        #st.text_area('Extracted Text', text, height=300)


        # Break text into chunks


        # Generate embeddings
        embeddings = FireworksEmbeddings(api_key=fireworks_api_key, model="nomic-ai/nomic-embed-text-v1.5")
        vector_store = FAISS.from_texts(chunks, embeddings)


        # Function to process question

        def submit_question():
            question=st.session_state.input_text
            if question:
                # Search for matching chunks in the vector store
                match = vector_store.similarity_search(question)


                # Use the QA chain to get an answer
                chain = load_qa_chain(llm, chain_type='stuff')
                answer = chain.run(question=question, input_documents=match)

                # Store question and answer in session state
                st.session_state.qa_history.append({'question': question, 'answer': answer})

                # Clear the input field by resetting the session state variable for input
                st.session_state.input_text = ''

                # Write a separator line after each complete line

        chatplaceholder=st.empty()
        with chatplaceholder.container():
        # Display all previous questions and answers from session state
            for i, qa_pair in enumerate(st.session_state.qa_history,0):

                if len(st.session_state.qa_history)-i==1:

                    st.write(f"Question: {qa_pair['question']}")
                    typing_container = st.empty()  # Create a container that will be updated with the answer

                    typed_text = ""
                    for char in qa_pair['answer']:
                        typed_text += char
                        # Update the container with the progressively typed text
                        typing_container.write(f"Answer: {typed_text}")
                        time.sleep(0.002)

                else:
                    st.write(f"Question: {qa_pair['question']}")
                    st.write(f"Answer: {qa_pair['answer']}")
                    st.write("---")


        # Use session state for the question input field
            st.text_input('Ask me anything about the document!', value=st.session_state.input_text, key='input_text',
                      on_change=submit_question)


else:
    st.session_state.qa_history = []
    st.session_state.input_text = ''