import streamlit as st
from PyPDF2 import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_fireworks import ChatFireworks
from langchain_fireworks import FireworksEmbeddings
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()
# os.environ["FIREWORKS_API_KEY"] = 'fw_3ZYDmbwntN1Yd4AnEZTKUZMC'
fireworks_api_key=os.getenv('FIREWORKS_API_KEY')

# CREATING THE UI
# chatbot name
st.header('ChattyBuddy')


# with st.sidebar:
st.title('My Documents')
file=st.file_uploader('Upload your pdf and shoot out your questions!!')



# EXTRACT THE TEXT FROM PDF
if file != None:
    typefile = file.type
    st.write(typefile)


    if typefile=='application/pdf':

        pdf_reader = PdfReader(file)
        text = ''
        for eachpage in pdf_reader.pages:
            text += eachpage.extract_text()
        st.text_area(':red[Extracted Text]', text, height=1000)


    if typefile=='image/jpeg':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        image_reader= Image.open(file)
        text = pytesseract.image_to_string(image_reader)
        st.text_area(':red[Extracted Text]', text, height=1000)



# BREAKING THE TEXT INTO CHUNKS
# defining the properties for splitting the text
    text_splitter=RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    #splitting the text into chunks
    chunks=text_splitter.split_text(text)
    #st.write(chunks)

    #st.write("Text chunks:")

    # Displaying few text chunks
    # st.write(chunks)



    # GENERATING EMBEDDINGS

    embeddings = FireworksEmbeddings(api_key=fireworks_api_key, model="nomic-ai/nomic-embed-text-v1.5")

    vector_store=FAISS.from_texts(chunks,embeddings)

    # Displaying few vector embeddings

    # st.write(f"Number of vectors in the vector store: {vector_store.index.ntotal}")
    # st.write("First few vector embeddings (as sample):")
    # st.write(vector_store.index.reconstruct_n())


    question=st.text_input('Ask me anything about the PDF!! ')
    if question:
        match=vector_store.similarity_search(question)
        #st.write(match)
        llm = ChatFireworks(
        api_key=fireworks_api_key,
        model="accounts/fireworks/models/llama-v3-70b-instruct",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        )


        chain=load_qa_chain(llm,chain_type='stuff')
        output=chain.run(question=question,input_documents=match)
        st.write(output)