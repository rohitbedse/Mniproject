import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# Save FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Saves index.pkl and related files
    return vector_store


# Load FAISS vector store
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Load the FAISS index
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except FileNotFoundError:
        return None


# Create a conversational QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context." Don't provide an incorrect answer.

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Handle user input and generate response
def handle_user_input(user_question):
    # Load the FAISS index
    vector_store = load_vector_store()
    if not vector_store:
        return "The context data is not available. Please upload and process your PDFs again."

    # Perform similarity search
    docs = vector_store.similarity_search(user_question)

    # Get the conversational chain
    chain = get_conversational_chain()

    # Generate response
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


# Streamlit app main function
def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("Chat with PDF using Gemini 💁")
    st.write("@rohitbedse_")

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing complete!")
                    else:
                        st.error("Uploaded PDFs are empty or unreadable.")
            else:
                st.warning("Please upload at least one PDF file.")

    # User question input and response display
    user_question = st.text_input("Ask a question based on the uploaded PDFs:")
    submit_button = st.button("Submit")  # Add a Submit button

    if submit_button and user_question.strip():  # Check if the button is clicked and question is provided
       with st.spinner("Generating response..."):
            response = handle_user_input(user_question)
            st.write("### Reply:", response)
       if response == "The context data is not available. Please upload and process your PDFs again.":
        st.error("No saved context data found. Please upload PDFs and process them first.")
    elif submit_button and not user_question.strip():  # If button is clicked but no question is entered
        st.warning("Please enter a question before submitting.")



if __name__ == "__main__":
    main()
