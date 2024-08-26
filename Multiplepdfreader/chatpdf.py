import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()  # Load your API key from environment variables
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create embeddings, text splitter, and load FAISS index outside the loop for efficiency
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

# Load FAISS index if it exists, otherwise create it
if os.path.exists("faiss_index"):
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = None  # Initialize as None in case the file doesn't exist

# --- Helper Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store  # Return the created vector store

def get_conversational_chain():
    prompt_template = """
    Answer the question as comprehensively as possible using the provided context. Include relevant facts, explanations, and comparisons when necessary.  If the answer cannot be determined from the context, state "The answer is not available in the provided context."\n\n

    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, embeddings):  # Pass embeddings as an argument
    if vector_store is not None:  # Check if vector store is initialized
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])
    else:
        st.error("Vector store is not yet created. Please upload and process your PDFs first.")

# --- Streamlit App --- 
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, embeddings)  # Pass the embeddings object

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)

                global vector_store
                vector_store = get_vector_store(text_chunks) 
                st.success("Done")

if __name__ == "__main__":
    main()
