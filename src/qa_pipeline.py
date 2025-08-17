from pathlib import Path
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHUNK_SIZE = 1000   
CHUNK_OVERLAP = 200

# Initialize Groq LLM client
llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=GROQ_API_KEY,
        streaming=True
    )

def load_and_chunk_file(filepath: Path):
    """Load a PDF file, chunk its content, and auto-generate metadata from filename."""
    loader = PyPDFLoader(str(filepath))
    docs = loader.load()

    # Auto-metadata parsing from filename
    filename = filepath.stem.lower()
    meta = {}
    if "grade" in filename:
        try:
            meta["grade"] = int([x for x in filename.split() if x.isdigit()][0])
        except:
            meta["grade"] = None
    # Detect subject from filename keywords
    if "physics" in filename:
        meta["subject"] = "Physics"
    elif "math" or "mathematics" in filename:
        meta["subject"] = "Math"
    elif "biology" in filename:
        meta["subject"] = "Biology"
    elif "chemistry" in filename:
        meta["subject"] = "Chemistry"
    elif "english" in filename:
        meta["subject"] = "English"
    elif "history" in filename:
        meta["subject"] = "History" 
    elif "geography" in filename:
        meta["subject"] = "Geography"
    elif "civics" in filename:
        meta["subject"] = "Civics"
    elif "economics" in filename:
        meta["subject"] = "Economics"
    elif "social science" in filename:
        meta["subject"] = "Social Science"
    else:
        meta["subject"] = "General"

    # Split document into chunks for embedding and retrieval
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)

    # Attach metadata to each chunk
    for d in split_docs:
        d.metadata.update(meta)
        d.metadata["source_file"] = filepath.name
    return split_docs

def create_vectorstore(uploaded_files):
    """Build a Chroma vectorstore from uploaded files."""
    all_chunks = []
    for f in uploaded_files:
        chunks = load_and_chunk_file(f)
        all_chunks.extend(chunks)

    # Generate embeddings for all chunks
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") 
    # Store vectors in Chroma DB
    vectorstore = Chroma.from_documents(all_chunks, embedding=embeddings)
    # Create retriever for similarity search
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever

def build_rag_chain(retriever):
    """Create a RAG chain for question answering using retrieved context."""
    template = """
    You are a helpful AI tutor. 
    Use ONLY the retrieved content from textbooks/notes to answer the student's question.
    Always provide a reference at the end like: "For more, see Grade {grade}, {subject}, file: {source_file}."

    Question: {question}
    Context: {context}
    """

    prompt = ChatPromptTemplate.from_template(template)

    def format_inputs(question):
        # Retrieve relevant docs
        docs = retriever.invoke(question)
        # Combine context from docs
        context = "\n".join([doc.page_content for doc in docs])
        # Extract metadata from the first doc (or customize as needed)
        meta = docs[0].metadata if docs else {}
        # Return formatted inputs for the prompt
        return {
            "context": context,
            "question": question,
            "grade": meta.get("grade", "N/A"),
            "subject": meta.get("subject", "N/A"),
            "source_file": meta.get("source_file", "N/A"),
        }

    # Build the chain: passthrough -> format inputs -> prompt -> LLM
    rag_chain = (
        RunnablePassthrough()  # Pass the question
        | format_inputs        # Format all required prompt variables
        | prompt
        | llm
    )
    return rag_chain

def build_summary_chain(file_path):
    """Create a chain to summarize the content of a textbook/note."""
    chunks = load_and_chunk_file(file_path)
    # Concatenate all chunk contents for summarization
    context = "\n".join([doc.page_content for doc in chunks])
    template = """
    You are a helpful AI tutor. 
    Summarize the content of the textbook/notes provided for a student.
    
    Context: {context}
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    # Build the chain: prompt -> LLM
    summary_chain = (
        prompt
        | llm
    )

    return summary_chain