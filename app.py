# (All imports remain the same)
import os
import sys
from typing import Optional, List
from pathlib import Path
import hashlib
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.schema import Document

# --- Constants and Setup (largely the same) ---
DATA_PATH = "data/"
PROCESSED_PATH = "processed_texts/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.doc', '.docx']

# --- Helper Functions ---
def get_embeddings():
    """Create and return the embeddings model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'}
    )

def ensure_directories_exist():
    """Ensure required directories exist."""
    for directory in [DATA_PATH, PROCESSED_PATH, "vectorstore"]:
        os.makedirs(directory, exist_ok=True)

def get_file_hash(file_path: str) -> str:
    """Generate a hash for a file."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_processed_files_record() -> dict:
    """Load the record of processed files."""
    record_file = os.path.join(PROCESSED_PATH, "processed_files.txt")
    processed = {}
    if os.path.exists(record_file):
        with open(record_file, 'r') as f:
            for line in f:
                if '|' in line:
                    parts = line.strip().split('|', 1)
                    processed[parts[0]] = parts[1]
    return processed

def save_processed_files_record(processed: dict):
    """Save the record of processed files."""
    record_file = os.path.join(PROCESSED_PATH, "processed_files.txt")
    with open(record_file, 'w') as f:
        for file_path, file_hash in processed.items():
            f.write(f"{file_path}|{file_hash}\n")

# --- 3. DOCUMENT PROCESSING ---
def process_pdf_file(file_path: str) -> List[Document]:
    """Process a single PDF file and return documents."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"  ✓ Loaded PDF: {os.path.basename(file_path)} ({len(documents)} pages)")
        return documents
    except Exception as e:
        print(f"  ✗ Error loading PDF {file_path}: {e}")
        return []

def process_text_file(file_path: str) -> List[Document]:
    """Process a single text file and return documents."""
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        print(f"  ✓ Loaded TXT: {os.path.basename(file_path)}")
        return documents
    except Exception as e:
        try:
            # Try with different encoding if UTF-8 fails
            loader = TextLoader(file_path, encoding='latin-1')
            documents = loader.load()
            print(f"  ✓ Loaded TXT: {os.path.basename(file_path)} (latin-1 encoding)")
            return documents
        except Exception as e2:
            print(f"  ✗ Error loading text file {file_path}: {e2}")
            return []

def process_doc_file(file_path: str) -> List[Document]:
    """Process a DOC/DOCX file and return documents."""
    try:
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        print(f"  ✓ Loaded DOC: {os.path.basename(file_path)}")
        return documents
    except Exception as e:
        print(f"  ✗ Error loading DOC file {file_path}: {e}")
        print(f"    Note: You may need to install python-docx: pip install python-docx")
        return []

def load_and_process_documents() -> List[Document]:
    all_documents = []
    processed_record = load_processed_files_record()
    new_processed_record = processed_record.copy()
    
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(Path(DATA_PATH).glob(f"**/*{ext}"))
    
    if not files:
        raise ValueError(f"No supported files found in {DATA_PATH}")

    print(f"\nFound {len(files)} files to process:")
    print("-" * 50)
    
    for file_path in files:
        file_str = str(file_path)
        file_hash = get_file_hash(file_str)
        
        # Check if file is unchanged and a cached version exists
        is_unchanged = file_str in processed_record and processed_record[file_str] == file_hash
        processed_txt_path = os.path.join(PROCESSED_PATH, f"{file_path.stem}_processed.txt")

        # REFINED LOGIC: Try to load from cache if unchanged
        if is_unchanged and os.path.exists(processed_txt_path):
            print(f"  → Loading from cache: {os.path.basename(file_str)}")
            docs = process_text_file(processed_txt_path)
            if docs:
                all_documents.extend(docs)
                continue # Skip to the next file if cache loading was successful
        
        # If not loaded from cache, process the original file
        print(f"  → Processing new/modified file: {os.path.basename(file_str)}")
        ext = file_path.suffix.lower()
        documents = []
        if ext == '.pdf':
            documents = process_pdf_file(file_str)
        elif ext == '.txt':
            documents = process_text_file(file_str)
        elif ext in ['.doc', '.docx']:
            documents = process_doc_file(file_str)
        
        if documents:
            all_documents.extend(documents)
            # Save to cache if it's not a plain text file
            if ext != '.txt':
                try:
                    with open(processed_txt_path, 'w', encoding='utf-8') as f:
                        f.write("\n\n".join(doc.page_content for doc in documents))
                    print(f"    → Saved to cache: {os.path.basename(processed_txt_path)}")
                except Exception as e:
                    print(f"    ⚠ Could not save to cache: {e}")
            # Update record
            new_processed_record[file_str] = file_hash
            
    save_processed_files_record(new_processed_record)
    print("-" * 50)
    return all_documents

# --- BUILD THE VECTOR STORE ---
def create_vector_db():
    """
    This function loads documents (including PDFs), splits them into chunks,
    creates embeddings, and stores them in a FAISS vector store.
    """
    try:
        print("\n" + "="*50)
        print("BUILDING VECTOR DATABASE")
        print("="*50)
        
        # Load and process all documents
        documents = load_and_process_documents()
        
        # Split the documents into smaller chunks for processing
        print(f"\nSplitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        print(f"✓ Split into {len(texts)} chunks")
        
        # Define the embedding model
        print(f"\nCreating embeddings...")
        embeddings = get_embeddings()
        
        # Create the FAISS vector store from the text chunks and embeddings
        print("Building FAISS vector store...")
        db = FAISS.from_documents(texts, embeddings)
        
        # Save the vector store locally
        db.save_local(DB_FAISS_PATH)
        print(f"✓ Vector store saved to {DB_FAISS_PATH}")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error creating vector database: {e}")
        sys.exit(1)

# --- 5. SETUP THE RAG CHAIN ---
def setup_rag_chain() -> Optional[object]:
    """
    This function sets up the RAG chain by loading the vector store,
    defining the LLM, creating a prompt template, and assembling the chain.
    """
    try:
        # Load the saved vector store
        embeddings = get_embeddings()
        
        if not os.path.exists(DB_FAISS_PATH):
            raise FileNotFoundError(f"Vector store not found at {DB_FAISS_PATH}")
        
        print("Loading vector store...")
        # Note: allow_dangerous_deserialization should be used carefully in production
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("✓ Vector store loaded")
        
        # Initialize the Ollama LLM
        print("Initializing language model...")
        try:
            llm = Ollama(model="llama3")
            print("✓ Language model initialized")
        except Exception as e:
            print(f"✗ Error initializing Ollama model: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Ollama is installed: https://ollama.ai")
            print("2. Make sure the llama3 model is downloaded: ollama pull llama3")
            print("3. Make sure Ollama service is running: ollama serve")
            return None
        
        # Create a prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        Think step-by-step and provide a detailed answer.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        
        <context>
        {context}
        </context>
        
        Question: {input}
        
        Answer:
        """)
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create the retriever from the vector store with specific parameters
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant documents
        )
        
        # Create the final retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        print("✓ RAG chain ready\n")
        return retrieval_chain
        
    except Exception as e:
        print(f"✗ Error setting up RAG chain: {e}")
        return None


# --- REFINED: Dependency Checker ---
def check_dependencies():
    """Check if all required dependencies for file processing are installed."""
    missing_deps = []
    deps_to_check = {
        "pypdf": "for PDF files",
        "unstructured": "for DOC/DOCX files",
        "docx": "a dependency for DOC/DOCX files"
    }
    install_commands = {
        "pypdf": "pypdf",
        "unstructured": "unstructured",
        "docx": "python-docx"
    }

    for lib, _ in deps_to_check.items():
        try:
            __import__(lib)
        except ImportError:
            missing_deps.append(lib)

    if missing_deps:
        print("\n" + "="*50)
        print("⚠️ MISSING DEPENDENCIES")
        print("="*50)
        print("The following packages are required to process all file types:")
        install_list = []
        for dep in missing_deps:
            print(f"  • {dep} ({deps_to_check[dep]})")
            install_list.append(install_commands[dep])
        print("\nPlease install them using pip:")
        print(f"  pip install {' '.join(install_list)}")
        # Add a note about system dependencies for unstructured
        if "unstructured" in missing_deps:
            print("\nNOTE: 'unstructured' may require system libraries on Linux/macOS.")
            print("See: https://unstructured-io.github.io/unstructured/installing.html")
        print("="*50 + "\n")
        return False
    return True

# --- main() function is great as is ---
# (No changes needed in the main execution logic)
def main():
    """Main execution function."""
    try:
        # Check dependencies
        if not check_dependencies():
            print("Please install the required dependencies and try again.")
            return
        
        # Ensure required directories exist
        ensure_directories_exist()
        
        # Check if we need to rebuild the vector store
        rebuild = False
        
        if not os.path.exists(DB_FAISS_PATH):
            print("\n✓ Vector store not found. Will create a new one.")
            rebuild = True
        else:
            # Check if there are new or modified files
            user_input = input("\nVector store exists. Check for new/modified files? (y/n): ").strip().lower()
            if user_input == 'y':
                rebuild = True
        
        if rebuild:
            # Check if there are files to process
            files_exist = False
            for ext in SUPPORTED_EXTENSIONS:
                if list(Path(DATA_PATH).glob(f"**/*{ext}")):
                    files_exist = True
                    break
            
            if not files_exist:
                print(f"\n✗ Error: No supported files found in {DATA_PATH}")
                print(f"  Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
                print(f"  Please add some files to the '{DATA_PATH}' directory and try again.")
                return
            
            create_vector_db()
        
        # Setup the RAG chain
        print("\nInitializing RAG system...")
        rag_chain = setup_rag_chain()
        
        if rag_chain is None:
            print("Failed to initialize the RAG chain. Exiting...")
            return
        
        # Start a conversation loop
        print("\n" + "="*50)
        print("💬 CHATBOT READY!")
        print("="*50)
        print("Commands:")
        print("  • Type your question and press Enter")
        print("  • Type 'exit', 'quit', or 'bye' to end")
        print("  • Press Ctrl+C to force quit")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                # Check for exit command
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Goodbye!")
                    break
                
                # Validate input
                if not user_input:
                    print("⚠ Please enter a question.")
                    continue
                
                # Get a response from the RAG chain
                print("\n🤔 Thinking...", end="", flush=True)
                response = rag_chain.invoke({"input": user_input})
                
                # Clear the "Thinking..." message and print the answer
                print("\r" + " "*50 + "\r", end="")  # Clear the line
                print("Bot:", response['answer'])
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error processing question: {e}")
                print("Please try again with a different question.")
    
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
