import os
import sys
import torch
import textwrap

# LangChain and FAISS for retrieval part
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Hugging Face Transformers for the core LLM logic
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Constants and Setup ---
# All paths remain the same
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
LLM_MODEL = "Qwen/Qwen1.5-7B-Chat" # We stick with the stable 7B model

def get_safe_input(prompt: str) -> str:
    """
    Reads input from the terminal as raw bytes and safely decodes it to UTF-8.
    This prevents UnicodeDecodeError on terminals with non-standard encodings.
    """
    # Print the prompt using the configured stdout
    print(prompt, end="", flush=True)
    
    # Read raw bytes from the standard input buffer
    buffer = sys.stdin.buffer
    line_bytes = buffer.readline()
    
    # Decode the bytes into a string, replacing any invalid characters
    # instead of crashing.
    text = line_bytes.decode('utf-8', errors='replace').strip()
    return text



def setup_retriever():
    """Loads the vector store and embedding model to create a retriever."""
    print("⏳ Loading Embedding Model and Vector Store...")
    
    if not os.path.exists(DB_FAISS_PATH):
        print(f"FATAL: Vector store not found at {DB_FAISS_PATH}")
        print("Please run a script to build it first (e.g., build_vector_store.py).")
        sys.exit(1)
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'}
    )
    
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print("✅ Vector Store and Embeddings are ready.")
    return db.as_retriever(search_kwargs={'k': 5})

def load_llm_and_tokenizer():
    """Loads the Qwen model and tokenizer."""
    print(f"⏳ Loading LLM and Tokenizer: {LLM_MODEL}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.bfloat16, # Use bfloat16 for better performance
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    
    print("✅ LLM and Tokenizer are ready.")
    return model, tokenizer

def main():
    # Setup Encoding for Terminal
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
    sys.stdin = open(sys.stdin.fileno(), mode='r', encoding='utf8', buffering=1)
    
    # --- Step 1: Load all necessary components ---
    retriever = setup_retriever()
    model, tokenizer = load_llm_and_tokenizer()

    print("\n" + "="*50)
    print("💬 CHATBOT READY! (Model: Qwen1.5-7B-Chat)")
    print("="*50)
    print("Type your question and press Enter. Type 'exit' or 'quit' to end.")
    print("="*50 + "\n")

    # --- Step 2: Main conversation loop ---
    while True:
        try:
            user_question = input("🧑‍💻 Your Question: ")
            if user_question.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break
            if not user_question.strip():
                continue

            print("\n🤖 Assistant is thinking...")

            # --- RAG Core Logic using Qwen's method ---
            
            # 1. Retrieve relevant documents
            docs = retriever.invoke(user_question)
            context = "\n\n".join([doc.page_content for doc in docs])

            # 2. Create the prompt using the official chat template
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user's question. Answer in Thai."},
                {"role": "user", "content": f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {user_question}"}
            ]
            
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 3. Tokenize the input
            model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

            # 4. Generate the response
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024
            )
            
            # 5. Decode the response, skipping the prompt part
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # --- Display results ---
            print("\n✅ Answer:")
            wrapped_answer = textwrap.fill(response, width=100)
            print(wrapped_answer)
            
            print("\n📚 Sources:")
            for i, doc in enumerate(docs):
                source_content_preview = doc.page_content[:120].replace('\n', ' ') + "..."
                print(f"  [{i+1}] {source_content_preview}")

            print("-" * 50)
        except Exception as e:
            print(f"\n❗️ An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    main()