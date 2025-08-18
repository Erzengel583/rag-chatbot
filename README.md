# RAG Chatbot (Qwen + HuggingFace + FAISS) & Multi-Format Support

This project is an open-source Retrieval-Augmented Generation (RAG) chatbot, designed for HPC workshops on Jupyter Notebook via Open OnDemand.
It integrates Qwen LLM, Hugging Face embeddings, and FAISS vector database to answer user queries based on custom documents (`.pdf`, `.docx`, `.txt`).

The chatbot supports Thai and multilingual Q&A, uses embeddings for context retrieval, and runs efficiently on GPU.

##  Features

* **Multi-Document Support:** Handles documents in `PDF`, `DOCX`, and `TXT`.
* **Vector Search with FAISS:** Stores embeddings in FAISS for fast document retrieval.
* **Qwen LLM Integration:** Uses `Qwen/Qwen1.5-7B-Chat` for accurate and natural responses.
* **Multilingual Embeddings:** Powered by `intfloat/multilingual-e5-large` for cross-language search.
---

## ⚙️ How It Works

1.  **Build Vector Store:**
    * Place files in `data/`.
    * Files are processed, split into chunks, and embedded using embeddeding model.
    * Embeddings are stored in `vectorstore/db_faiss/`.
2.  **Compare & Decide:** 
    * Each file is hashed (MD5).
    * If unchanged → skipped.
    * If new/modified → fully reprocessed and cached in `processed_texts/`.
3.  **Query Processing:**
    * User query → FAISS retriever → fetch relevant chunks.
4.  **Answer Generation:** 
    * Retrieved context + user query → sent to Qwen/Qwen1.5-7B-Chat or other LLM models.
    * Model generates a Thai/Multilingual response, citing retrieved documents.

---

## 📂 Project Structure

```
rag-chatbot/
├── data/                  # <-- Put your documents here (.pdf, .docx, .txt)
│   └── your_document.pdf
├── processed_texts/       # <-- Cached plain text versions of documents
│   └── your_document_processed.txt
├── vectorstore/           # <-- FAISS vector database
├── .gitignore             # <-- Specifies files/folders for git to ignore
├── app.py                 # <-- Main chatbot script
├── README.md              # <-- This file
└── requirements.txt       # <-- A list of Python dependencies
```

---

## Setup and Installation (Conda + Jupyter Notebook via Open OnDemand)

#### 1. Start a Jupyter Session
![Workflow](.github/images/Jupyter1.png)

#### 2. Clone Repository
```bash
git clone https://github.com/Erzengel583/rag-chatbot
cd rag-chatbot
```
#### 3. Create and Activate the Conda Environment
```bash
# Create environment (Python 3.10 recommended)
conda create --name rag-chatbot-env python=3.10 -y

# Activate the environment
conda activate rag-chatbot-env
```

#### 4. Install Dependencies
Install all required packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```
#### 5.(Optional)
---

## Running the Chatbot

```bash
python app.py
```
##### The chatbot will:
* 1. Ensure required directories exist.
* 2. Build or update FAISS vector store from data/.
* 3. Load embeddings + Qwen LLM.
* 4. Start interactive chat loop.

Type your question and press `Enter`.
Type `exit` / `quit` / `bye` to stop the chatbot.

## Customization (Models & Parameters)
You can edit constants in `app.py`:
* Embedding Model
```python
EMBEDDING_MODEL = 'intfloat/multilingual-e5-large'
```
→ Replace with any HuggingFace sentence-transformer model.

* LLM
```python
LLM_MODEL = "Qwen/Qwen1.5-7B-Chat"
```
→ Replace with another HuggingFace chat model (e.g. Llama2).

* Chunking Parameters
```pyhton
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
```
→ Adjust for document splitting.