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

1.  **Build Vector Store:** Place files in `data/`, split into chunks, and generate embeddings.
    * Documents in `data/` are split into chunks.
    * Each chunk is embedded using `intfloat/multilingual-e5-large` or other models.
    * The embeddings are stored in FAISS under `vectorstore/db_faiss/`.
2.  **Compare & Decide:** It compares these hashes to a record of previously processed files.
    * If a file is **new** or its **hash has changed**, it's marked for full processing.
    * If a file's hash is **unchanged**, the script will load its content directly from the fast text cache in the `processed_texts/` folder.
3.  **Query Processing:**
    * User query → FAISS retriever → relevant chunks. 
4.  **Generate an Answer:** 
    * Retrieved context + user query → sent to Qwen/Qwen1.5-7B-Chat.
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

## Setup and Installation (Jupyter Notebook via Open OnDemand)

This project uses **Conda** to manage its environment.

#### 1. Start a Jupyter Session
![Workflow](.github/images/Jupyter1.png)

#### 2. 
```bash
git clone https://github.com/Erzengel583/rag-chatbot
cd rag-chatbot
```

#### 3. Create and Activate the Conda Environment
```bash
# Create the environment with Python 3.10
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

#### 1.  
```bash
python app.py
```
#### 2. 


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

