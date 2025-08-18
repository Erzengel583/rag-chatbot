# RAG Chatbot with Caching & Multi-Format Support

This project provides open-source Retrieval-Augmented Generation (RAG) chatbot built with Python. It intelligently processes multiple document types (`.pdf`, `.docx`, `.txt`), uses an efficient caching system to ensure fast startups, and automatically detects file changes to keep its knowledge base perfectly up-to-date.

The entire system is designed for easy use on a local machine and is optimized for deployment on High-Performance Computing (HPC) clusters using Singularity.

##  Features

* **Multi-Document Support:** Ingests and understands various file formats, including **PDF**, **Microsoft Word** (`.doc`, `.docx`), and **plain text** (`.txt`).
* **Intelligent Caching:** Automatically saves a processed text version of complex files. On subsequent runs, it loads from this fast cache instead of reprocessing the original document, saving significant time.
* **Automatic Change Detection:** Uses **MD5 hashing** to detect if a source file has been modified. The chatbot only spends time reprocessing files that are new or have been changed, making knowledge base updates incredibly efficient.
* **HPC Ready:** Includes detailed, best-practice instructions for deploying the Ollama server in a containerized environment on HPC clusters using **Singularity**.
* **Fully Open-Source:** Built with leading open-source libraries like **LangChain**, **Ollama**, **FAISS**, and **Hugging Face** models.

---

## ⚙️ How It Works

The script employs a smart processing pipeline to minimize redundant work and maximize speed.

1.  **Scan & Hash:** The script scans the `data/` directory and calculates a unique hash (MD5 checksum) for each file.
2.  **Compare & Decide:** It compares these hashes to a record of previously processed files.
    * If a file is **new** or its **hash has changed**, it's marked for full processing.
    * If a file's hash is **unchanged**, the script will load its content directly from the fast text cache in the `processed_texts/` folder.
3.  **Process & Cache:** New or modified PDF and DOCX files are read, their text is extracted, and this plain text is saved to the cache for future use.
4.  **Build Vector Store:** The text content from all documents is split into chunks, converted into embeddings, and stored in a **FAISS vector database**. This database is what the chatbot uses to find relevant information to answer questions.



This workflow ensures that you only pay the "cost" of processing a complex document once.

---

## 📂 Project Structure

```
rag-chatbot/
├── data/                  # <-- Place your source documents here(docx, pdf, txt)
│   └── your_document.pdf
├── processed_texts/       # <-- Cached plain text versions of documents
│   └── your_document_processed.txt
├── vectorstore/           # <-- The FAISS vector store is saved here
├── .gitignore             # <-- Specifies files/folders for git to ignore
├── app.py                 # <-- The main Python application script
├── README.md              # <-- This file
└── requirements.txt       # <-- A list of Python dependencies
```

---

## Setup and Installation (Conda)

This project uses **Conda** to manage its environment.

#### 1. Prerequisites
* [Git](https://git-scm.com/downloads) installed.
* [Conda](https://www.anaconda.com/download) (Anaconda or Miniconda) installed.
* [Ollama](https://ollama.com/) installed and running.

#### 2. Clone the Repository
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

---

## Ollama Deployment with Singularity

For Jupyter Notebooks, running Ollama inside a **Singularity container** is the recommended method for portability and stability.

#### 1. Run Ollama on a Compute Node
After starting an interactive job on a compute node

**Using `singularity run` **
This method uses standard Linux commands to run the server in the background.

```bash
# Run the 'serve' command and send it to the background (&)
singularity run --nv ollama.sif serve &

# Pull your model by passing arguments directly
singularity run ollama.sif pull llama3

# When finished, stop the server process
pkill -f "singularity run --nv ollama.sif serve"
```

---

## 💬 How to Use the Chatbot

The application is designed to be simple to run after setup.

1.  **Add Documents:** Place your `.pdf`, `.docx`, `.doc`, and `.txt` files into the `data/` folder.
2.  **Run the Application:** Open your terminal, activate the Conda environment, and run the script.
    ```bash
    conda activate rag-chatbot-env
    python app.py
    ```
3.  **Follow Prompts:**
    * On the **first run**, the script will automatically process all documents and build the vector store.
    * On **subsequent runs**, it will ask if you want to check for new or modified files. Type `y` and press Enter to update the knowledge base. If you type `n`, it will start instantly using the existing database.
4.  **Start Chatting:** Once you see the "CHATBOT READY!" message, you can start asking questions based on your documents! Type `exit` to end the session.