# 📘 AI Tutor - Learn From Your Own Books

AI Tutor is a Streamlit web application that allows students and educators to upload their own textbooks or notes (PDFs), ask questions, and receive AI-generated answers and summaries grounded in their materials. The app uses LangChain for document processing, retrieval, and LLM-powered generation.

---

## Features

- **Document Ingestion:** Upload multiple PDF textbooks or notes. Files are automatically chunked for efficient retrieval.
- **Indexing & Storage:** Text chunks are embedded using Sentence Transformers and stored in a Chroma vector database.
- **Retrieval-Augmented Generation (RAG):** Ask questions about your uploaded materials and receive answers with references.
- **Summarization:** Select any uploaded file and get a concise summary of its content.
- **User-Friendly Interface:** Interact via a simple Streamlit web app.

---

## Live Demo

Access the app here: [https://textbook-tutor.onrender.com](https://textbook-tutor.onrender.com)

---

## Getting Started

### Prerequisites

- Python 3.9+
- [pip](https://pip.pypa.io/en/stable/)
- Groq API key (for LLM access)

### Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/textbook-tutor.git
   cd textbook-tutor
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv .venv
   .venv\Scripts\activate   # On Windows
   # Or
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up your Groq API key:**
   - Create a `.env` file in the project root:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

---

## Usage

1. **Start the Streamlit app:**
   ```sh
   streamlit run src/app_streamlit.py
   ```

2. **Upload your textbooks/notes:**
   - Use the sidebar to upload PDF files.
   - **Important:** Name your files using the format `Grade <number> <Subject>`, e.g., `Grade 9 Biology.pdf`.

3. **Ask questions or request summaries:**
   - Enter your question in the main page and click "Get Answer".
   - Or, select a file and click "Summarize" for a summary.

---

## File Structure

```
textbook-tutor/
│
├── src/
│   ├── app_streamlit.py      # Streamlit web app
│   ├── qa_pipeline.py        # Document processing, retrieval, and chains
│   └── ...                   # Other source files
├── requirements.txt
├── .env                      # Your Groq API key
└── README.md
```

---

## Technologies Used

- [Streamlit](https://streamlit.io/) - Web app framework
- [LangChain](https://python.langchain.com/) - Document loaders, chains, prompts
- [Chroma](https://www.trychroma.com/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [Groq](https://groq.com/) - LLM API

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## Acknowledgements

- LangChain documentation and community
- Streamlit community
- Groq API

---

## Troubleshooting

- **Groq API Key Error:** Make sure your `.env` file is present and contains a valid `GROQ_API_KEY`.
- **PDF Loading Issues:** Ensure your PDFs are not password-protected and are named correctly.
- **Module Import Errors:** Double-check your Python environment and installed packages.

---

## Contact

For questions or support, open an issue or contact [Sumeya: lesumeya3@gmail.com](lesumeya3@gmail.com)