# 💼 Financial Analysis RAG (Retrieval-Augmented Generation using Gemini API)

This project is a **Retrieval-Augmented Generation (RAG)** based financial assistant that answers user questions using uploaded financial documents like PDFs and text files. It combines local document retrieval (via FAISS) and **Gemini (Google Generative AI)** for intelligent, context-aware financial query responses.

---

## 📌 Features

- ✅ Upload financial data files (`.pdf`, `.txt`)
- ✅ Extract and store content in vector database (FAISS)
- ✅ Ask questions in natural language
- ✅ Time-aware question tracking with `question_history.py`
- ✅ Streamed, context-rich responses via Gemini API

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|-------------|
| LLM | Gemini (Google Generative AI) |
| Vector DB | FAISS |
| Embedding | Google Embeddings |
| File Parsing | PyMuPDF (PDF), standard text reading |
| Backend | Python |
| Environment | CLI (or Streamlit UI optional) |

---

## 🗂️ Project Structure

```bash
RAG_Financial/
├── app.py                      # Main application script
├── db_loader.py               # Loads documents and initializes vector store
├── get_answer.py              # Handles retrieval + Gemini response
├── get_pdf_text.py            # Extracts text from uploaded PDFs
├── get_txt_text.py            # Reads text files
├── prompt.py                  # Prompt template generator
├── question_history.py        # Maintains time-aware question history
├── .env                       # Contains your Gemini API key
└── requirements.txt           # Python dependencies



⚙️ Setup Instructions
1. Clone the repository
git clone <repo_url>
cd RAG_Financial

2. Install dependencies
pip install -r requirements.txt

3. Add your Gemini API key
Create .env 
GOOGLE_API_KEY=your-gemini-api-key-here

Run the Application
python app.py

