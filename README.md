# 📄 PDF Chatbot — LangChain + Groq + Streamlit

An interactive chatbot that lets you upload PDF files and ask questions about their content, powered by LangChain, Groq LLMs, HuggingFace embeddings, and FAISS vector search.

---

## ⚙️ Requirements

- **Python 3.11** (not 3.12+, not 3.14 — many AI packages are incompatible with newer versions)
- **uv** package manager ([install here](https://docs.astral.sh/uv/getting-started/installation/))
- A **Groq API key** ([get one here](https://console.groq.com/keys))

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/dvl1-o/pdf-chatbot.git
cd pdf-chatbot
```

### 2. Create a virtual environment with Python 3.11

```bash
uv venv --python 3.11
```

Then activate it:

- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate
  ```
- **Linux/Mac:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Update `pyproject.toml`

Make sure the Python version constraint is set correctly:

```toml
requires-python = ">=3.11, <3.12"
```

### 4. Install dependencies

```bash
uv add streamlit langchain==0.2.16 langchain-community==0.2.17 langchain-groq==0.1.9 pypdf sentence-transformers==2.7.0 faiss-cpu greenlet==3.0.3 chromadb==0.4.24
uv add torch==2.1.0 --extra-index-url https://download.pytorch.org/whl/cpu
uv add transformers==4.35.0
```

### 5. Set your Groq API key

Create a `.env` file at the root of the project:

```
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ **Never hardcode your API key in the Python files.** The code uses `os.environ.get("GROQ_API_KEY", "")` to read it safely.

---

## ▶️ Running the app

```bash
uv run streamlit run ieee_club_kef_v2_@2026.py
```

The app will open automatically at **http://localhost:8501**

---

## 🧪 How to use

1. Upload one or more PDF files using the **sidebar**
2. Choose a **LLM model** (recommended: `llamma` — most stable on Groq)
3. Set the **temperature** (0.0 = precise, 1.0 = creative)
4. Ask questions about your documents in the chat

---

## ⚠️ Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `No module named streamlit` | Wrong Python environment | Use `uv run` instead of `python -m` |
| `greenlet` install error | Python 3.14 incompatibility | Use Python 3.11 strictly |
| `DLL initialization failed` | Wrong torch version | Install `torch==2.1.0` with the PyTorch CPU index |
| `model_decommissioned` | Groq deprecated the model | Switch to `llamma` or `gemma` in the sidebar |
| `organization_restricted` | Groq API key blocked | Create a new Groq account and generate a new key |
| `No module named langchain.chains` | Wrong langchain version | Use `langchain==0.2.16` exactly |
| GitHub push rejected (secret detected) | API key in code | Remove key from code, use `.env` file, delete and recreate the repo |

---

## 📁 Project structure

```
pdf-chatbot/
├── ieee_club_kef_v2_@2026.py   # Main application
├── ieee_club_kef_@2026.py      # Original version
├── pyproject.toml               # Project dependencies
├── uv.lock                      # Lock file
├── .gitignore                   # Ignored files
├── .env                         # API key (not committed)
└── data/
    └── 2405.01564v1.pdf         # Sample PDF for testing
```

---

## 🔑 Models available on Groq

| Name in app | Model ID | Notes |
|---|---|---|
| `llamma` | `llama-3.1-8b-instant` | ✅ Recommended — fast and stable |
| `gemma` | `gemma2-9b-it` | ✅ Good alternative |
| `deepseek` | `deepseek-r1-distill-llama-70b-specdec` | ⚠️ May be decommissioned |

Check available models at: https://console.groq.com/dashboard/limits

---

## 📝 License

MIT
