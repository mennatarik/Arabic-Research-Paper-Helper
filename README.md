# Arabic Research Paper Helper

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mennatarik/Arabic-Research-Paper-Helper/blob/main/notebooks/run_in_colab.ipynb)

Upload **English research papers** (PDF), ask questions **in Arabic**, and get an Arabic answer with the
source page, a confidence score, a category, and a follow-up question.

It is a simple cross-lingual RAG (retrieval-augmented generation) project built with LangChain.

## How it works

```
PDF ──► pypdf ──► split into chunks ──► bge-m3 embeddings ──► FAISS index
                                                                  │
Arabic question ──► bge-m3 embedding ──► top 4 chunks ◄───────────┘
                                              │
                                  Qwen3-4B (4-bit) writes the Arabic answer
                                              │
                         answer + category + follow-up + confidence + sources
```

| Step | Tool |
|---|---|
| PDF parsing | `PyPDFLoader` (pypdf) |
| Chunking | `RecursiveCharacterTextSplitter` (1000 characters, 150 overlap) |
| Embeddings | `BAAI/bge-m3` (multilingual, so Arabic questions match English text) |
| Vector store | FAISS (saved in `data/faiss_index/`) |
| LLM | `Qwen/Qwen3-4B-Instruct-2507`, loaded in 4-bit (needs an NVIDIA GPU) |
| UI | Streamlit (public link through ngrok) |

## Files

| File | What it does |
|---|---|
| `rag.py` | The whole pipeline: `index_pdf`, `retrieve`, `ask`, and the settings at the top |
| `app.py` | The Streamlit page: upload and index, then ask |
| `run_ngrok.py` | Starts Streamlit and prints a public ngrok URL |
| `notebooks/run_in_colab.ipynb` | Launcher for Google Colab (clones this repo, installs, runs) |
| `requirements.txt` | Libraries |

## Run in Google Colab (recommended, needs a GPU)

1. Open `notebooks/run_in_colab.ipynb` in Colab and choose **Runtime → T4 GPU**.
2. Add `NGROK_AUTHTOKEN` under **Secrets** (free token from dashboard.ngrok.com).
3. Put your GitHub name in the clone cell, then run the cells top to bottom.
4. Open the **Public URL** printed by the last cell.

Or from a terminal on a machine with an NVIDIA GPU:

```bash
pip install -r requirements.txt
export NGROK_AUTHTOKEN=...
python run_ngrok.py          # or: streamlit run app.py
```

The first question is slow because Qwen and bge-m3 are downloaded and loaded.

## Settings (top of `rag.py`)

| Name | Meaning |
|---|---|
| `EMBED_MODEL`, `LLM_MODEL` | Which models to use |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` | Chunking and retrieval |

## Confidence score

The score is the cosine similarity of the best retrieved chunk, shown as a percentage. It is **not** the
probability that the answer is correct. It looks low because:

- an Arabic question compared with an English chunk usually scores about 0.3 to 0.7, even for a correct match;
- each chunk is long (1000 characters) and mixes several topics, which lowers the match;
- it measures retrieval only, not whether Qwen wrote a good answer.

Read it as "how close was the best chunk", and judge the answer by checking the cited page in **Sources**.
