"""Simple cross-lingual RAG with LangChain: English PDFs in, Arabic answers out."""
import re
from functools import lru_cache
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- settings
EMBED_MODEL = "BAAI/bge-m3"                  # one multilingual embedding model
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"    # check the exact name on Hugging Face
CHUNK_SIZE = 1000                            # characters
CHUNK_OVERLAP = 150
TOP_K = 4
PER_PAPER_K = 2   # chunks taken from each paper when comparing papers

BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "data" / "uploads"
INDEX_DIR = BASE / "data" / "faiss_index"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """أنت مرشد أبحاث يشرح الأوراق العلمية الإنجليزية للطلاب العرب بالعربية الفصحى المبسطة.

القواعد:
1. أجب فقط من المقتطفات المرفقة. إذا لم تجد الإجابة فيها اكتب بالضبط: لا أملك معلومات كافية في الورقة.
2. لا تخترع أرقاماً أو نتائج.
3. اكتب المصطلحات التقنية بالعربية ثم الإنجليزية بين قوسين عند أول ذكر.
4. استخدم جملاً قصيرة.
5. ضع رقم المقتطف مثل [1] بعد كل معلومة.

اكتب ردك بهذا الشكل بالضبط:
الإجابة: ...
التصنيف: (مشكلة البحث أو المنهج أو البيانات أو النتائج أو القيود أو تعريف أو أخرى)
سؤال المتابعة: ..."""

NOT_FOUND = "لا أملك معلومات كافية"


# ---------------------------------------------------------------- models (loaded once)
@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                 encode_kwargs={"normalize_embeddings": True})


# Embeddings are normalized, so inner product == cosine similarity (higher = more similar).
STRATEGY = DistanceStrategy.MAX_INNER_PRODUCT
_store = None   # FAISS index; None until the first PDF is indexed


def get_store():
    """Load the saved FAISS index from disk if it exists, else None."""
    global _store
    if _store is None and (INDEX_DIR / "index.faiss").exists():
        _store = FAISS.load_local(str(INDEX_DIR), get_embeddings(),
                                  allow_dangerous_deserialization=True,  # it is our own file
                                  distance_strategy=STRATEGY)
    return _store


@lru_cache(maxsize=1)
def get_llm():
    """Qwen in 4-bit (needs an NVIDIA GPU + bitsandbytes)."""
    import torch
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, quantization_config=bnb,
                                                 device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tok,
                    max_new_tokens=600, do_sample=False, return_full_text=False)
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))


# ---------------------------------------------------------------- indexing
def index_pdf(path) -> int:
    """pypdf -> split -> embed -> FAISS. Returns the number of chunks."""
    path = Path(path)
    pages = PyPDFLoader(str(path)).load()                  # one Document per page (page is 0-based)
    pages = [p for p in pages if p.page_content.strip()]   # only cleaning: drop empty pages
    for p in pages:
        p.metadata["doc"] = path.name

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("No text found (scanned PDF? it needs OCR).")

    global _store
    store = get_store()
    if store is None:
        store = FAISS.from_documents(chunks, get_embeddings(), distance_strategy=STRATEGY)
    else:
        old = [i for i, d in store.docstore._dict.items() if d.metadata.get("doc") == path.name]
        if old:                                   # re-indexing replaces the old version
            store.delete(old)
        store.add_documents(chunks)
    store.save_local(str(INDEX_DIR))
    _store = store
    return len(chunks)


def list_docs() -> list[str]:
    store = get_store()
    if store is None:
        return []
    return sorted({d.metadata["doc"] for d in store.docstore._dict.values()})


# ---------------------------------------------------------------- question answering
def retrieve(question: str, k: int = TOP_K, per_paper: bool = False):
    """Returns [(Document, similarity)], best first. With inner product the score is the similarity.
 
    per_paper=True takes the best chunks from EVERY paper (useful for comparing papers)."""
    store = get_store()
    if store is None:
        return []
    if not per_paper:
        results = store.similarity_search_with_score(question, k=k)
    else:
        results = []
        for doc in list_docs():
            # FAISS filters after searching fetch_k candidates, so search all chunks (fine for a few papers)
            results += store.similarity_search_with_score(
                question, k=PER_PAPER_K, filter={"doc": doc}, fetch_k=store.index.ntotal)
    results = sorted(results, key=lambda r: r[1], reverse=True)
    return [(doc, float(score)) for doc, score in results]
 
 
def _field(text: str, label: str) -> str:
    m = re.search(rf"{label}\s*:\s*(.*?)(?=\n(?:الإجابة|التصنيف|سؤال المتابعة)\s*:|\Z)", text, re.S)
    return m.group(1).strip() if m else ""


def ask(question: str, per_paper: bool = False) -> dict:
    hits = retrieve(question, per_paper=per_paper)
    if not hits:
        return {"answer": "لا توجد أوراق مفهرسة بعد.", "category": "", "follow_up": "",
                "confidence": 0, "found": False, "hits": []}

    context = "\n\n".join(
        f"[{i}] ({d.metadata['doc']}, p. {d.metadata['page'] + 1})\n{d.page_content}"
        for i, (d, _) in enumerate(hits, start=1)
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "السؤال:\n{question}\n\nالمقتطفات:\n{context}"),
    ])
    chain = prompt | get_llm() | StrOutputParser()
    text = chain.invoke({"question": question, "context": context})
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    answer = _field(text, "الإجابة") or text        # fall back to the raw text if the format is broken
    found = NOT_FOUND not in answer
    confidence = round(100 * max(hits[0][1], 0))     # raw top similarity, not calibrated
    return {
        "answer": answer,
        "category": _field(text, "التصنيف"),
        "follow_up": _field(text, "سؤال المتابعة"),
        "confidence": confidence if found else min(confidence, 20),
        "found": found,
        "hits": hits,
    }