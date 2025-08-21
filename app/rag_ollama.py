import os, json, glob, re, faiss, numpy as np, pandas as pd
import ollama
from pathlib import Path

# ----------------------------
# Config
# ----------------------------
STORE_DIR = Path("rag_store")
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL  = "llama3"
CHUNK_SIZE  = 900          # ~900 chars per chunk
CHUNK_OVERLAP = 150        # ~150 char overlap
TOP_K = 5
MIN_SIM_THRESHOLD = 0.25   # below this -> "not in KB"

STORE_DIR.mkdir(exist_ok=True)

def read_textlike(path: Path) -> str:
    p = str(path).lower()
    if p.endswith((".md", ".txt")):
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    elif p.endswith(".csv"):
        # assume columns 'question' and 'answer' if present; else join all cells
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        if "question" in cols and "answer" in cols:
            qcol, acol = cols.index("question"), cols.index("answer")
            return "\n\n".join(f"Q: {r[qcol]}\nA: {r[acol]}" for r in df.itertuples(index=False, name=None))
        return "\n\n".join(" ".join(map(str, row)) for row in df.itertuples(index=False, name=None))
    else:
        return ""

def clean_text(s: str) -> str:
    # light cleanup to avoid whitespace
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    s = clean_text(s)
    chunks = []
    i = 0
    while i < len(s):
        chunk = s[i:i+size]
        chunks.append(chunk)
        i += max(1, size - overlap)
    return chunks

def embed_texts(texts):
    # Ollama embeddings API returns one vector per input string.
    vecs = []
    for t in texts:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=t)
        vecs.append(np.array(resp["embedding"], dtype="float32"))
    return np.vstack(vecs)

def l2_normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

class RagIndex:
    def __init__(self, store_dir=STORE_DIR):
        self.store_dir = Path(store_dir)
        self.meta_path = self.store_dir / "meta.json"
        self.faiss_path = self.store_dir / "faiss.index"
        self.texts_path = self.store_dir / "texts.jsonl"
        self.index = None
        self.texts = []  # parallel to vectors

    def build_or_load(self, docs_glob="data/faq/*"):
        if self.faiss_path.exists() and self.texts_path.exists():
            self._load()
            return

        # 1) Read & chunk
        all_chunks = []
        sources = []
        for fp in glob.glob(docs_glob):
            raw = read_textlike(Path(fp))
            if not raw:
                continue
            for ch in chunk_text(raw):
                all_chunks.append(ch)
                sources.append(os.path.basename(fp))

        if not all_chunks:
            raise RuntimeError("No documents found under data/faq/. Add .md/.txt or a .csv with question/answer.")

        # 2) Embed and normalize
        embs = embed_texts(all_chunks)
        embs = l2_normalize(embs).astype("float32")

        # 3) FAISS index (cosine via inner product on normalized vectors)
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embs)

        # 4) Persist
        faiss.write_index(index, str(self.faiss_path))
        with open(self.texts_path, "w", encoding="utf-8") as f:
            for t, s in zip(all_chunks, sources):
                f.write(json.dumps({"text": t, "source": s}) + "\n")
        with open(self.meta_path, "w") as f:
            json.dump({"emb_model": EMBED_MODEL, "chunk_size": CHUNK_SIZE}, f)

        self.index = index
        self.texts = [json.loads(l) for l in open(self.texts_path, encoding="utf-8")]

    def _load(self):
        self.index = faiss.read_index(str(self.faiss_path))
        self.texts = [json.loads(l) for l in open(self.texts_path, encoding="utf-8")]

    def search(self, query, top_k=TOP_K):
        qv = embed_texts([query]).astype("float32")
        qv = l2_normalize(qv)
        sims, ids = self.index.search(qv, top_k)  # inner product = cosine
        sims, ids = sims[0], ids[0]
        results = []
        for score, idx in zip(sims, ids):
            if idx == -1:
                continue
            rec = self.texts[idx]
            results.append({"text": rec["text"], "source": rec["source"], "score": float(score)})
        return results

def build_prompt(question, passages):
    context = "\n\n---\n\n".join([p["text"] for p in passages])
    return f"""You are a helpful RMIT FAQ assistant. Answer using ONLY the context.
If the answer is not in the context, say: "I'm not sure based on our FAQ."

Question: {question}

Context:
{context}

Rules:
- Be concise and factual.
- If uncertain or out-of-scope, explicitly say you're not sure.
"""

def answer_with_ollama(question, passages):
    prompt = build_prompt(question, passages)
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )
    return resp["message"]["content"]

if __name__ == "__main__":
    idx = RagIndex()
    idx.build_or_load()

    print("RAG ready. Ask questions (Ctrl+C to exit).")
    while True:
        q = input("\nYou: ").strip()
        if not q: 
            continue
        hits = idx.search(q, TOP_K)

        # Not-in-KB guard (like Walert's out-of-KB handling)
        if not hits or (hits[0]["score"] < MIN_SIM_THRESHOLD):
            print("Bot: I'm not sure based on our FAQ.")
            continue

        print("\n[Top sources]")
        for h in hits:
            print(f"- {h['source']} (score={h['score']:.2f})")

        out = answer_with_ollama(q, hits)
        print("\nBot:", out)