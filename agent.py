"""
Fully-local Retrieval-Augmented Chatbot
======================================

• Embeddings:  all-MiniLM-L6-v2 (384-d).
• Vector store: in-memory cosine search (no FAISS).
• LLM:         tiiuae/falcon-7b-instruct, 4-bit quantised.
"""

import os, sys, numpy as np, torch, re
from parser.web import parse_url
from parser.pdf import extract_study_plan, post_process_table
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

DOC_DIR     = Path("docs")
EMB_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "tiiuae/Falcon3-3B-Instruct"
CHUNK_LEN   = 500
CHUNK_OVERLAP = 50
TOP_K       = 3


# ---------- Vector store (simple, in-RAM) ----------
class MemoryStore:
    def __init__(self):
        self.vectors: List[np.ndarray] = []
        self.texts:   List[str]        = []
        self.ids:     List[str]        = []

    def add(self, vecs: np.ndarray, texts: List[str], ids: List[str]):
        self.vectors.extend(vecs)
        self.texts.extend(texts)
        self.ids.extend(ids)

    def search(self, q_vec: np.ndarray, k: int = TOP_K):
        if not self.vectors:
            return [], []
        sims = cosine_similarity([q_vec], np.array(self.vectors))[0]
        top   = sims.argsort()[::-1][:k]
        return sims[top], top


# ---------- Chatbot ----------
class RAGBot:
    def __init__(self):
        print("Loading embedder …")
        self.embedder = SentenceTransformer(EMB_MODEL)

        print("Loading Falcon-7B-Instruct (4-bit) …")
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        self.model     = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            quantization_config=cfg,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.store = MemoryStore()
        self.meta  = []   # keeps chunk metadata

    # ---------- Utility ----------
    @staticmethod
    def _chunks(text: str, size: int = CHUNK_LEN, overlap: int = CHUNK_OVERLAP):
        words = text.split()
        for i in range(0, len(words), size - overlap):
            yield " ".join(words[i:i + size])

    # ---------- Indexing ----------
    def index_docs(self, folder: Path = DOC_DIR):
        if not folder.exists():
            folder.mkdir()
            print(f"Place .txt files inside {folder}/ and rerun.")
            sys.exit()

        files = list(folder.glob("*.txt"))
        if not files:
            print("No .txt documents found.")
            sys.exit()

        chunks, ids = [], []
        for fp in files:
            text = fp.read_text(encoding="utf-8", errors="ignore").strip()
            for i, chunk in enumerate(self._chunks(text)):
                chunks.append(chunk)
                ids.append(f"{fp.name}#{i}")

        print(f"Embedding {len(chunks)} chunks …")
        vecs = self.embedder.encode(chunks, show_progress_bar=True)
        self.store.add(vecs, chunks, ids)
        self.meta = [{"id": i, "text": t, "file": i.split("#")[0]}
                     for i, t in zip(ids, chunks)]
        print("Indexing complete.")

    # ---------- Retrieval ----------
    def retrieve(self, query: str, k: int = TOP_K):
        q_vec = self.embedder.encode([query])[0]
        sims, idxs = self.store.search(q_vec, k)
        return [(*self.meta[i].values(), float(s)) for i, s in zip(idxs, sims)]

    # ---------- Generation ----------
    def answer(self, question: str) -> Dict:
        ctx = self.retrieve(question)
        if not ctx:
            return {"answer": "Документы не найдены.", "sources": []}

        context_text = "\n\n".join(t for _, t, _, _ in ctx)
        prompt = (
            "You are a helpful assistant. Use ONLY the information in <context> "
            "to answer the question in Russian.\n\n"
            f"<context>\n{context_text}\n</context>\n\n"
            f"Вопрос: {question}\nОтвет:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        answer = full[len(prompt):].strip()

        return {
            "answer": answer,
            "sources": [
                {"file": f, "similarity": sim, "excerpt": t[:160] + ("…" if len(t) > 160 else "")}
                for f, t, _, sim in ctx
            ],
        }


# ---------- CLI ----------
def main():
    print('Parsing: https://abit.itmo.ru/program/master/ai...')
    _ = parse_url('https://abit.itmo.ru/program/master/ai', 'docs')
    print('Parsing: https://abit.itmo.ru/program/master/ai_product...', 'docs')
    _ = parse_url('https://abit.itmo.ru/program/master/ai_product', 'docs')

    print('Parsing PDFs...')
    df1 = extract_study_plan(Path('docs/pdf/10033-abit.pdf'))
    df1 = post_process_table(df1)
    df1['program'] = 'Искусственный интеллект'
    with open('docs/ai_courses.txt', 'w') as fd:
        fd.write(str(df1))
    df2 = extract_study_plan(Path('docs/pdf/10130-abit.pdf'))
    df2 = post_process_table(df2)
    df2['program'] = 'Управление ИИ-продуктами/AI Product'
    with open('docs/ai_product.txt', 'w') as fd:
        fd.write(str(df2))
    bot = RAGBot()
    print('Документация успешно собрана')
    bot.index_docs()

    print("\nЧат готов! (type 'exit' to quit)")
    while True:
        q = input("\nВы: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        res = bot.answer(q)
        print(f"\nБот: {res['answer']}\n")
        if res["sources"]:
            print("Источники:")
            for i, s in enumerate(res["sources"], 1):
                print(f"{i}. {s['file']}  (sim={s['similarity']:.3f})")
                print(f"   «{s['excerpt']}»")

if __name__ == "__main__":
    main()
