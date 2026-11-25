"""
HybridRAG：混合检索器（TF-IDF 关键词路 + bge-m3 语义路 + RRF 融合）
================================================================
这是整个项目的"检索内核"，被 LangGraph 包装成一个工具节点调用。
设计目标：自包含、可单测、接口极简：
    rag = HybridRAG()
    rag.add_text(text, source="xxx.md")   # 增量建索引
    hits = rag.retrieve("员工请假怎么请", top_k=4)
    rag.stats()                            # 看索引状态

注：语义路依赖本地 Ollama 的 bge-m3 嵌入；Ollama 没起时自动降级为纯 TF-IDF。
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ollama


class HybridRAG:
    def __init__(self, embed_model: str = None, use_semantic: bool = True, rrf_k: int = 60):
        self.embed_model = embed_model or os.getenv("EMBED_MODEL", "bge-m3")
        self.use_semantic = use_semantic
        self.rrf_k = rrf_k  # RRF 常数，论文常用 60

        self.docs = []          # [{"text":..., "source":...}, ...]
        self.vectorizer = None  # sklearn TF-IDF
        self.tfidf_matrix = None
        self.embeds = None      # np.ndarray (n, d)

        self._client = ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    # ---------- 索引构建 ----------
    def add_text(self, text: str, source: str = "unknown") -> None:
        """把一段文本按空行切块后加入索引。"""
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        for c in chunks:
            self.docs.append({"text": c, "source": source})
        self._rebuild()

    def _embed(self, texts):
        """调用 Ollama 拿 bge-m3 嵌入向量。"""
        out = []
        for t in texts:
            try:
                r = self._client.embeddings(model=self.embed_model, prompt=t)
                out.append(r["embedding"])
            except Exception as e:
                # 嵌入失败（Ollama 未启动 / 模型未拉取）时退化为零向量
                # 此时整体退化为纯 TF-IDF（use_semantic 实际失效）
                print(f"[warn] embed failed ({self.embed_model}): {e}")
                out.append([0.0])
        return np.array(out, dtype=np.float32)

    def _rebuild(self):
        if not self.docs:
            return
        corpus = [d["text"] for d in self.docs]
        self.vectorizer = TfidfVectorizer().fit(corpus)
        self.tfidf_matrix = self.vectorizer.transform(corpus)
        if self.use_semantic:
            self.embeds = self._embed(corpus)

    # ---------- 检索 ----------
    @staticmethod
    def _rrf(sim: np.ndarray, k: int) -> np.ndarray:
        """Reciprocal Rank Fusion：把相似度排好序后换成 1/(rank+k) 再相加。"""
        order = np.argsort(-sim)
        scores = np.zeros(len(sim), dtype=np.float64)
        for rank, idx in enumerate(order):
            scores[idx] = 1.0 / (rank + 1 + k)
        return scores

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        if not self.docs:
            return []
        tfidf_q = self.vectorizer.transform([query])
        tfidf_sim = cosine_similarity(tfidf_q, self.tfidf_matrix)[0]

        if self.use_semantic and self.embeds is not None and self.embeds.shape[1] > 1:
            q_emb = self._embed([query])[0]
            sem_sim = cosine_similarity([q_emb], self.embeds)[0]
        else:
            sem_sim = np.zeros(len(self.docs))

        combined = self._rrf(tfidf_sim, self.rrf_k) + self._rrf(sem_sim, self.rrf_k)

        top_idx = np.argsort(-combined)[:top_k]
        return [
            {
                "text": self.docs[i]["text"],
                "source": self.docs[i]["source"],
                "score": float(combined[i]),
            }
            for i in top_idx
        ]

    # ---------- 状态 ----------
    def stats(self) -> dict:
        return {
            "num_chunks": len(self.docs),
            "sources": sorted({d["source"] for d in self.docs}),
            "semantic_enabled": self.use_semantic and self.embeds is not None,
            "embed_model": self.embed_model,
        }
