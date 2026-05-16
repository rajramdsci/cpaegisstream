# src/retrieval/reranker.py

from typing import List, Dict
from collections import defaultdict

from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from config.settings import settings


class Reranker:
    """
    Advanced Reranker for Project Aegis:
    - Reciprocal Rank Fusion (RRF) for Multi-Query Expansion
    - Cross-Encoder Reranking using **Cohere Reranker** (Recommended)
    """

    def __init__(self, k: int = 60):
        self.k = k  # RRF constant
        
        # Initialize Cohere Cross-Encoder Reranker
        print("🔄 Loading Cohere Reranker...")
        self.cohere_reranker = CohereRerank(
            cohere_api_key=settings.COHERE_API_KEY,   # Add this to .env
            model="rerank-english-v3.0",             # Strong & cost-effective
            # model="rerank-english-v4.0"            # Newer & more powerful (if available)
            top_n=10
        )
        print("✅ Cohere Reranker initialized successfully")

    # ====================== RRF (for MQE) ======================
    def reciprocal_rank_fusion(
        self, 
        multi_query_results: List[List[Dict]], 
        top_n: int = 15
    ) -> List[Dict]:
        """Reciprocal Rank Fusion for combining multiple queries"""
        if not multi_query_results:
            return []

        rrf_scores = defaultdict(float)
        chunk_metadata = {}

        for query_results in multi_query_results:
            for rank, chunk in enumerate(query_results, start=1):
                chunk_id = chunk["chunk_id"]
                rrf_scores[chunk_id] += 1.0 / (rank + self.k)

                if chunk_id not in chunk_metadata:
                    chunk_metadata[chunk_id] = chunk.copy()

        # Sort by RRF score
        ranked = []
        for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            chunk = chunk_metadata[chunk_id]
            chunk["rrf_score"] = round(score, 6)
            ranked.append(chunk)

        return ranked[:top_n]

    # ====================== COHERE CROSS-ENCODER RERANKING ======================
    def cross_encoder_rerank(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_n: int = 5
    ) -> List[Dict]:
        """
        Use Cohere Reranker (Cross-Encoder) for final high-quality ranking.
        """
        if not chunks:
            return []

        print(f"🔄 Cohere Cross-Encoder reranking {len(chunks)} candidate chunks...")

        # Convert to LangChain Document format
        docs = [
            Document(
                page_content=chunk["chunk_text"],
                metadata={
                    **chunk.get("metadata", {}),
                    "chunk_id": chunk["chunk_id"],
                    "original_score": chunk.get("score")
                }
            )
            for chunk in chunks
        ]

        # Perform reranking
        reranked_docs = self.cohere_reranker.compress_documents(docs, query)

        # Convert back to our dict format
        final_results = []
        for doc in reranked_docs[:top_n]:
            result = {
                "chunk_id": doc.metadata.get("chunk_id"),
                "chunk_text": doc.page_content,
                "metadata": doc.metadata,
                "rerank_score": doc.metadata.get("relevance_score"),   # Cohere provides this
                "score": doc.metadata.get("relevance_score")
            }
            final_results.append(result)

        print(f"✅ Cohere reranking completed → Top {len(final_results)} results")
        return final_results

    # ====================== COMBINED RERANKING ======================
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict], 
        top_n: int = 5
    ) -> List[Dict]:
        """
        Full reranking pipeline: RRF (if MQE) → Cohere Cross-Encoder
        """
        return self.cross_encoder_rerank(query, chunks, top_n=top_n)