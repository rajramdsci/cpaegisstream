# src/retrieval/retriever.py

from typing import List, Dict, Optional
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from config.settings import settings


class PolicyRetriever:
    """
    Responsible for retrieving relevant chunks from Pinecone.
    Supports metadata filtering as per Part 2 - Step 2.
    """

    def __init__(
        self,
        index_name: str = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2" #"BAAI/bge-large-en-v1.5"
    ):
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index_name = index_name or settings.PINECONE_INDEX_NAME
        self.index = self.pc.Index(self.index_name)
        
        print(f"🔌 Connected to Pinecone index: {self.index_name}")
        self.embedder = SentenceTransformer(embedding_model)

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for query."""
        return self.embedder.encode(text, normalize_embeddings=True).tolist()

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve top_k most relevant chunks with optional metadata filter.
        """
        query_embedding = self._get_embedding(query)

        # Prepare query parameters
        query_params = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True
        }

        if filter:
            query_params["filter"] = filter

        try:
            response = self.index.query(**query_params)
            
            results = []
            for match in response.matches:
                results.append({
                    "chunk_id": match.id,
                    "score": float(match.score),
                    "chunk_text": match.metadata.get("chunk_text", ""),
                    "metadata": match.metadata
                })
            
            print(f"📥 Retrieved {len(results)} chunks for query (top_k={top_k})")
            return results

        except Exception as e:
            print(f"❌ Pinecone retrieval error: {e}")
            return []

    def retrieve_with_category_filter(
        self, 
        query: str, 
        policy_category: str = None,
        top_k: int = 20
    ) -> List[Dict]:
        """Convenience method with policy_category pre-filter."""
        filter_dict = {"policy_category": policy_category} if policy_category else None
        return self.retrieve(query, top_k=top_k, filter=filter_dict)


# ========================== TEST ==========================
if __name__ == "__main__":
    retriever = PolicyRetriever()
    
    test_query = "What is the policy for international travel per diem?"
    
    results = retriever.retrieve(test_query, top_k=10)
    
    print("\n" + "="*60)
    print("RETRIEVAL TEST RESULTS")
    print("="*60)
    for i, r in enumerate(results[:3], 1):
        print(f"{i}. Score: {r['score']:.4f} | {r['metadata'].get('h1_header', 'N/A')}")
        print(f"   → {r['chunk_text'][:150]}...\n")