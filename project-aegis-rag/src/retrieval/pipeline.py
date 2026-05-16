# src/retrieval/pipeline.py

from typing import List, Dict, Optional
from src.retrieval.query_transformer import QueryTransformer
from src.retrieval.retriever import PolicyRetriever


class RetrievalPipeline:
    """
    Full Retrieval Pipeline for Project Aegis (Part 2)
    
    Supports flexible combinations:
    - With / Without Multi-Query Expansion (MQE)
    - With / Without Category-based Pre-filtering
    """

    def __init__(self):
        self.query_transformer = QueryTransformer()
        self.retriever = PolicyRetriever()

    # ========================== SECTION 1: MULTI-QUERY EXPANSION + RETRIEVAL ==========================
    def _retrieve_with_mqe(
        self, 
        question: str, 
        top_k: int = 15,
        policy_category: Optional[str] = None
    ) -> List[Dict]:
        """
        Section 1: Multi-Query Expansion → Send each query to Retriever
        """
        print(f"🔀 [MQE] Expanding query: '{question}'")
        
        expansion_result = self.query_transformer.expand_query(question, num_queries=4)
        expanded_queries = expansion_result["expanded_queries"]
        
        all_results = []
        seen_ids = set()  # Deduplication

        for q in expanded_queries:
            print(f"   → Searching with: '{q}'")
            
            if policy_category:
                results = self.retriever.retrieve_with_category_filter(
                    query=q, 
                    policy_category=policy_category.lower(),
                    top_k=top_k
                )
            else:
                results = self.retriever.retrieve(query=q, top_k=top_k)
            
            # Deduplicate by chunk_id
            for res in results:
                if res["chunk_id"] not in seen_ids:
                    seen_ids.add(res["chunk_id"])
                    all_results.append(res)

        print(f"✅ MQE completed: {len(expanded_queries)} queries → {len(all_results)} unique chunks")
        return all_results

    # ========================== SECTION 2: SINGLE QUERY RETRIEVAL ==========================
    def _retrieve_single(
        self, 
        question: str, 
        top_k: int = 15,
        policy_category: Optional[str] = None
    ) -> List[Dict]:
        """
        Simple single query retrieval (with optional category filter)
        """
        if policy_category:
            return self.retriever.retrieve_with_category_filter(
                query=question, 
                policy_category=policy_category.lower(),
                top_k=top_k
            )
        else:
            return self.retriever.retrieve(query=question, top_k=top_k)

    # ========================== MAIN PUBLIC METHOD ==========================
    def retrieve(
        self,
        question: str,
        use_mqe: bool = True,
        policy_category: Optional[str] = None,
        top_k: int = 15,
        final_top_k: int = 10
    ) -> Dict:
        """
        Main retrieval method with flexible options.
        
        Parameters:
            use_mqe: True = Use Multi-Query Expansion
            policy_category: Filter by category (security, training, travel, work_policies)
            top_k: Chunks to retrieve per query
            final_top_k: Final number of unique results to return
        """
        if use_mqe:
            results = self._retrieve_with_mqe(
                question=question,
                top_k=top_k,
                policy_category=policy_category
            )
        else:
            results = self._retrieve_single(
                question=question,
                top_k=top_k,
                policy_category=policy_category
            )

        # Sort by score and limit final results
        results.sort(key=lambda x: x["score"], reverse=True)
        final_results = results[:final_top_k]

        return {
            "question": question,
            "use_mqe": use_mqe,
            "policy_category": policy_category,
            "total_retrieved": len(results),
            "final_results": final_results,
            "num_expanded_queries": len(self.query_transformer.expand_query(question, 1)["expanded_queries"]) 
                                  if use_mqe else 1
        }


# ========================== TEST / DEMO ==========================
if __name__ == "__main__":
    pipeline = RetrievalPipeline()
    
    #test_question = "What is the allowance for taxi in international travel?"
    test_question = "Can I expense a taxi from the airport?" 
    
    print("="*80)
    print("TEST 1: With MQE + Category Filter")
    print("="*80)
    result1 = pipeline.retrieve(
        question=test_question,
        use_mqe=True,
        policy_category="travel",
        top_k=10,
        final_top_k=8
    )
    
    print(f"\nRetrieved {result1['total_retrieved']} chunks → showing top {len(result1['final_results'])}")
    
    print("\n" + "="*80)
    print("TEST 2: Without MQE (Single Query)")
    print("="*80)
    result2 = pipeline.retrieve(
        question=test_question,
        use_mqe=False,
        policy_category=None,   # No category filter
        top_k=15
    )
    
    print(f"Retrieved {len(result2['final_results'])} chunks")