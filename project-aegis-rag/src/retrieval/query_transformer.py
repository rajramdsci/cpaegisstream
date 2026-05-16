# src/retrieval/query_transformer.py

from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings


class QueryTransformer:
    """
    Handles Query Transformation techniques from Project Aegis Part 2 - Step 1:
    - Multi-Query Expansion (MQE)
    - HyDE (Hypothetical Document Embeddings)
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            max_tokens=500
        )

        # ==================== MULTI-QUERY PROMPT ====================
        self.multi_query_prompt = ChatPromptTemplate.from_template(
            """You are an expert at reformulating user questions to retrieve relevant corporate policy documents.

Given the original user question, generate **{num_queries}** different versions that would help retrieve more comprehensive results.
Focus on different phrasings, synonyms, and policy-specific terminology.

Original Question: {question}

Provide only the alternative questions, numbered 1 to {num_queries}."""
        )

        # ==================== HyDE PROMPT ====================
        self.hyde_prompt = ChatPromptTemplate.from_template(
            """You are a helpful corporate policy expert. 
Write a detailed, hypothetical answer to the user's question as if you are quoting directly from the official company policy document.
Use formal policy language, include specific rules, conditions, and examples where possible.

User Question: {question}

Hypothetical Policy Answer:"""
        )

    # ====================== MULTI-QUERY ======================
    def generate_multi_queries(self, question: str, num_queries: int = 4) -> List[str]:
        """Generate multiple reformulated versions of the user query."""
        prompt = self.multi_query_prompt.format(
            question=question,
            num_queries=num_queries
        )
        
        response = self.llm.invoke(prompt)
        generated_text = response.content.strip()

        queries = []
        for line in generated_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith(('-', '*'))):
                cleaned = line.split('. ', 1)[-1].strip() if '. ' in line else line
                cleaned = cleaned.lstrip('-* ').strip()
                if cleaned and len(cleaned) > 10:
                    queries.append(cleaned)

        # Fallback + always include original
        if not queries:
            queries = [question]
        if question not in queries:
            queries.insert(0, question)

        print(f"🔀 Generated {len(queries)} queries for: '{question}'")
        return queries[:num_queries + 1]

    # ====================== HyDE ======================
    def generate_hyde_document(self, question: str) -> str:
        """
        Generate a Hypothetical Document (fake answer) using LLM.
        """
        prompt = self.hyde_prompt.format(question=question)
        response = self.llm.invoke(prompt)
        hyde_doc = response.content.strip()
        
        print(f"📝 HyDE generated hypothetical document ({len(hyde_doc)} chars)")
        return hyde_doc

    # ====================== MAIN METHODS ======================
    def expand_query(self, question: str, num_queries: int = 4) -> Dict:
        """Multi-Query Expansion only (existing behavior)"""
        queries = self.generate_multi_queries(question, num_queries)
        
        return {
            "original_query": question,
            "expanded_queries": queries,
            "num_expanded": len(queries),
            "method": "multi_query"
        }

    def generate_hyde(self, question: str) -> Dict:
        """HyDE only"""
        hyde_doc = self.generate_hyde_document(question)
        
        return {
            "original_query": question,
            "hyde_document": hyde_doc,
            "method": "hyde"
        }

    def expand_with_hyde(self, question: str, num_queries: int = 3) -> Dict:
        """
        Combined approach: Multi-Query Expansion + HyDE
        Returns both expanded queries and HyDE document.
        """
        queries = self.generate_multi_queries(question, num_queries)
        hyde_doc = self.generate_hyde_document(question)

        return {
            "original_query": question,
            "expanded_queries": queries,
            "hyde_document": hyde_doc,
            "method": "multi_query + hyde",
            "num_expanded": len(queries)
        }


# ========================== TEST / USAGE ==========================
if __name__ == "__main__":
    transformer = QueryTransformer()

    test_question = "Can I expense a taxi from the airport to the hotel?"

    print("=" * 80)
    print("1. MULTI-QUERY EXPANSION")
    print("=" * 80)
    mq_result = transformer.expand_query(test_question, num_queries=4)
    for i, q in enumerate(mq_result["expanded_queries"], 1):
        print(f"{i}. {q}")

    print("\n" + "=" * 80)
    print("2. HyDE (Hypothetical Document)")
    print("=" * 80)
    hyde_result = transformer.generate_hyde(test_question)
    print(hyde_result["hyde_document"][:500] + "..." if len(hyde_result["hyde_document"]) > 500 else hyde_result["hyde_document"])

    print("\n" + "=" * 80)
    print("3. COMBINED (MQE + HyDE)")
    print("=" * 80)
    combined = transformer.expand_with_hyde(test_question)
    print("Generated successfully!")