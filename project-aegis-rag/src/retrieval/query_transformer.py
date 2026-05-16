# src/retrieval/query_transformer.py

from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings


class QueryTransformer:
    """
    Handles Query Transformation techniques from Part 2 - Step 1:
    - Multi-Query Expansion
    - (HyDE will be added later)
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            max_tokens=400
        )

        self.multi_query_prompt = ChatPromptTemplate.from_template(
            """You are an expert at reformulating user questions to retrieve relevant corporate policy documents.
            
Given the original user question, generate **{num_queries}** different versions that would help retrieve more comprehensive and relevant results.
Focus on different phrasings, synonyms, and policy-specific terminology.

Original Question: {question}

Provide only the alternative questions, numbered 1 to {num_queries}."""
        )

    def generate_multi_queries(self, question: str, num_queries: int = 4) -> List[str]:
        """
        Generate multiple reformulated versions of the user query.
        """
        prompt = self.multi_query_prompt.format(
            question=question,
            num_queries=num_queries
        )
        
        response = self.llm.invoke(prompt)
        generated_text = response.content.strip()

        # Parse the numbered questions
        queries = []
        for line in generated_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Clean the line
                cleaned = line.split('. ', 1)[-1].strip() if '. ' in line else line
                cleaned = cleaned.lstrip('-* ').strip()
                if cleaned:
                    queries.append(cleaned)

        # Fallback: if parsing failed, return original + one variation
        if len(queries) == 0:
            queries = [question]

        # Always include the original query
        if question not in queries:
            queries.insert(0, question)

        print(f"🔀 Generated {len(queries)} queries for: '{question}'")
        return queries[:num_queries + 1]  # Limit to requested number + original

    def expand_query(self, question: str, num_queries: int = 4) -> Dict:
        """
        Main method: Returns expanded queries with metadata.
        """
        queries = self.generate_multi_queries(question, num_queries)
        
        return {
            "original_query": question,
            "expanded_queries": queries,
            "num_expanded": len(queries)
        }


# ========================== TEST / USAGE ==========================
if __name__ == "__main__":
    transformer = QueryTransformer()
    
    test_question = "Can I expense a taxi from the airport?"
    
    result = transformer.expand_query(test_question, num_queries=4)
    
    print("\n" + "="*60)
    print("MULTI-QUERY EXPANSION RESULT")
    print("="*60)
    print(f"Original : {result['original_query']}")
    print("\nExpanded Queries:")
    for i, q in enumerate(result['expanded_queries'], 1):
        print(f"  {i}. {q}")