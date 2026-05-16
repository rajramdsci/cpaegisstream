# src/retrieval/pipeline.py

from typing import List, Dict, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.query_transformer import QueryTransformer
from src.retrieval.retriever import PolicyRetriever
from src.retrieval.reranker import Reranker


class RetrievalPipeline:
    """
    Optimized Project Aegis Pipeline with Early Intent Detection
    """

    def __init__(self):
        self.query_transformer = QueryTransformer()
        self.retriever = PolicyRetriever()
        self.reranker = Reranker(k=60)
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.chat_history: List[Dict] = []

    # ====================== INTENT DETECTION ======================
    def _is_conversational_query(self, question: str) -> bool:
        """Quick heuristic + lightweight LLM check for meta/conversational questions"""
        q_lower = question.lower().strip()
        
        meta_phrases = [
            "what was my first", "what did i ask", "previous question", 
            "earlier question", "last question", "conversation history",
            "summarize our chat", "what have we discussed", "recall",
            "what was the first", "my first question"
        ]
        
        if any(phrase in q_lower for phrase in meta_phrases):
            return True
        
        # Optional: Light LLM intent classification (uncomment if needed)
        # return self._llm_intent_check(question)
        return False

    def _llm_intent_check(self, question: str) -> bool:
        """Fallback LLM-based intent detection"""
        prompt = f"""Classify this user message as either:
A) Policy Question (needs retrieval)
B) Conversational/Meta Question (about chat history)

Message: "{question}"

Answer with only A or B."""
        
        try:
            resp = self.llm.invoke(prompt)
            return resp.content.strip().upper().startswith("B")
        except:
            return False

    # ====================== MEMORY-BASED ANSWER ======================
    def _answer_from_memory(self, question: str) -> str:
        """Answer meta questions directly from chat history"""
        if not self.chat_history:
            return "This is our first conversation. You haven't asked any questions yet."

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Aegis, a helpful assistant. Answer the user's question about our conversation history."),
            *self._get_full_history_messages(),
            ("human", question)
        ])

        messages = prompt.format_messages()
        response = self.llm.invoke(messages)
        return response.content.strip()

    def _get_full_history_messages(self):
        messages = []
        for entry in self.chat_history:
            messages.append(HumanMessage(content=entry["question"]))
            messages.append(AIMessage(content=entry["answer"]))
        return messages

    # ====================== CORE RETRIEVAL (Policy Questions) ======================
    def _base_retrieve(self, query: str, policy_category: Optional[str] = None, top_k: int = 25) -> List[Dict]:
        if policy_category:
            return self.retriever.retrieve_with_category_filter(query, policy_category.lower().strip(), top_k)
        return self.retriever.retrieve(query, top_k)

    def _mqe_retrieve(self, question: str, policy_category=None, top_k=25) -> List[Dict]:
        expansion = self.query_transformer.expand_query(question, num_queries=4)
        all_results = [self._base_retrieve(q, policy_category, top_k) for q in expansion["expanded_queries"]]
        return self.reranker.reciprocal_rank_fusion(all_results, top_n=30)

    def _hyde_retrieve(self, question: str, policy_category=None, top_k=25) -> List[Dict]:
        hyde_doc = self.query_transformer.generate_hyde_document(question)
        results = self._base_retrieve(hyde_doc, policy_category, top_k)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results

    # ====================== FINAL ANSWER GENERATION ======================
    def _generate_policy_answer(self, question: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "I don't have sufficient information in the current policies to answer this accurately."

        context_text = "\n\n".join([
            f"Source: {chunk['metadata'].get('h1_header', 'Policy')} "
            f"({chunk['metadata'].get('policy_category', 'General')})\n"
            f"{chunk['chunk_text']}"
            for chunk in context_chunks
        ])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are **Aegis**, an expert corporate policy assistant.
Answer accurately using ONLY the provided policy context.
Cite relevant sections when possible."""),
            ("system", f"Policy Context:\n{context_text}"),
            *self._get_full_history_messages(),
            ("human", question)
        ])

        messages = prompt.format_messages()
        response = self.llm.invoke(messages)
        answer = response.content.strip()

        self._update_memory(question, answer)
        return answer

    def _update_memory(self, question: str, answer: str):
        self.chat_history.append({"question": question, "answer": answer})
        if len(self.chat_history) > 10:
            self.chat_history.pop(0)

    # ====================== MAIN OPTIMIZED PIPELINE ======================
    def retrieve_and_answer(
        self,
        question: str,
        use_mqe: bool = True,
        use_hyde: bool = True,
        use_reranker: bool = True,
        policy_category: Optional[str] = None,
        top_k: int = 25,
        final_top_n: int = 6
    ) -> Dict:
        """Smart Optimized Pipeline with Early Intent Detection"""

        print(f"\n🔍 User: {question}")

        # === EARLY CHECK: Conversational / Meta Question ===
        if self._is_conversational_query(question):
            print("💬 Detected conversational/meta question → Answering from memory")
            answer = self._answer_from_memory(question)
            self._update_memory(question, answer)
            return {
                "question": question,
                "method": "Memory Only (Conversational)",
                "answer": answer,
                "final_results": [],
                "from_memory": True
            }

        # === POLICY QUESTION → Full Retrieval ===
        print("📄 Policy-related question → Running retrieval pipeline")

        if use_mqe and use_hyde:
            cand1 = self._mqe_retrieve(question, policy_category, top_k)
            cand2 = self._hyde_retrieve(question, policy_category, top_k)
            candidates = cand1 + cand2
            method = "MQE + HyDE"
        elif use_mqe:
            candidates = self._mqe_retrieve(question, policy_category, top_k)
            method = "MQE + RRF"
        elif use_hyde:
            candidates = self._hyde_retrieve(question, policy_category, top_k)
            method = "HyDE"
        else:
            candidates = self._base_retrieve(question, policy_category, top_k)
            method = "Single Query"

        if use_reranker and candidates:
            final_chunks = self.reranker.cross_encoder_rerank(question, candidates, top_n=final_top_n)
        else:
            final_chunks = candidates[:final_top_n]

        answer = self._generate_policy_answer(question, final_chunks)

        return {
            "question": question,
            "method": method,
            "answer": answer,
            "final_results": final_chunks,
            "from_memory": False
        }


# ========================== TEST CONVERSATION ==========================
if __name__ == "__main__":
    pipeline = RetrievalPipeline()
    
    print("=== Conversation Test ===")
    result = pipeline.retrieve_and_answer("What are the tiers in Annual Stipend ?")
    print("\nFinal Answer:", result["answer"])
    pipeline.retrieve_and_answer("Tell me about international travel per diem")
    print("\nFinal Answer:", result["answer"])
    result = pipeline.retrieve_and_answer("What was my first question?")
    
    print("\nFinal Answer:", result["answer"])