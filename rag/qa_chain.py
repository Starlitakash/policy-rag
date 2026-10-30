"""
rag/qa_chain.py
---------------
Builds and executes the Question-Answering RAG Chain using LangChain and Groq LLM.

Sole AI Model: Groq
  Uses Groq's high-speed inference engine powered by LPU (Language Processing Units).
  Default model: `llama-3.1-8b-instant` (or `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`).

Grounding & Anti-Hallucination Policy:
  If the answer is NOT present in the retrieved policy context, the LLM is strictly instructed
  to reply: "I couldn't find this information in the uploaded policy documents."
"""

import os
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from rag.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    TOP_K
)
from rag.vector_store import VectorStoreManager
from rag.retriever import PolicyRetriever


# Strict RAG System Prompt designed to prevent hallucinations
SYSTEM_PROMPT_TEMPLATE = """You are an official Policy & Compliance Assistant.
Your sole purpose is to answer user questions strictly and accurately based ONLY on the provided policy context below.

CONTEXT FROM UPLOADED POLICY DOCUMENTS:
======================================
{context}
======================================

USER QUESTION: {question}

STRICT INSTRUCTIONS:
1. Answer the question relying ONLY on the facts directly mentioned in the CONTEXT above.
2. If the context does NOT contain enough information to answer the question with certainty, respond exactly with:
   "I couldn't find this information in the uploaded policy documents."
3. Do NOT assume, extrapolate, speculate, or use any outside knowledge not present in the context.
4. Keep your answer clear, polite, concise, and well-structured using bullet points where appropriate.
5. Whenever possible, mention relevant clause numbers or rule specifics (e.g. percentages, fees, timelines, limits) given in the text.
"""


def get_groq_llm(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0
) -> ChatGroq:
    """
    Instantiates the Groq Chat Model.
    
    Args:
        api_key: Optional Groq API key override. If None, reads from GROQ_API_KEY config.
        model_name: Optional Groq model name (default: 'llama-3.1-8b-instant').
        temperature: Set to 0.0 for deterministic, factual policy answers.
    """
    key = api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError(
            "Groq API key not found. Please set GROQ_API_KEY in your .env file or enter it in the frontend sidebar."
        )

    m_name = model_name or GROQ_MODEL
    return ChatGroq(
        model=m_name,
        api_key=key,
        temperature=temperature
    )


class PolicyQAEngine:
    """
    Complete RAG QA Engine that orchestrates Qdrant retrieval, prompt construction,
    Groq LLM invocation, and source citation packaging.
    """

    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        top_k: int = TOP_K
    ):
        self.vector_store_manager = vector_store_manager or VectorStoreManager()
        self.retriever = PolicyRetriever(self.vector_store_manager, top_k=top_k)
        self.prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
        self.output_parser = StrOutputParser()

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the end-to-end RAG pipeline for a given user question:
        1. Retrieve top-k chunks from Qdrant
        2. Format retrieved chunks into context string
        3. Build prompt and send to Groq LLM
        4. Return response with verified sources
        """
        # Step 1: Retrieve relevant chunks from Qdrant
        retrieved_docs = self.retriever.get_relevant_documents(question, top_k=top_k)
        sources = self.retriever.extract_sources_metadata(retrieved_docs)

        # Step 2: If no documents are in Qdrant at all
        if not retrieved_docs:
            return {
                "question": question,
                "answer": "I couldn't find this information in the uploaded policy documents.",
                "sources": [],
                "context_used": ""
            }

        # Step 3: Format the context string
        context_str = self.retriever.format_docs_for_prompt(retrieved_docs)

        # Step 4: Resolve Groq LLM instance
        try:
            llm = get_groq_llm(api_key=api_key, model_name=model_name)
        except ValueError as val_err:
            # If no API key is provided, return helpful notice with retrieved context
            # so the application remains demonstrable and shows the retrieval output
            return {
                "question": question,
                "answer": (
                    f"[Groq API Key Notice]: {str(val_err)}\n\n"
                    f"Retrieved Policy Context Preview (Top matching passage from Qdrant):\n"
                    f"> {retrieved_docs[0].page_content[:300]}...\n\n"
                    f"Please add your GROQ_API_KEY in the .env file or in the frontend sidebar to generate live Groq LLM responses."
                ),
                "sources": sources,
                "context_used": context_str
            }

        # Step 5: LangChain LCEL Chain Execution (Prompt -> Groq LLM -> Output Parser)
        chain = self.prompt | llm | self.output_parser
        raw_answer = chain.invoke({
            "context": context_str,
            "question": question
        })

        return {
            "question": question,
            "answer": raw_answer.strip(),
            "sources": sources,
            "context_used": context_str
        }


if __name__ == "__main__":
    print("Testing PolicyQAEngine with Qdrant and Groq...")
    engine = PolicyQAEngine()
    
    test_q = "What is the minimum attendance required to appear in examinations?"
    print(f"\nAsking Question: '{test_q}'")
    
    result = engine.answer_question(test_q)
    print("\n--- RAG Response ---")
    print("Answer:")
    print(result["answer"])
    print("\nSources:")
    for src in result["sources"]:
        print(f" - Document: {src['source']} | Page: {src['page']} | Chunk: {src['chunk_id']}")
