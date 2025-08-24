from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re


class RAGSystem:
    def __init__(self):
        # Sample documents
        self.documents = [
            "Retrieval-Augmented Generation (RAG) combines retrieval and generation for better AI responses.",
            "FAISS is a library for efficient similarity search and clustering of dense vectors.",
            "Hugging Face Transformers provide pre-trained models for NLP tasks.",
            "Flask is a lightweight web framework for Python.",
            "xAI's API can be used for advanced generative AI: https://x.ai/api.",
            "RAG improves AI accuracy by grounding responses in retrieved documents, reducing hallucinations.",
            "LangChain is a framework for building applications with LLMs, including RAG pipelines.",
            "RAG is a design pattern where a retrieval step fetches information from an external knowledge base before the generative model creates a response",
            "RAG does not involve food, facial recognition, or modeling neural networks for faces; it focuses on text-based retrieval and generation."]

        # Improved embeddings model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Create vector store with cosine similarity
        self.vector_store = FAISS.from_texts(
            texts=self.documents,
            embedding=self.embeddings,
            metadatas=[{} for _ in self.documents],
            normalize_L2=True  # Normalize embeddings for cosine similarity
        )

        # Use gpt2 for speed and compatibility
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer,
                                       device=0 if torch.cuda.is_available() else -1,
                                       temperature=0.5,  # Low for coherence
                                       max_new_tokens=150)  # Short for focus
        self.llm = HuggingFacePipeline(pipeline=self.text_generator)

        # Simplified prompt to avoid fragments
        prompt_template = """

        Context: {context}

        Question: {question}

        Answer: """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        # RAG chain with tuned retriever
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": 3, "score_threshold": 0.10}  # higher threshold
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True  # For debugging
        )

    def query(self, question):
        try:
            # Run query and get result with source documents
            result = self.qa_chain({"query": question})
            source_docs = [(doc.page_content, score) for doc, score in zip(result["source_documents"],
                                                                           result.get("source_scores", [None] * len(
                                                                               result["source_documents"])))]

            # Debug: Log retrieved documents and scores
            print("Retrieved docs and scores:", source_docs)

            # Concatenate source documents into context
            context = "\n".join([doc for doc, _ in source_docs])

            # Debug: Log context sent to LLM
            print("Context sent to LLM:", context)

            # Simplified prompt
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer: "

            # Generate response directly with LLM
            raw_response = self.text_generator(prompt)[0]["generated_text"]

            # Debug: Log raw LLM output
            print("Raw LLM output:", raw_response)

            # Clean response: Remove prompt fragments, whitespace, and partial prompts
            cleaned_answer = re.sub(r"^(Context|Question|Answer):.*?\n|\s*Answer\s*:?\s*", "", raw_response,
                                    flags=re.MULTILINE)
            cleaned_answer = re.sub(r"\s+", " ", cleaned_answer.strip())

            print("Clean LLM output:", cleaned_answer)
            return cleaned_answer
        except Exception as e:
            print("Error:", str(e))
            return "I don't know."
