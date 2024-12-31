from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# Define State
class GradeState(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Define Agent
class GenerationAgent:

    def retrieval_agent(fetch_k:int, k:int, db_path:str):
        embed_model = OllamaEmbeddings(base_url="http://localhost:11434", model="bge-m3:latest")
        vectorstore = Chroma(collection_name="collection_01", persist_directory=db_path, embedding_function=embed_model)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': k, "fetch_k":fetch_k})
        return retriever
    
    def retrieval_grader_agent(state):
        llm = ChatGroq(temperature=0, model_name= "llama-3.3-70b-versatile")
        structured_llm_grader = llm.with_structured_output(state)
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        retrieval_grader = grade_prompt | structured_llm_grader
        return retrieval_grader
    
    def generate_agent(question:str, context:list):
        prompt = ChatPromptTemplate.from_messages([
        ("human", 
        """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:"""),
        ])
        llm = ChatGroq(temperature=0, model_name= "llama-3.3-70b-versatile")

        rag_chain = prompt | llm | StrOutputParser()

        # Post-processing
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        total_docs = format_docs(context)

        gen_agent = rag_chain.invoke({"context": total_docs, "question": question})
        return gen_agent





if __name__ == "__main__":
    # How to Use
    question = """
    according to lr rule, explain about ships with installed process plant for chemicals
    """
    retriever = GenerationAgent.retrieval_agent(fetch_k=10, k=3, db_path="../db/chroma_db_02")
    docs = retriever.invoke(question)
    print(docs)
    print("-"*70)


    retrieval_grader = GenerationAgent.retrieval_grader_agent(state=GradeState)
    yes_result = [doc for doc in docs if retrieval_grader.invoke({"question": question, "document": doc.page_content}).binary_score == 'yes']
    print(yes_result)
    no_result = [doc for doc in docs if retrieval_grader.invoke({"question": question, "document": doc.page_content}).binary_score == 'no']
    print(no_result)
    print("-"*70)


    result = GenerationAgent.generate_agent(question=question, context=yes_result)
    print(result)




