import chromadb
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


# Define State
class RouteState(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["similarity_search", "vectorstore", "web_search", "database"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore or a similarity or a database.",
    )

# Define Agent
class RouteAgent:

    def vectordb_targets(db_path:str):
        client = chromadb.PersistentClient(path=db_path)
        for collection in client.list_collections():
            data = collection.get(include=['metadatas'])
        lv1 = list(set([d['First Division'] for d in data["metadatas"]]))
        lv2 = list(set([d['Second Division'] for d in data["metadatas"]]))
        rag_target = lv1 + lv2
        rag_target.insert(0, "vectorstore")
        rag_target.insert(0, "vectordb")
        docs = ", ".join(rag_target)
        return docs

    def routing_agent(state, db_index):

        system = f"""You are an expert at routing a user question to a vectorstore, web search or database.
        The vectorstore contains documents related to {db_index}, Use the vectorstore for questions on these topics. 
        The question contains words of similarity or sim search, Use similarity_search for the question.
        The question contains words related to database, Use the database for the question. 
        Otherwise, use web-search."""

        route_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}"),])
        llm = ChatGroq(temperature=0, model_name= "llama-3.3-70b-versatile")    
        structured_llm_router = llm.with_structured_output(state)
        question_router = route_prompt | structured_llm_router
        return question_router



if __name__ == "__main__":
    
    # How to Use
    db_path = "../db/chroma_db_02"
    db_index = RouteAgent.vectordb_targets(db_path=db_path)
    print(db_index)
    print("-"*70)

    question_router = RouteAgent.routing_agent(state=RouteState, db_index=db_index)
    query = 'with reference to "lr" rule, explain the measurement procedure of "noise"'
    print(question_router.invoke({"question": query}))

