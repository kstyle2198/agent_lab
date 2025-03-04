from typing import Any
import asyncio
import bs4
import uvicorn
import random
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.base import AsyncCallbackHandler
from fastapi import WebSocket, FastAPI, WebSocketDisconnect

from dotenv import load_dotenv
load_dotenv()


app = FastAPI()

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

embed_model = OllamaEmbeddings(base_url='http://localhost:11434',
                               model="bge-m3:latest")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embed_model)

retriever = vectorstore.as_retriever()
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


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@app.get("/predict")
def predict(query: str):
    llm = ChatGroq(temperature=0, 
                   model_name= "llama-3.2-11b-text-preview",
                   )

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain.invoke(query)


class LLMCallbackHandler(AsyncCallbackHandler):
    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        await self.websocket.send_json({"message": token})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    callback_manager = AsyncCallbackManager(
        [LLMCallbackHandler(websocket)])

    llm = ChatGroq(temperature=0, 
                   model_name= "llama-3.2-11b-text-preview",
                   streaming=True,
                   callback_manager=callback_manager
                   )

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
            )

    while True:
        query = await websocket.receive_text()
        response  = await rag_chain.ainvoke(query)
        await websocket.send_text(f"Message received: {response}")

@app.websocket("/ws/random-number")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Generate a random integer between 1 and 10
            random_number = random.randint(1, 10)

            # Send the random number to the client
            await websocket.send_text(f"Random number: {random_number}")

            # Check if the number is greater than 5
            if random_number > 5:
                # Send an alert message if the condition is met
                await websocket.send_text(f"Alert: Number({random_number}) is greater than 5!")

            # Wait for 5 seconds before sending the next number
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)