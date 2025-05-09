import json
import asyncio
import time
import streamlit as st
import websockets
import requests
from websockets.exceptions import ConnectionClosedError

st.set_page_config(layout="wide")

# Define the FastAPI server endpoints
API_URL = "http://127.0.0.1:8000"
WEBSOCKET_URL = "ws://127.0.0.1:8000/chat"

# Function to send query using REST API for synchronous prediction
def predict_query(query):
    response = requests.get(f"{API_URL}/predict", params={"query": query})
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Prediction failed"}

# Asynchronous function to connect to WebSocket and handle streaming response
async def websocket_query(query):
    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            await websocket.send(query)
            response = []
            while True:
                try:
                    message = await websocket.recv()
                    # message = json.loads(message)
                    st.write(f"{type(message)} - {message}")
                    response.append(message)                    
                    # Continue receiving until timeout or "close" signal
                    if "Message received:" in message:
                        break
                except ConnectionClosedError:
                    st.warning("Connection closed unexpectedly. Reconnecting...")
                    await asyncio.sleep(1)  # Short delay before reconnecting
                    continue
        return "\n".join(response)
    
    except Exception as e:
        st.error(f"Connection failed: {e}")

async def connect_to_websocket():
    uri = "ws://localhost:8000/ws/random-number"
    async with websockets.connect(uri) as websocket:
        while True:
            try:
                # Receive message from the WebSocket
                message = await websocket.recv()
                # Display the message in Streamlit
                
                placeholder.markdown(f"### {message}")
            except websockets.ConnectionClosed:
                st.write("Connection closed.")
                break


async def connect_to_multiple_ws():
    uri1 = "ws://127.0.0.1:8000/chat"
    uri2 = "ws://localhost:8000/ws/random-number"
    async with websockets.connect(uri1) as websocket1, \
               websockets.connect(uri2) as websocket2:
        
        col31, col32 = st.columns(2)

        with col31:
            st.markdown("### Chat")
            query = st.text_input("Enter your question:", "", key="wefewe")
            if query:
                await websocket1.send(query)
                response = []
                while True:
                    try:
                        message = await websocket1.recv()
                        # message = json.loads(message)
                        st.write(f"{type(message)} - {message}")
                        response.append(message)                    
                        # Continue receiving until timeout or "close" signal
                        if "Message received:" in message:
                            break
                    except ConnectionClosedError:
                        st.warning("Connection closed unexpectedly. Reconnecting...")
                        await asyncio.sleep(1)  # Short delay before reconnecting
                        continue
        with col32:
            st.markdown("### Monitoring")
            placeholder = st.empty()
            while True:
                try:
                    # Receive message from the WebSocket
                    message = await websocket2.recv()
                    # Display the message in Streamlit
                    placeholder.markdown(f"### {message}")
                except websockets.ConnectionClosed:
                    st.write("Connection closed.")
                    break
        return message

        
        

    


if "num_check" not in st.session_state: st.session_state.num_check=False
if "multiple_ws" not in st.session_state: st.session_state.multiple_ws=False


if __name__ == "__main__":
    # Main Streamlit app interface
    st.title("FastAPI WebSocket Test with LLM")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Send a query to the LLM through WebSocket or REST API.")

        # Input query
        query = st.text_input("Enter your question:", "")

        # WebSocket-based query
        if st.button("Send via WebSocket"):
            if query:
                st.write("Connecting via WebSocket...")
                # Run the async function in Streamlit using asyncio.run
                try:
                    result = asyncio.run(websocket_query(query)) 
                    st.write(result)
                except Exception as e:
                    st.warning(e)
            else:
                st.warning("Please enter a query to send.")

        # REST API-based query
        if st.button("Send via REST API"):
            if query:
                st.write("Sending request to REST API...")
                result = predict_query(query)
                st.write("Response:", result)
            else:
                st.warning("Please enter a query to send.")

    with col2:
        # Button to start WebSocket connection
        st.session_state.num_check = st.checkbox("Start WebSocket Connection- Number Check")
        placeholder = st.empty()

        if st.session_state.num_check:
            # Start the WebSocket connection asynchronously
            asyncio.run(connect_to_websocket())

    st.markdown("---")

    st.session_state.multiple_ws = st.checkbox("Start Multiple Websocket")
    
    if st.session_state.multiple_ws:
        asyncio.run(connect_to_multiple_ws())



        
        

