from agent_graph import JokeAgent

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
import streamlit as st
import os,io
from dotenv import load_dotenv
import tempfile

from config import (
    OPENAI_MODEL,
    logging,
)

st.set_page_config(page_title="Knock Knock Jokes", page_icon="😀")
st.title("Knock Knock Jokes 😀")

# def llm_selector():
#     models=["gpt-4o-mini","gpt-4o", "gpt-3.5-turbo" ]
#     with st.sidebar:
#         return st.selectbox("Model", models)

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    st.info("Export an OpenAI API Key \"export OPENAI_API_KEY='sk-xxx'\" to continue")
    st.stop()

# Set up the LangChain, passing in Message History
@st.cache_resource
def get_joke_agent():
    return JokeAgent()

joke_agent = get_joke_agent()

client = OpenAI(api_key=openai_api_key)

# Speech to text
def speech_to_text(audio_data):
    transcript = ''
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript

# Text to speech
def text_to_speech(input_text, file_path):
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=input_text,
    ) as response:
        response.stream_to_file(file_path)


# Play audio
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    st.audio(audio_bytes, format="audio/mpeg", autoplay=True)

# Get LLM response
def get_llm_response(input: str) -> str:

    inputs = {
        "messages": [
            ("user", input),
        ]
    }

    response = joke_agent.graph.invoke(inputs)
    return response["messages"][-1].content

audio_bytes = None

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

with st.sidebar:
    audio_bytes = audio_recorder(pause_threshold=1.0, sample_rate=16_000)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    msgs.add_user_message(prompt)
    audio_bytes = None

# If audio bytes are available, transcribe and add to chat history
if audio_bytes is not None:

    # Save the recorded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

        # Write the audio bytes to a file
        with st.spinner("Transcribing..."):
            # STT
            transcript = speech_to_text(temp_audio_path)
            if transcript:
                st.chat_message("human").write(transcript)
                msgs.add_user_message(transcript)
        
        # Clean up
        os.unlink(temp_audio_path)

# If last message is not from the AI, generate a new response
if st.session_state.langchain_messages[-1].type != "ai":

    with st.spinner("Thinking🤔..."):
        response = get_llm_response(st.session_state.langchain_messages[-1].content)

    st.chat_message("ai").write(response)
    msgs.add_ai_message(response)
    with st.spinner("Generating audio response..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = temp_audio.name

            tts_audio = text_to_speech(response, temp_audio_path)
            # auto-play audio
            autoplay_audio(temp_audio_path)
            
            # Clean up
            os.unlink(temp_audio_path)