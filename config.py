import os
import logging
from chromadb.config import Settings


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

OPENAI_MODEL="gpt-4o-mini"

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

# logging config
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
