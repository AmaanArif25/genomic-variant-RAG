import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key-here")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = "variants-index"

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Retrieval Configuration
TOP_K = 3
SIMILARITY_THRESHOLD = 0.5

# Generation Configuration
LLM_MODEL = "gpt2" 
MAX_LENGTH = 512
TEMPERATURE = 0.3

# Data paths
DATA_PATH = "data/variants.json"

# Evaluation Configuration
EVAL_METRICS = ["accuracy", "relevance", "citation_quality"]

# Ethics and Safety
DISCLAIMER = """
⚠️ IMPORTANT MEDICAL DISCLAIMER:
This system is for research and educational purposes only. 
All recommendations must be validated by qualified healthcare professionals.
Do not use this for clinical decision-making without expert consultation.
"""