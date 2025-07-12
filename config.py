import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

def get_hf_token():
    if not HUGGINGFACE_TOKEN:
        raise ValueError('Hugging Face token not set in environment!')
    return HUGGINGFACE_TOKEN 