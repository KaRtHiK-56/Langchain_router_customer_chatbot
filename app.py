#import necessaary libraries
import os
import pandas as pd
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq 

#loading the environment variables

load_dotenv()

HF = os.environ.get("HUGGING_FACE_API")
GROQ = os.environ.get("GROQ_API")

print("Hello")