from tavily import TavilyClient
from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI

load_dotenv()
client = TavilyClient(api_key="TAVILY_KEY")  # Replace with your actual Tavily API key


llm = AzureChatOpenAI(
    model="gpt-4o",
    temperature=0,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version="2025-01-01-preview"
)

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2025-01-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def get_embeddings_vector(text):

    embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("AZURE_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_EMBEDDING_API_KEY"),
    openai_api_version="2023-05-15",
    )
    single_vector = embeddings.embed_query(text)
    return single_vector
