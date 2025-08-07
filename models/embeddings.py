import os
import uuid
from dotenv import load_dotenv
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchField
)
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
azure_search_key = os.getenv("AZURE_SEARCH_KEY")
index_name = "index_field_1"

# Azure clients
credential = AzureKeyCredential(azure_search_key)
index_client = SearchIndexClient(endpoint=azure_search_endpoint, credential=credential)
search_client = SearchClient(endpoint=azure_search_endpoint, index_name=index_name, credential=credential)

import re

def clean_document_key(key):
    return re.sub(r'[^a-zA-Z0-9_\-=]', '_', key)

def create_index_if_not_exists():
    try:
        index_client.get_index(name=index_name)
        print("Index already exists.")
    except:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.lucene"),
            SimpleField(name="source", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=3072,  # Dimensions of text-embedding-3-large
                vector_search_profile_name="default-profile"
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="default-hnsw")
            ],
            profiles=[
                VectorSearchProfile(name="default-profile", algorithm_configuration_name="default-hnsw")
            ]
        )

        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
        index_client.create_index(index)
        print("Index with vector field created.")

# Chunk PDFs using LangChain
def extract_chunks_with_langchain(file_path, chunk_size=500, chunk_overlap=50):
    loader = PyPDFLoader(file_path)
    documents = loader.load()  # returns LangChain Document objects
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks

# Upload chunks to Azure Search
def upload_chunks_to_search():
    data_dir = "data"
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            print(f"Processing {filename}")
            chunks = extract_chunks_with_langchain(file_path)
            docs = []

            for i, chunk in enumerate(chunks):
                raw_id = f"{filename}_{i}_{str(uuid.uuid4())}"
                doc = {
                    "id": clean_document_key(raw_id),
                    "content": chunk.page_content,
                    "source": filename
                }
                docs.append(doc)
            if docs:
                result = search_client.upload_documents(documents=docs)
                print(f"Uploaded {len(result)} chunks from {filename}")

# Run
create_index_if_not_exists()
upload_chunks_to_search()
