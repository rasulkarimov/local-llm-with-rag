import chromadb
import logging
import sys

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global query_engine
query_engine = None

def init_llm():
  Settings.llm = Ollama(model="llama3.1", request_timeout=400.0)
  # embeddings model
  Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def init_index(embed_model):

  # chunking
  reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
  documents = reader.load_data()
  logging.info("index creating with `%d` documents", len(documents))

  # Vector databases
  # EphemeralClient does not store any data on disk and Creates an in-memory instance of Chroma
  chroma_client = chromadb.EphemeralClient()
  chroma_collection = chroma_client.create_collection("iollama")

  vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
  storage_context = StorageContext.from_defaults(vector_store=vector_store)

  index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

  return index

def init_query_engine(index):
  global query_engine

  # custome prompt template
  template = (
      "Imagine you are an advanced AI expert in document analysis, with access to all current and relevant documents, "
      "case studies, and expert analyses. Your goal is to provide insightful, accurate, and concise answers to questions.\n\n"
      "Here is some context related to the query:\n"
      "-----------------------------------------\n"
      "{context_str}\n"
      "-----------------------------------------\n"
      "Considering the above information, please respond to the following inquiry with detailes, "
      "precedents, or principles where appropriate:\n\n"
      "Question: {query_str}\n\n"
      "Answer succinctly, starting with the phrase 'According to available information,' and ensure your response is understandable to someone without expert background."
  )
  qa_template = PromptTemplate(template)

  # Query engine
  # build query engine with custom template
  # text_qa_template specifies custom template
  # similarity_top_k configure the retriever to return the top 3 most similar documents,
  # the default value of similarity_top_k is 2
  query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)

  return query_engine

def chat(input_question, user):
  global query_engine

  response = query_engine.query(input_question)
  logging.info("got response from llm - %s", response)

  return response.response