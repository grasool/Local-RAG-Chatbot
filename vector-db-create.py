#!/usr/bin/env python

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import glob
import logging
import config_util
config = config_util.setup()

#vector-db-create.py
# create a vector database from a pdf file

# Directory path
dir_path = config['pdf_dir']
# List all PDF files in the specified directory
pdf_files = glob.glob(f"{dir_path}/*.pdf")
if len(pdf_files) < 1:
   logging.error(f"No PDF files found in {dir_path}") 
   exit()

docs = []
for file in pdf_files:
    docs.extend(PyPDFLoader(file).load())

#split text to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"])
docs = text_splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings(model_name=config["transformer_type"], model_kwargs={'device': 'cpu'})

vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory=config["chromadb_dir"])
logging.info(f"Collection count: {vectorstore._collection.count()}")