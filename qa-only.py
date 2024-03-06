#!/usr/bin/env python

# One question-only version of the QA system
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
import logging
import config_util
config = config_util.setup()
question = config["qa_question"]

if not config["url_valid"]:
   logging.error(f"{config['base_url']} isn't responding. Are your sure LM Studio server is running?")
   exit()

embedding_function=HuggingFaceEmbeddings(model_name=config["transformer_type"])
vector_db = Chroma(persist_directory=config["chromadb_dir"], embedding_function=embedding_function)
logging.info(f"\n\nSearching for similar documents to: {question}")

search_results = vector_db.similarity_search(question, k=2)

# make a string of the search results
search_results_string = ""
for result in search_results:
    search_results_string += result.page_content + "\n\n"

# print the string
#logging.info(f"search results: {search_results_string}")

llm = ChatOpenAI(temperature=0.0, base_url=config["base_url"], api_key=config["api_key"])

# Build prompt
from langchain.prompts import PromptTemplate
template = config["template"]
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=vector_db.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

logging.info("## Running AI ##")

result = qa_chain.invoke({"query": question})
logging.info(result["result"])
