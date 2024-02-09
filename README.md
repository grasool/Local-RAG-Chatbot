# Local Chatbot Using LM Studio, Chroma DB, and LangChain

The idea for this work stemmed from the requirements related to data privacy in hospital settings. This may be true for many use cases. The outcome of this work is a private conversational agent that runs on the local machine. Here are some details:

## LLMs
It can use any LLM from LM Studio. Just change the LLM from LM studio GUI and rerun the server. I am currently using Mistral 7B.

## Vector DB
For the Retrieval Augmented Generation (RAG) component, I am using Chroma DB. You may any others.

## Embeddings
For the embeddings, I am using sentence-transformers through langchain/HugginFace. These can be switched easily.

# Installation and Use

1.  Make conda environment and activate it.
2.  Install requirements.
3.  Install LM Studio and download some LLMs.
4.  Run the LM Studio server.
5.  Create a vector database from PDF files.
6.  Run the conversational agent.
   
