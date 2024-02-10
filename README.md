# Local Chatbot Using LM Studio, Chroma DB, and LangChain

The idea for this work stemmed from the requirements related to data privacy in hospital settings. This may be true for many use cases. The outcome of this work is a private conversational agent that runs on the local machine. Here are some details:

## LLMs
It can use any LLM from LM Studio. Just change the LLM from LM studio GUI and rerun the server. I am currently using Mistral 7B.

## Vector DB
For the Retrieval Augmented Generation (RAG) component, I am using Chroma DB. You may any others.

## Embeddings
For the embeddings, I am using sentence-transformers through langchain/HugginFace. These can be switched easily.

# Installation and Use
1.  Install miniconda. Download from https://docs.anaconda.com/free/miniconda/index.html
2.  Create a conda environment and activate it.
```
conda create --name testing-123 python=3.10
conda activate testing-123
```
3.  Install requirements.
```
pip install -r requirements.txt
```
4.  Install LM Studio. Use the following link: https://lmstudio.ai/
5.  Download some LLMs by searching and selecting from the LM studio search panel. I experimented with (1) llama-2-7b-chat.Q8_0.gguf and (2) mistral-7b-instruct-v0.2.Q6_K.gguf.
6.  Run the LM Studio server. Accept all default options.
7.  Create a vector database from PDF files by running the vector-db-create.py. Please specify the appropriate path for the PDF file.
8.  Run qa-only.py. This will help you take care of any errors that may pop up.
9. Run the conversational agent using the script in RAG-chatbot.py.
   
