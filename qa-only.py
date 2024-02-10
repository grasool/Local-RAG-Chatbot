# One question-only version of the QA system
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI

from langchain_openai import ChatOpenAI


embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)

query = "What are PCR and FISH for gliom patients?"

print("\n\nSearching for similar documents to:", query)

search_results = vector_db.similarity_search(query, k=2)

# make a string of the search results
search_results_string = ""
for result in search_results:
    search_results_string += result.page_content + "\n\n"

# print the string
print(search_results_string)

llm = ChatOpenAI(temperature=0.0, base_url="http://localhost:1234/v1", api_key="not-needed")

# Build prompt
from langchain.prompts import PromptTemplate
template = """Use the following pieces of context to answer the question at the end. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer. \
    Use three sentences maximum. Keep the answer as concise as possible. Always say \
    "thanks for asking!" at the end of the answer.  {context} \
    Question: {question}
    Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
from langchain.chains import RetrievalQA
question = "What are PCR and FISH?"
qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=vector_db.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

print("\n\nRunning AI\n\n")

result = qa_chain.invoke({"query": question})
print(result["result"])
