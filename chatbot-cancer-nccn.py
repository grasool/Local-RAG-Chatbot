#chatbot-cancer-nccn.py

# One question only version of the QA system

# Following the example from: https://github.com/InsightEdge01/AutogenLangchainPDFchat/blob/main/app.py

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import openai
import autogen
from langchain_openai import ChatOpenAI


openai.api_type = "open_ai"
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "NULL"

embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)

query = "What are PCR and FISH for gliom patients?"

print("\n\nSearching for similar documents to:", query)

search_results = vector_db.similarity_search(query, k=5)
# for result in search_results:
#     print(result.page_content, "\n---\n")

llm = ChatOpenAI(temperature=0.0, base_url="http://localhost:1234/v1", api_key="not-needed")


# # Chat with an intelligent assistant in your terminal
# from openai import OpenAI

# # Point to the local server
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

history = [
    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
    {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
]

while True:
    completion = llm.chat.completions.create(
        model="local-model", # this field is currently unused
        messages=history,
        temperature=0.7,
        stream=True,
    )

    new_message = {"role": "assistant", "content": ""}
    
    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] += chunk.choices[0].delta.content

    history.append(new_message)
    
    # Uncomment to see chat history
    # import json
    # gray_color = "\033[90m"
    # reset_color = "\033[0m"
    # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
    # print(json.dumps(history, indent=2))
    # print(f"\n{'-'*55}\n{reset_color}")

    print()
    next_input = input("> ")
    history.append({"role": "user", "content": next_input})



# # Build prompt
# from langchain.prompts import PromptTemplate
# template = """Use the following pieces of context to answer the question at the end. \
#     If you don't know the answer, just say that you don't know, don't try to make up an answer. \
#     Use three sentences maximum. Keep the answer as concise as possible. Always say \
#     "thanks for asking!" at the end of the answer.  {context} \
#     Question: {question}
#     Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)


# print(type(QA_CHAIN_PROMPT))





# prompt_template = ChatPromptTemplate.from_template(template_string)


# customer_messages = prompt_template.format_messages(
#                     style=new_style,
#                     text=customer_email)






# # Run chain
# from langchain.chains import RetrievalQA
# question = "What are PCR and FISH?"
# qa_chain = RetrievalQA.from_chain_type(llm,
#                                         retriever=vector_db.as_retriever(),
#                                         return_source_documents=True,
#                                         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

# print("\n\nRunning AI\n\n")

# result = qa_chain.invoke({"query": question})
# print(result["result"])
