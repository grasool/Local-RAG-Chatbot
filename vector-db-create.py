#vector-db-create.py

#main.py

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

loaders = [PyPDFLoader('./pdfs/brain-gliomas-patient.pdf')]

docs = []
for file in loaders:
    docs.extend(file.load())
#split text to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(docs)
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
#print(len(docs))

vectorstore = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_nccn")
# persist_directory="./chroma_db_nccn": https://python.langchain.com/docs/integrations/vectorstores/chroma


#vectorstore.add_documents(docs)
print(vectorstore._collection.count())

query = "What are PCR and FISH?"

search_results = vectorstore.similarity_search(query, k=2)
for result in search_results:
    print(result.page_content, "\n---\n")

# #question = "What is sterotactic biopsy?"
# #search_results = vectorstore.similarity_search(question,k=5)

# # for result in search_results:
# #     print(result.page_content, "\n---\n")

# llm = ChatOpenAI(temperature=0.0, base_url="http://localhost:1234/v1", api_key="not-needed")


# # Build prompt
# from langchain.prompts import PromptTemplate
# template = """Use the following pieces of context to answer the question at the end. \
#     If you don't know the answer, just say that you don't know, don't try to make up an answer. \
#     Use three sentences maximum. Keep the answer as concise as possible. Always say \
#     "thanks for asking!" at the end of the answer.  {context} \
#     Question: {question}
#     Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# # Run chain
# from langchain.chains import RetrievalQA
# question = "What is sterotactic biopsy?"
# qa_chain = RetrievalQA.from_chain_type(llm,
#                                        retriever=vectorstore.as_retriever(),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


# result = qa_chain.invoke({"query": question})
# print(result["result"])






# # from langchain_openai import ChatOpenAI
# # from langchain.prompts import ChatPromptTemplate

# # chat = ChatOpenAI(temperature=0.0, base_url="http://localhost:1234/v1", api_key="not-needed")


# # template_string = """you are an expert in helping cancer paitnets wiht their questions. Answer the following  \
# # questions ```{text}``` """



# # prompt_template = ChatPromptTemplate.from_template(template_string)


# # question_for_chatbot = prompt_template.format_messages(
# #                     text=question)

# # print(question_for_chatbot)



# customer_response = chat(question_for_chatbot)

# print(customer_response.content)





# qa = ConversationalRetrievalChain.from_llm(
#     OpenAI(temperature=0),
#     vectorstore.as_retriever(),
#     memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# )








# #set config for autogen
# config_list = [
#     {
#          "api_base": "http://localhost:1234/v1",
#         "api_key": "NULL"
#     }
# ]

# #set autogen user agent and assistant agent with function calling
# llm_config={
#     "request_timeout": 600,
#     "seed": 42,
#     "config_list": config_list,
#     "temperature": 0,
#     "functions": [
#         {
#             "name": "chat_docs",
#             "description": "Answer any chat_docs related questions",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "question": {
#                         "type": "string",
#                         "description": "The question to ask in relation to chat_docs",
#                     }
#                 },
#                 "required": ["question"],
#             },
#         }
#     ],
# }

# #the function takes a parameter question,calls the qa chain and answer it by returnin the answer
# # from the chain
# def chat_docs(question):
#     response = qa({"question": question})
#     return response["answer"]


# # create an AssistantAgent instance "assistant"
# assistant = autogen.AssistantAgent(
#     name="assistant",
#     llm_config=llm_config,
# )


# # create a UserProxyAgent instance "user_proxy"
# user_proxy = autogen.UserProxyAgent(
#     name="user_proxy",
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=1,
#     code_execution_config={"work_dir": "docs"},
#     llm_config=llm_config,
#     system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
# Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
#     function_map={"chat_docs":chat_docs}
# )

# # the assistant receives a message from the user, which contains the task description
# user_proxy.initiate_chat(
#     assistant,
#     message="""
# Find the answers to the 3 questions below from the chat_docs.pdf and do not write any code.

# 1.Who is the CEO of walmart?
# 2.What did Doug McMillon write about walmart.
# 3.Write about the Executive Shuffle?

# Start the work now.
# """
# )