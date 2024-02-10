# Conversational agent with RAG
#
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def main():

    # Point to the local server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)


    history = [
        {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
        {"role": "user", "content": "Hello, introduce yourself to someone opening this program for the first time. Be concise."},
    ]

    while True:
        completion = client.chat.completions.create(
            model="local-model",
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
        
        #Uncomment to see chat history
        # import json
        # gray_color = "\033[90m"
        # reset_color = "\033[0m"
        # print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
        # print(json.dumps(history, indent=2))
        # print(f"\n{'-'*55}\n{reset_color}")

        print()
        next_input = input("> ")
        search_results = vector_db.similarity_search(next_input, k=2)
        some_context = ""
        for result in search_results:
            some_context += result.page_content + "\n\n"
        history.append({"role": "user", "content": some_context + next_input})

if __name__ == "__main__":
    main()