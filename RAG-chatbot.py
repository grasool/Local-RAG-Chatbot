#!/usr/bin/env python

# Conversational agent with RAG
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import config_util
config = config_util.setup(log=False)

def main():

    # Point to the local server
    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])

    embedding_function=HuggingFaceEmbeddings(model_name=config["transformer_type"])
    vector_db = Chroma(persist_directory=config["chromadb_dir"], embedding_function=embedding_function)

    system_prompt = config["system_prompt"]
    intro_prompt = config["intro_prompt"]
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": intro_prompt}
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
        if config["see_log_history"]:
           import json
           gray_color = "\033[90m"
           reset_color = "\033[0m"
           print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
           print(json.dumps(history, indent=2))
           print(f"\n{'-'*55}\n{reset_color}")

        print()
        next_input = input("> ")
        search_results = vector_db.similarity_search(next_input, k=2)
        some_context = ""
        for result in search_results:
            some_context += result.page_content + "\n\n"
        history.append({"role": "user", "content": some_context + next_input})

if __name__ == "__main__":
    main()
