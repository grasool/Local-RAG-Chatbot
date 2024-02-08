#embedding-FAQ.py

import torch
from datasets import load_dataset
import requests

model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()


#faqs_embeddings = load_dataset('grasool/random-embeddings')
faqs_embeddings = load_dataset('ITESM/embedded_faqs_medicare')
dataset_embeddings = torch.from_numpy(faqs_embeddings["train"].to_pandas().to_numpy()).to(torch.float)
print(dataset_embeddings.shape)



question = ["How can Medicare help me?"]
output = query(question)

query_embeddings = torch.FloatTensor(output)
# print(query_embeddings.shape)
# print(query_embeddings)

from sentence_transformers.util import semantic_search

hits = semantic_search(query_embeddings, dataset_embeddings, top_k=5)

texts = ["How do I get a replacement Medicare card?",
        "What is the monthly premium for Medicare Part B?",
        "How do I terminate my Medicare Part B (medical insurance)?",
        "How do I sign up for Medicare?",
        "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        "How do I sign up for Medicare Part B if I already have Part A?",
        "What are Medicare late enrollment penalties?",
        "What is Medicare and who can get it?",
        "How can I get help with my Medicare Part A and Part B premiums?",
        "What are the different parts of Medicare?",
        "Will my Medicare premiums be higher because of my higher income?",
        "What is TRICARE ?",
        "Should I sign up for Medicare Part B if I have Veterans' Benefits?"]

print([texts[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])
