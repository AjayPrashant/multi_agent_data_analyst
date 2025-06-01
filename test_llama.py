# test_llama.py

from llama_cpp import Llama

llm = Llama(model_path="/Users/ajayprashantmuralidharan/Library/Application Support/nomic.ai/GPT4All/mistral-7b-instruct-v0.2.Q4_0.gguf")
output = llm("Explain data preprocessing in machine learning.", max_tokens=128)
print(output["choices"][0]["text"])
