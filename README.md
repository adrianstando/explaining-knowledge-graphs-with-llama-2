# Explaining knowledge graphs with Llama 2

Describe knowledge graphs with Llama 2. This project attempts to use LLMs to turn Knowledge Graphs into human readable content.

# Installation
To install the proper libraries run the following commands in a Python environment. Tested on Python 3.10.12.
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python==0.2.26
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf?download=true
mv llama-2-7b-chat.Q4_K_M.gguf?download=true ./llama-2-7b-chat.Q4_K_M.gguf
pip install rdflib==7.0.0
```

Take care if trying to use higher version of libraries, as using llama-cpp-python 0.2.28 used to break the LLM model loading functions.

# Usage
Example usage looks as follows. Note that GraphExplainer uses a default set of ontologies, expected to be available in 'data' directory. This can be overriden in the `set_ontologies_lookup` method using the respective parameters `default_ontologies`, `default_iris`.

```python
from KGExplainer import GraphExplainer

explainer = GraphExplainer()
explainer.set_ontologies_lookup(['pilot2.ttl'], ['https://assist-iot.eu/ontologies'])
input_text, ontology_text = explainer.process_graph('cam_0000003596.ttl')

system_prompt = """
You are a master in the field of knowledge graphs, and you explain their contents using layman's terms. 
Use the following ontology to understand the data.
""" + ontology_text

input_prompt = """
Summarize only the data of the following graph, in sentences only.
Focus on the values at the end of each line.
""" + input_text

explainer.explain_graph(system_prompt_1, input_prompt_1)
```
