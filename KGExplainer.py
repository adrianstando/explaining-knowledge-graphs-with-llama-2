from llama_cpp import Llama
import os
import re
from rdflib import Graph
from copy import deepcopy

model_path_middle = "./llama-2-7b-chat.Q4_K_M.gguf"
DEFAULT_ONTOLOGIES = ["/data/sosa.ttl", "/data/rdfs.ttl", "/data/rdf.ttl", "/data/owl.ttl"]
DEFAULT_IRIS = [
    "http://www.w3.org/ns/sosa",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2002/07/owl#",
]
get_iri = lambda x: "/".join(x.split("/")[:-1])


def get_graph_data(graph):
    """Serialize the rdflib Graph into a turtle format with cleaned up data.

    Args:
        graph (rdflib.Graph): Graph to format.

    Returns:
        str: the data of the Graph, properly formatted.
    """
    graph_data = graph.serialize()
    graph_data = re.sub("<[^ ]*\/([^ \/]*)>", "\g<1>", graph_data)
    graph_data = "\n".join(
        [x for x in graph_data.split("\n") if x[:7] != "@prefix" and x != ""]
    )
    return graph_data


class GraphExplainer:
    def __init__(self, model_path=model_path_middle, n_ctx=4096, n_batch=2048):
        """Create the GraphExplainer object, with initialized LLM.

        Args:
            model_path (str, optional): Path to the LLM model weights. Defaults to model_path_middle.
            n_ctx (int, optional): n_ctx parameter forwarded to the model. For further reading refer to the llama_cpp documentation. Defaults to 4096.
            n_batch (int, optional): n_batch parameter forwarded to the model. For further reading refer to the llama_cpp documentation. Defaults to 2048.
        """
        self.model_path = model_path
        self.n_threads = os.cpu_count()
        self.n_gpu = -1
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.llm = self._load_model(n_ctx=self.n_ctx, n_batch=self.n_batch)

    def _load_model(self, *args, **kwargs):
        """Load the LLM object.

        Returns:
            LLama: LLM object.
        """
        return Llama(
            model_path=self.model_path,
            n_gpu_layers=self.n_gpu,
            n_threads=self.n_threads,
            *args,
            **kwargs
        )

    def explain_graph(self, system_prompt, user_prompt, *args, **kwargs):
        """Query the LLM to create a explanation.

        Args:
            system_prompt (str): System prompt for the LLM. Usually a description of how the model should behave.
            user_prompt (str): User prompt for the LLM. Usually a question or a task for the model to perform.

        Returns:
            str: The models answer.
        """
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            *args,
            **kwargs
        )
        return out["choices"][0]["message"]["content"]

    def set_ontologies_lookup(
        self,
        ontologies_paths,
        ontologies_iris,
        default_ontologies=DEFAULT_ONTOLOGIES,
        default_iris=DEFAULT_IRIS,
    ):
        """_summary_

        Args:
            ontologies_paths (list, str): List of paths to files, for the ontologies we want to add to the model explanations.
            ontologies_iris (list, str): List of iris for the ontologies we want to add to the model explanations.
            default_ontologies (list, optional): Default list of ontologies paths. Defaults to DEFAULT_ONTOLOGIES.
            default_iris (list, optional): Default list of ontologies iris. Defaults to DEFAULT_IRIS.
        """
        if type(ontologies_paths) == str:
            ontologies_paths = [ontologies_paths]
        if type(ontologies_iris) == str:
            ontologies_iris = [ontologies_iris]
        all_ontologies = deepcopy(default_ontologies)
        all_ontologies.extend(ontologies_paths)
        all_iris = deepcopy(default_iris)
        all_iris.extend(ontologies_iris)

        self.ontologies_dict = {}
        for ont, iri in zip(all_ontologies, all_iris):
            ont_graph = Graph().parse(ont, format="ttl")
            self.ontologies_dict[iri] = ont_graph
        return

    def process_graph(self, file_path):
        """Process the graph to return the pure data from it, and the required data extraced from ontologies.

        Args:
            file_path (str): Path to the graph file in 'ttl' format.

        Returns:
            str: The processed input data, cleaned up.
            str: The processed required data from ontologies, cleaned up.
        """
        processed_graph = Graph().parse(file_path, format="ttl")
        required_ontologies = []
        for s, p, o in processed_graph.triples((None, None, None)):
            for element in [s, p, o]:
                if "http" in str(element):
                    required_ontologies.append(element)
        required_ontologies = list(set(required_ontologies))

        ontologies_graph = Graph()
        for data_required in required_ontologies:
            temp_iri = get_iri(data_required)
            if temp_iri in self.ontologies_dict.keys():
                relevant_ontology = self.ontologies_dict[temp_iri]
                for s, p, o in relevant_ontology.triples((data_required, None, None)):
                    ontologies_graph.add((s, p, o))
        ontology_text = get_graph_data(ontologies_graph)
        input_text = get_graph_data(processed_graph)
        return input_text, ontology_text
