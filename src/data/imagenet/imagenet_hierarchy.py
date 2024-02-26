# See the README in ILSVRC2012_devkit_t12 for more information on the data
# %%
from __future__ import annotations
import scipy.io
from dataclasses import dataclass, field
from typing import List
import numpy as np

META_PATH = '/shared/nas2/blume5/fa23/ecole/data/imagenet/devkit_t12/data/meta.mat'
# %%
@dataclass
class ImageNetConcept:
    imagenet_id: int
    wordnet_id: str
    words: List[str]
    definitions: List[str]
    num_children: int
    children: List[int]
    wordnet_height: int
    num_train_images: int

    child_concepts: List[ImageNetConcept] = field(default_factory=list)
    parent_concepts: List[ImageNetConcept] = field(default_factory=list)

    @staticmethod
    def fromarray(arr: np.ndarray):
        return ImageNetConcept(
            imagenet_id=arr[0][0][0],
            wordnet_id=arr[1][0],
            words=list(arr[2]),
            definitions=list(arr[3]),
            num_children=arr[4][0][0],
            children=list(arr[5][0]),
            wordnet_height=arr[6][0][0],
            num_train_images=arr[7][0][0]
        )

    def __repr__(self):
        # Prevent infinite recursion into child_concepts and parent_concepts
        return (
            f'ImageNetConcept(imagenet_id={self.imagenet_id},'
            + f' wordnet_id={self.wordnet_id},'
            + f' words={self.words},'
            + f' definitions={self.definitions},'
            + f' num_children={self.num_children},'
            + f' children={self.children},'
            + f' wordnet_height={self.wordnet_height},'
            + f' num_train_images={self.num_train_images},'
            + f' child_concepts=[{[f"{c.imagenet_id}: {c.words[0]}" for c in self.child_concepts]}],'
            + f' parent_concepts=[{[f"{p.imagenet_id}: {p.words[0]}" for p in self.parent_concepts]}],'
            + ')'
        )

# NOTE Hierarchy is a DAG but NOT a tree, as there are multiple paths to same node
def dfs(root: ImageNetConcept, concepts: List[ImageNetConcept], is_reachable: List[bool] = None):
    '''
    Populates child_concepts and parent_concepts of each concept.
    Populates is_reachable array with reachability from root node, if provided.

    Assumes that ImageNetConcept.imagenet_id is 1-indexed, whereas is_reachable is 0-indexed.

    Args:
        root (ImageNetConcept): Concept to start DFS from. This should be the root node (in-degree == 0).
        is_reachable (List[bool], optional): Optional reachability array to populate
    '''
    def clear_pointers(root: ImageNetConcept):
        for child in root.child_concepts:
            clear_pointers(child)

        root.child_concepts.clear()
        root.parent_concepts.clear()

    def helper(root: ImageNetConcept, visited: List[bool]):
        if visited[root.imagenet_id - 1]:
            return

        if is_reachable is not None:
            is_reachable[root.imagenet_id - 1] = True

        visited[root.imagenet_id - 1] = True

        for child in root.children:
            child_concept = concepts[child - 1]

            root.child_concepts.append(child_concept)
            child_concept.parent_concepts.append(root)

            helper(child_concept, visited)

    clear_pointers(root) # Clear child and parent pointers in case they were populated previously
    helper(root, [False] * len(concepts))

def load_concepts():
    data = scipy.io.loadmat(META_PATH)
    synsets = data['synsets'].squeeze()

    return [ImageNetConcept.fromarray(synset) for synset in synsets]

def get_in_degrees(concepts: List[ImageNetConcept]) -> List[int]:
    in_degrees = np.zeros(len(concepts)) # In-degrees of each concept; imagenet_ids are 1-indexed
    for concept in concepts:
        for child in concept.children:
            in_degrees[child - 1] += 1

    return in_degrees

def get_root_nodes(concepts: List[ImageNetConcept]) -> List[ImageNetConcept]:
    in_degrees = get_in_degrees(concepts)
    root_node_inds = [i for i, degree in enumerate(in_degrees) if degree == 0]

    return [concepts[i] for i in root_node_inds]

def get_hierarchy(concepts: list[ImageNetConcept] = None, return_concepts=False) -> ImageNetConcept:
    concepts = load_concepts() if concepts is None else concepts
    root_nodes = get_root_nodes(concepts)

    assert len(root_nodes) == 1
    root_node = root_nodes[0]

    dfs(root_node, concepts) # Populate parent, children pointers

    if return_concepts:
        return root_node, concepts

    return root_node

# %%
if __name__ == '__main__':
    concepts = load_concepts()

    # %% Validate reachability from root node
    root_node = get_hierarchy()

    is_reachable = np.zeros(len(concepts), dtype=bool)
    dfs(root_node, concepts, is_reachable)

    print('All nodes reachable from root node:', np.all(is_reachable))

    # %% Compute in-degree distribution
    from collections import Counter

    in_degrees = get_in_degrees(concepts)
    counts = Counter(in_degrees)
    counts

    # %% View subgraph
    import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout

    # Visualize a BFS tree
    n_levels = 5
    G = nx.DiGraph()

    def get_bfs_graph(root: ImageNetConcept, n_levels: int, G: nx.DiGraph = None):
        all_concepts= []
        curr_level = []
        next_level = []

        curr_level.append(root)
        for i in range(n_levels - 1):
            all_concepts.extend(curr_level)

            for concept in curr_level:
                for child in concept.child_concepts:
                    next_level.append(child)

                    if G is not None:
                        G.add_edge(concept.words[0], child.words[0])

            curr_level = next_level
            next_level = []

        all_concepts.extend(curr_level)

        if G is not None:
            return all_concepts, G

        return all_concepts

    def vis_graph(G: nx.DiGraph):
        subgraph_nodes = list(G.nodes())
        subgraph = G.subgraph(subgraph_nodes)

        # pos = nx.spring_layout(subgraph)
        pos = graphviz_layout(subgraph, prog='dot') # This needs the input to be a tree

        plt.figure(figsize=(30, 10))
        nx.draw(subgraph, pos, with_labels=True, font_size=8, node_size=500, node_color="skyblue", font_color="black")
        plt.show()

    subgraph_concepts, G = get_bfs_graph(root_node, n_levels, G=nx.DiGraph())
    vis_graph(G)

    # %% View subgraph of "whole, unit"
    whole_unit = [concept for concept in concepts if 'whole, unit' in concept.words[0]][0]
    subgraph_concepts, G = get_bfs_graph(whole_unit, n_levels=3, G=nx.DiGraph())
    vis_graph(G)

# %%
