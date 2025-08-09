# marelle_graph.py
import networkx as nx
from marelle_layout import marelle_layout

def marelle_edges():
    # Arêtes le long de chaque carré (cycle) + les 4 colonnes de connexions ext->moy->int (top, right, bottom, left)
    outer = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0)]
    middle = [(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,8)]
    inner = [(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,16)]

    # Connexions radiales (chaque colonne est une chaîne de 3 points)
    # top middle: outer 1 -> middle 9 -> inner 17
    # right middle: outer 3 -> middle 11 -> inner 19
    # bottom middle: outer 5 -> middle 13 -> inner 21
    # left middle: outer 7 -> middle 15 -> inner 23
    connections = [
        (1,9), (9,17),
        (3,11), (11,19),
        (5,13), (13,21),
        (7,15), (15,23)
    ]

    return outer + middle + inner + connections

def create_marelle_graph():
    G = nx.Graph()
    pos = marelle_layout()
    G.add_nodes_from(pos.keys())
    nx.set_node_attributes(G, pos, "pos")
    G.add_edges_from(marelle_edges())
    nx.set_node_attributes(G, 0, "state")  # état initial : 0 = vide
    return G
