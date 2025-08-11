# visualisation.py
import matplotlib.pyplot as plt
import networkx as nx
from environnement.marelle_layout import marelle_layout

def plot_marelle(G, moves=None, ax=None, removable_pawns=None):
    pos = nx.get_node_attributes(G, "pos")
    colors = []
    
    for n in G.nodes():
        s = G.nodes[n].get("state", 0)
        if s == 1:
            # Colorer en orange si le pion rouge peut être supprimé
            if removable_pawns and n in removable_pawns:
                colors.append("orange")
            else:
                colors.append("red")
        elif s == -1:
            # Colorer en orange si le pion bleu peut être supprimé
            if removable_pawns and n in removable_pawns:
                colors.append("orange")
            else:
                colors.append("blue")
        else:
            colors.append("white")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Améliorer le placement visuel
    nx.draw(G, pos, 
            with_labels=False,  # On ne dessine pas les labels par défaut
            node_color=colors, 
            node_size=800, 
            edgecolors="black", 
            linewidths=2,
            edge_color='gray',
            width=3,
            alpha=0.7,
            ax=ax)

    # Ajouter les labels personnalisés
    for n in G.nodes():
        x, y = pos[n]
        s = G.nodes[n].get("state", 0)
        
        if s != 0:  # Si un jeton est placé
            # Trouver le numéro du coup
            move_number = None
            for idx, (node, player) in enumerate(moves or []):
                if node == n:
                    move_number = idx + 1
                    break
            
            if move_number is not None:
                # Afficher le numéro du coup en blanc
                ax.text(x, y, str(move_number), 
                       color="white", 
                       ha="center", 
                       va="center", 
                       fontsize=12, 
                       fontweight='bold')
        else:
            # Afficher le numéro du nœud si vide
            ax.text(x, y, str(n), 
                   color="black", 
                   ha="center", 
                   va="center", 
                   fontsize=10, 
                   fontweight='normal')

    ax.axis("equal")
    ax.set_axis_off()
    return ax
