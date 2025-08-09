# visualisation.py
import matplotlib.pyplot as plt
import networkx as nx
from marelle_layout import marelle_layout

def plot_marelle(G, moves=None, ax=None, removable_pawns=None):
    pos = nx.get_node_attributes(G, "pos")
    colors = []
    
    for n in G.nodes():
        s = G.nodes[n].get("state", 0)
        if s == 1:
            colors.append("red")
        elif s == -1:
            # Colorer en orange si le pion peut être supprimé
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

def play_with_clicks():
    """Fonction pour jouer interactivement avec des clics de souris"""
    from marelle_env import MarelleEnv
    
    env = MarelleEnv()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def on_click(event):
        if event.inaxes != ax:
            return
        
        # Trouver le nœud le plus proche du clic
        pos = nx.get_node_attributes(env.G, "pos")
        min_dist = float('inf')
        closest_node = None
        
        for node, (x, y) in pos.items():
            dist = ((event.xdata - x)**2 + (event.ydata - y)**2)**0.5
            if dist < min_dist and dist < 1.0:  # Seuil de distance
                min_dist = dist
                closest_node = node
        
        if closest_node is None:
            return
            
        # Mode suppression de pion
        if env.waiting_for_removal:
            if env.remove_pawn(closest_node):
                print(f"Pion retiré de la position {closest_node}")
            else:
                print("Position invalide pour la suppression")
        # Mode placement de pion
        elif env.G.nodes[closest_node]["state"] == 0:
            try:
                env.play_move(closest_node)
                print(f"Pion placé sur la position {closest_node}")
            except Exception as e:
                print(f"Erreur: {e}")
        
        # Rafraîchir l'affichage
        ax.clear()
        plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
        
        # Afficher le statut
        if env.waiting_for_removal:
            plt.title(f"⚡ Joueur {env.current_player} a formé un moulin ! Cliquez sur un pion adverse à retirer")
        else:
            plt.title(f"Tour du joueur {env.current_player} (Rouge: {env.pawns_to_place[1]} jetons, Bleu: {env.pawns_to_place[-1]} jetons)")
        
        plt.draw()
        
        if env.is_phase1_over():
            plt.title("Phase de placement terminée !")
            plt.draw()
    
    ax.figure.canvas.mpl_connect('button_press_event', on_click)
    plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
    plt.title(f"Tour du joueur {env.current_player} (Rouge: {env.pawns_to_place[1]} jetons, Bleu: {env.pawns_to_place[-1]} jetons)")
    plt.show()

def play_with_clicks_against_agent(agent_class, agent_player=1):
    """Jouer contre un agent - Version simplifiée"""
    from marelle_env import MarelleEnv
    import random
    
    env = MarelleEnv()
    agent = agent_class(agent_player)
    human_player = -agent_player
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def force_agent_turn():
        """Forcer l'agent à jouer son tour"""
        while env.current_player == agent_player and not env.is_phase1_over():
            if env.waiting_for_removal:
                removable = env.get_removable_pawns()
                if removable:
                    remove_choice = random.choice(removable)
                    env.remove_pawn(remove_choice)
                    print(f"L'agent a retiré le pion de la position {remove_choice}")
            else:
                agent_move = agent.choose_move(env)
                if agent_move is not None:
                    env.play_move(agent_move)
                    print(f"L'agent a joué sur la position {agent_move}")
                else:
                    break
    
    def on_click(event):
        if event.inaxes != ax:
            return
        
        # Vérifier que c'est le tour du joueur humain
        if env.current_player != human_player:
            print(f"Ce n'est pas votre tour ! C'est le tour du joueur {env.current_player}")
            return
        
        # Trouver le nœud le plus proche du clic
        pos = nx.get_node_attributes(env.G, "pos")
        min_dist = float('inf')
        closest_node = None
        
        for node, (x, y) in pos.items():
            dist = ((event.xdata - x)**2 + (event.ydata - y)**2)**0.5
            if dist < min_dist and dist < 1.0:
                min_dist = dist
                closest_node = node
        
        if closest_node is None:
            return
            
        # Mode suppression de pion
        if env.waiting_for_removal:
            if env.remove_pawn(closest_node):
                print(f"Pion retiré de la position {closest_node}")
                # Forcer l'agent à jouer
                force_agent_turn()
            else:
                print("Position invalide pour la suppression")
        # Mode placement de pion
        elif env.G.nodes[closest_node]["state"] == 0:
            try:
                env.play_move(closest_node)
                print(f"Vous avez placé un pion sur la position {closest_node}")
                
                # Forcer l'agent à jouer
                force_agent_turn()
                
            except Exception as e:
                print(f"Erreur: {e}")
        
        # Rafraîchir l'affichage
        ax.clear()
        plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
        
        # Afficher le statut
        if env.waiting_for_removal:
            current = "Vous" if env.current_player == human_player else "Agent"
            plt.title(f"⚡ Moulin formé par {current} ! Cliquez sur un pion adverse à retirer")
        else:
            current = "Vous" if env.current_player == human_player else "Agent"
            plt.title(f"Tour de {current} (Rouge: {env.pawns_to_place[1]} jetons, Bleu: {env.pawns_to_place[-1]} jetons)")
        
        plt.draw()
        
        if env.is_phase1_over():
            plt.title("Phase de placement terminée !")
            plt.draw()
    
    ax.figure.canvas.mpl_connect('button_press_event', on_click)
    plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
    plt.title(f"Vous jouez contre {agent_class.__name__} (Vous = Bleu, Agent = Rouge)")
    
    # Si l'agent commence, faire son premier coup
    if env.current_player == agent_player:
        print("L'agent commence...")
        force_agent_turn()
        ax.clear()
        plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
        plt.title(f"Tour de Vous (Rouge: {env.pawns_to_place[1]} jetons, Bleu: {env.pawns_to_place[-1]} jetons)")
        plt.draw()
    
    plt.show()
