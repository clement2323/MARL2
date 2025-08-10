import matplotlib.pyplot as plt
import networkx as nx
from environnement.visualisation import plot_marelle  # à adapter à ton projet
from environnement.marelle_env import MarelleEnv

def play_with_clicks():

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

def play_with_clicks_against_agent(agent, agent_player=1):
    """
    Jouer contre un agent (objet BaseAgent avec placement_strategy et removal_strategy)
    """
    
    env = MarelleEnv()
    human_player = -agent_player
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def show_phase1_end_message(env, ax):
        """Affiche un message à la fin de la phase 1 avec le différentiel de pions."""
        red_pawns = sum(1 for _, data in env.G.nodes(data=True) if data["state"] == 1)
        blue_pawns = sum(1 for _, data in env.G.nodes(data=True) if data["state"] == -1)
        diff = red_pawns - blue_pawns

        if diff > 0:
            leader = "Rouge"
        elif diff < 0:
            leader = "Bleu"
        else:
            leader = "Égalité"

        ax.clear()
        plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
        plt.title(f" Fin de la phase 1 ! Leader : {leader} | Différentiel : {diff} (Rouge: {red_pawns}, Bleu: {blue_pawns})")
        plt.draw()


    def force_agent_turn():
        """Fait jouer l'agent tant que c'est son tour"""
        while env.current_player == agent_player and not env.is_phase1_over():
            if env.waiting_for_removal:
                removable = env.get_removable_pawns()
                if removable:
                    remove_choice = agent.removal_strategy(env, removable)
                    env.remove_pawn(remove_choice)
                    print(f"L'agent a retiré le pion de la position {remove_choice}")
            else:
                legal_moves = env.get_legal_moves()
                if legal_moves:
                    agent_move = agent.placement_strategy(env, legal_moves)
                    env.play_move(agent_move)
                    print(f"L'agent a joué sur la position {agent_move}")
                else:
                    break

            if env.is_phase1_over():  # ⬅️ Vérification fin phase 1
                show_phase1_end_message(env, ax)
                return


    def on_click(event):
        if event.inaxes != ax:
            return
        
        # Vérifier tour humain
        if env.current_player != human_player:
            print(f"Ce n'est pas votre tour !")
            return
        
        # Trouver nœud le plus proche
        pos = nx.get_node_attributes(env.G, "pos")
        closest_node = None
        min_dist = float('inf')
        
        for node, (x, y) in pos.items():
            dist = ((event.xdata - x)**2 + (event.ydata - y)**2)**0.5
            if dist < min_dist and dist < 1.0:
                min_dist = dist
                closest_node = node
        
        if closest_node is None:
            return
            
        # Suppression
        if env.waiting_for_removal:
            print(closest_node)
            if closest_node in env.get_removable_pawns():
                env.remove_pawn(closest_node)
                print(f"Vous avez retiré le pion en {closest_node}")
                if env.is_phase1_over():
                    show_phase1_end_message(env, ax)
                    return
                force_agent_turn()
            else:
                print("Position invalide pour suppression")
        # Placement
        elif env.G.nodes[closest_node]["state"] == 0:
            env.play_move(closest_node)
            print(f"Vous avez placé un pion en {closest_node}")
            if env.is_phase1_over():
                show_phase1_end_message(env, ax)
                return
            force_agent_turn()
        else:
            print("Position déjà occupée")
        
        # Rafraîchir affichage si pas fin
        if not env.is_phase1_over():
            ax.clear()
            plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
            
            if env.waiting_for_removal:
                current = "Vous" if env.current_player == human_player else "Agent"
                plt.title(f"⚡ Moulin formé par {current} ! Cliquez sur un pion adverse à retirer")
            else:
                current = "Vous" if env.current_player == human_player else "Agent"
                plt.title(f"Tour de {current} (Rouge: {env.pawns_to_place[1]} jetons, Bleu: {env.pawns_to_place[-1]} jetons)")
            
            plt.draw()

    
    ax.figure.canvas.mpl_connect('button_press_event', on_click)
    plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
    plt.title(f"Vous jouez contre {type(agent).__name__} (Vous = Bleu, Agent = Rouge)")
    
    # Si l'agent commence
    if env.current_player == agent_player:
        print("L'agent commence...")
        force_agent_turn()
        ax.clear()
        plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
        plt.title(f"Tour de Vous")
        plt.draw()
    
    plt.show()
