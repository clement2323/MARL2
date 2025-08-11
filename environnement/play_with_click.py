import time
import matplotlib.pyplot as plt
import networkx as nx
import csv
import datetime
from environnement.visualisation import plot_marelle
from environnement.marelle_env import MarelleEnv

class GameLogger:
    def __init__(self, filename="game_logs.csv"):
        self.filename = filename
        self.game_id = None
        self.moves = []
        self.game_start_time = None
        
        # Créer le fichier CSV s'il n'existe pas
        try:
            with open(self.filename, 'r') as f:
                pass
        except FileNotFoundError:
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['game_id', 'timestamp', 'player', 'action_type', 'position', 'game_state'])
    
    def start_new_game(self):
        """Démarre une nouvelle partie"""
        self.game_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.moves = []
        self.game_start_time = datetime.datetime.now()
        print(f"🆕 Nouvelle partie démarrée: {self.game_id}")
    
    def log_move(self, player, action_type, position, env):
        """Enregistre un coup"""
        if self.game_id is None:
            self.start_new_game()
        
        # État du jeu après le coup
        game_state = {
            'current_player': env.current_player,
            'waiting_for_removal': env.waiting_for_removal,
            'pawns_red': env.pawns_to_place.get(1, 0),
            'pawns_blue': env.pawns_to_place.get(-1, 0),
            'board_state': {node: data['state'] for node, data in env.G.nodes(data=True)}
        }
        
        move_record = {
            'game_id': self.game_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'player': player,
            'action_type': action_type,
            'position': position,
            'game_state': str(game_state)
        }
        
        self.moves.append(move_record)
        
        # Sauvegarder immédiatement dans le CSV
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                move_record['game_id'],
                move_record['timestamp'],
                move_record['player'],
                move_record['action_type'],
                move_record['position'],
                move_record['game_state']
            ])
        
        print(f"📝 {action_type} par {player} sur position {position}")
    
    def end_game(self, winner):
        """Termine la partie et enregistre le résultat"""
        if self.game_id:
            print(f"🏁 Partie {self.game_id} terminée. Vainqueur: {winner}")
            self.game_id = None

# Instance globale du logger
game_logger = GameLogger()

def play_with_clicks():
    env = MarelleEnv()
    game_logger.start_new_game()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def on_click(event):
        if event.inaxes != ax:
            return
        
        pos = nx.get_node_attributes(env.G, "pos")
        closest_node, min_dist = None, float('inf')
        
        for node, (x, y) in pos.items():
            dist = ((event.xdata - x)**2 + (event.ydata - y)**2)**0.5
            if dist < min_dist and dist < 1.0:
                min_dist, closest_node = dist, node
        
        if closest_node is None:
            return

        # Mode suppression
        if env.waiting_for_removal:
            if closest_node in env.get_removable_pawns():
                env.remove_pawn(closest_node)
                game_logger.log_move("Joueur", "retrait", closest_node, env)
                print(f"Pion retiré de la position {closest_node}")
            else:
                print("Position invalide pour la suppression")
        # Mode placement avec limitation
        elif env.G.nodes[closest_node]["state"] == 0:
            if env.pawns_to_place[env.current_player] > 0:
                try:
                    env.play_move(closest_node)
                    game_logger.log_move("Joueur", "placement", closest_node, env)
                    print(f"Pion placé sur la position {closest_node}")
                except Exception as e:
                    print(f"Erreur: {e}")
            else:
                print("Vous n'avez plus de pions à placer")
        
        # Rafraîchir l'affichage
        ax.clear()
        plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())

        if env.waiting_for_removal:
            plt.title(f"⚡ Joueur {env.current_player} a formé un moulin ! Cliquez sur un pion adverse à retirer")
        else:
            plt.title(f"Tour du joueur {env.current_player} (Rouge: {env.pawns_to_place[1]} jetons, Bleu: {env.pawns_to_place[-1]} jetons)")
        
        plt.draw()

        if env.is_phase1_over():
            winner = env.get_winner()
            game_logger.end_game(winner)
            plt.title("Phase de placement terminée !")
            plt.draw()
    
    ax.figure.canvas.mpl_connect('button_press_event', on_click)
    plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
    plt.title(f"Tour du joueur {env.current_player} (Rouge: {env.pawns_to_place[1]} jetons, Bleu: {env.pawns_to_place[-1]} jetons)")
    plt.show()


def play_with_clicks_against_agent(agent, agent_player=1):
    env = MarelleEnv()
    human_player = -agent_player
    game_logger.start_new_game()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def show_phase1_end_message(env, ax):
        red_pawns = sum(1 for _, data in env.G.nodes(data=True) if data["state"] == 1)
        blue_pawns = sum(1 for _, data in env.G.nodes(data=True) if data["state"] == -1)
        diff = red_pawns - blue_pawns
        leader = "Rouge" if diff > 0 else "Bleu" if diff < 0 else "Égalité"

        ax.clear()
        plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
        plt.title(f" Fin de la phase 1 ! Leader : {leader} | Différentiel : {diff} (Rouge: {red_pawns}, Bleu: {blue_pawns})")
        plt.draw()

    def refresh_display():
        """Rafraîchit l'affichage avec l'état actuel"""
        ax.clear()
        removable = env.get_removable_pawns() if env.waiting_for_removal else []
        plot_marelle(env.G, env.moves, ax, removable)
        
        if env.waiting_for_removal:
            current = "Vous" if env.current_player == human_player else "Agent"
            plt.title(f"⚡ Moulin formé par {current} ! Cliquez sur un pion adverse à retirer\nPions supprimables: {removable}")
        else:
            current = "Vous" if env.current_player == human_player else "Agent"
            plt.title(f"Tour de {current} (Rouge: {env.pawns_to_place[1]} jetons, Bleu: {env.pawns_to_place[-1]} jetons)")
        
        plt.draw()

    def force_agent_turn():
        """Fait jouer l'agent tant que c'est son tour"""
        while env.current_player == agent_player and not env.is_phase1_over():
            agent_move = agent.choose_move(env)
            
            if env.waiting_for_removal:
                env.remove_pawn(agent_move)
                game_logger.log_move("Agent", "retrait", agent_move, env)
                print(f"L'agent a retiré le pion de la position {agent_move}")
            else:
                env.play_move(agent_move)
                game_logger.log_move("Agent", "placement", agent_move, env)
                print(f"L'agent a joué sur la position {agent_move}")

            # Rafraîchir l'affichage après chaque coup de l'agent
            refresh_display()

            if env.is_phase1_over():
                winner = env.get_winner()
                game_logger.end_game(winner)
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
            
        # Suppression avec vérification des pions supprimables
        if env.waiting_for_removal:
            removable_pawns = env.get_removable_pawns()
            if closest_node in removable_pawns:
                env.remove_pawn(closest_node)
                game_logger.log_move("Humain", "retrait", closest_node, env)
                print(f"Vous avez retiré le pion en {closest_node}")
                
                if env.is_phase1_over():
                    winner = env.get_winner()
                    game_logger.end_game(winner)
                    show_phase1_end_message(env, ax)
                    return
                force_agent_turn()
            else:
                print(f"Position invalide pour suppression. Pions supprimables: {removable_pawns}")
        # Placement
        elif env.G.nodes[closest_node]["state"] == 0:
            env.play_move(closest_node)
            game_logger.log_move("Humain", "placement", closest_node, env)
            print(f"Vous avez placé un pion en {closest_node}")
            
            if env.is_phase1_over():
                winner = env.get_winner()
                game_logger.end_game(winner)
                show_phase1_end_message(env, ax)
                return
            force_agent_turn()
        else:
            print("Position déjà occupée")
        
        # Rafraîchir affichage si pas fin
        if not env.is_phase1_over():
            refresh_display()

    ax.figure.canvas.mpl_connect('button_press_event', on_click)
    plot_marelle(env.G, env.moves, ax, env.get_removable_pawns())
    plt.title(f"Vous jouez contre {agent.name} (Vous = Bleu, Agent = Rouge)")

    if env.current_player == agent_player:
        print("L'agent commence...")
        force_agent_turn()
        refresh_display()
    
    plt.show()
