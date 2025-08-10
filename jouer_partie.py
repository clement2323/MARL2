from environnement.marelle_env import MarelleEnv
from environnement.visualisation import plot_marelle
from environnement.play_with_click import play_with_clicks, play_with_clicks_against_agent
from marelle_agents.agents import BaseAgent
from marelle_agents.strategies import greedy_placement, SmartRemoval, SmartPlacement, block_opponent, ModelStrategy
from marelle_agents.modeles import MarelleDualHeadNet

import random
import matplotlib.pyplot as plt

import torch
# Agent totalement random
agent_random = BaseAgent(player_id=1, name = "dumb")

# Agent offensif qui retire intelligemment
agent_offensif = BaseAgent(
    player_id=1,
    placement_strategy=greedy_placement,
    removal_strategy=SmartRemoval(1),
    name = "offensif"
)

# Agent défensif qui bloque et retire au hasard
agent_defensif = BaseAgent(
    player_id=-1,
    placement_strategy=block_opponent,
    removal_strategy=None,
    name ="defensif"
)

smart_agent = BaseAgent(
    player_id=1,
    placement_strategy=SmartPlacement(1),
    removal_strategy=SmartRemoval(1),
    name = "smart"
)


model = MarelleDualHeadNet()
device_detected = "cuda" if torch.cuda.is_available() else "cpu"

agent_ml = BaseAgent(
    player_id=1,
    placement_strategy=ModelStrategy(model, 1, mode="placement", device= device_detected),
    removal_strategy=ModelStrategy(model, 1, mode="removal", device =device_detected),
    name="ML Agent"
)
def main():
    print("=== JEU DE LA MARELLE ===")
    print("1. Mode automatique (simulation)")
    print("2. Mode interactif (2 joueurs)")
    print("3. Jouer contre RandomAgent")
    print("4. Jouer contre GreedyAgent")
    print("5. Jouer contre DefensiveAgent")
    print("6. Jouer contre SmartAgent")
    print("7. Modèle ML !")
    
    choice = input("Choisissez le mode (1-7): ")
    
    if choice == "1":
        # Mode automatique
        env = MarelleEnv()
        print("Simulation automatique en cours...")
        
        plt.close('all')
        
        while not env.is_full() and not env.is_phase1_over():
            move = random.choice(env.get_legal_moves())
            env.play_move(move)
            print(f"Joueur {env.current_player * -1} joue sur la position {move}")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_marelle(env.G, env.moves, ax)
        plt.title("Partie terminée - Mode automatique")
        plt.tight_layout()
        plt.show()
        
    elif choice == "2":
        # Mode interactif 2 joueurs
        print("Mode interactif - Cliquez sur les positions pour jouer")
        print("Rouge = Joueur 1, Bleu = Joueur 2")
        plt.close('all')
        play_with_clicks()
        
    elif choice == "3":
        # Contre RandomAgent
        print("Vous jouez contre RandomAgent (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(agent_random, 1)
        
    elif choice == "4":
        # Contre GreedyAgent
        print("Vous jouez contre GreedyAgent (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(agent_offensif, 1)
        
    elif choice == "5":
        # Contre DefensiveAgent
        print("Vous jouez contre DefensiveAgent (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(
            agent_defensif, 1)
        
    elif choice == "6":
        # Contre SmartAgent
        print("Vous jouez contre SmartAgent (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(smart_agent, 1)
    
    elif choice == "7":
        # Contre SmartAgent
        print("Vous jouez contre le modèle ! (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(agent_ml, 1)
        
    else:
        print("Choix invalide")

if __name__ == "__main__":
    main()



