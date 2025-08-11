from environnement.marelle_env import MarelleEnv
from environnement.visualisation import plot_marelle
from environnement.play_with_click import play_with_clicks, play_with_clicks_against_agent
from marelle_agents.agents import BaseAgent
from marelle_agents.strategies import greedy_placement, SmartRemoval, SmartPlacement, block_opponent, ModelStrategy
from marelle_agents.modeles import MarelleDualHeadNet

import random
import matplotlib.pyplot as plt

from marelle_agents.agent_configs import AGENTS

def main():
    print("=== JEU DE LA MARELLE ===")
    print("1. Mode automatique (simulation)")
    print("2. Mode interactif (2 joueurs)")
    print("3. Jouer contre RandomAgent")
    print("4. Jouer contre GreedyAgent")
    print("5. Jouer contre DefensiveAgent")
    print("6. Jouer contre SmartAgent")
    print("7. Modèle ML !")
    print("8. Modèle AC !")
    
    choice = input("Choisissez le mode (1-7): ")
    
    if choice == "1":
        # Mode automatique
        env = MarelleEnv()
        print("Simulation automatique en cours...")
        
        plt.close('all')
        
        while not env.is_phase1_over():
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
        play_with_clicks_against_agent(AGENTS["random"](), 1)
        
    elif choice == "4":
        # Contre GreedyAgent
        print("Vous jouez contre GreedyAgent (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(AGENTS["offensif"](), 1)
        
    elif choice == "5":
        # Contre DefensiveAgent
        print("Vous jouez contre DefensiveAgent (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(
            AGENTS["defensif"](), 1)
        
    elif choice == "6":
        # Contre SmartAgent
        print("Vous jouez contre SmartAgent (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(AGENTS["smart"](), 1)
    
    elif choice == "7":
        # Contre SmartAgent
        print("Vous jouez contre le modèle ! (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(AGENTS["ml"](), 1)


    elif choice == "8":
        # Contre SmartAgent
        print("Vous jouez contre le modèle ! (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(AGENTS["ac"](), 1)
        
        
    else:
        print("Choix invalide")

if __name__ == "__main__":
    main()



