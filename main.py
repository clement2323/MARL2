from marelle_env import MarelleEnv
from visualisation import plot_marelle, play_with_clicks, play_with_clicks_against_agent
from agents import RandomAgent, GreedyAgent, DefensiveAgent, SmartAgent
import random
import matplotlib.pyplot as plt

def main():
    print("=== JEU DE LA MARELLE ===")
    print("1. Mode automatique (simulation)")
    print("2. Mode interactif (2 joueurs)")
    print("3. Jouer contre RandomAgent")
    print("4. Jouer contre GreedyAgent")
    print("5. Jouer contre DefensiveAgent")
    print("6. Jouer contre SmartAgent")
    
    choice = input("Choisissez le mode (1-6): ")
    
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
        play_with_clicks_against_agent(RandomAgent, 1)
        
    elif choice == "4":
        # Contre GreedyAgent
        print("Vous jouez contre GreedyAgent (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(GreedyAgent, 1)
        
    elif choice == "5":
        # Contre DefensiveAgent
        print("Vous jouez contre DefensiveAgent (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(DefensiveAgent, 1)
        
    elif choice == "6":
        # Contre SmartAgent
        print("Vous jouez contre SmartAgent (Rouge)")
        plt.close('all')
        play_with_clicks_against_agent(SmartAgent, 1)
        
    else:
        print("Choix invalide")

main()



