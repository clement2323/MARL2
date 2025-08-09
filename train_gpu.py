import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from marelle_env import MarelleEnv
from agents import MarelleDualHeadAgent, RandomAgent, GreedyAgent
from visualisation import plot_marelle
import matplotlib.pyplot as plt

def train_on_gpu(episodes=1000, save_interval=100):
    """Entraîne l'agent RL sur GPU"""
    
    # Vérifier si GPU disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")
    
    # Créer l'agent et le déplacer sur GPU
    agent = MarelleDualHeadAgent(player_id=1, learning_rate=0.001, epsilon=0.3)
    agent.network = agent.network.to(device)
    
    # Adversaire pour l'entraînement
    opponent = RandomAgent(player_id=-1)
    
    # Statistiques d'entraînement
    episode_rewards = []
    win_rates = []
    wins = 0
    
    print("Début de l'entraînement...")
    
    for episode in range(episodes):
        env = MarelleEnv()
        episode_reward = 0
        moves_count = 0
        
        while not env.is_phase1_over():
            # Tour de l'agent RL
            if env.current_player == agent.player_id:
                state = agent.get_state_representation(env).to(device)
                
                if env.waiting_for_removal:
                    action = agent._choose_capture(env)
                    action_type = "capture"
                else:
                    action = agent._choose_placement(env)
                    action_type = "placement"
                
                if action is not None:
                    # Sauvegarder l'état avant
                    old_env = copy.deepcopy(env)
                    
                    # Jouer le coup
                    if action_type == "capture":
                        env.remove_pawn(action)
                    else:
                        env.play_move(action)
                    
                    # Calculer la récompense
                    reward = agent.reward_function(old_env, action, env)
                    episode_reward += reward
                    moves_count += 1
                    
                    # Stocker l'expérience
                    next_state = agent.get_state_representation(env).to(device)
                    done = env.is_phase1_over()
                    agent.store_experience(state, action, reward, next_state, done, action_type)
            
            # Tour de l'adversaire
            else:
                opponent_move = opponent.choose_move(env)
                if opponent_move is not None:
                    env.play_move(opponent_move)
        
        # Déterminer le gagnant (simplifié)
        our_pawns = sum(1 for n in env.G.nodes() if env.G.nodes[n]["state"] == agent.player_id)
        opponent_pawns = sum(1 for n in env.G.nodes() if env.G.nodes[n]["state"] == -agent.player_id)
        
        if our_pawns > opponent_pawns:
            wins += 1
            episode_reward += 10  # Bonus pour la victoire
        
        episode_rewards.append(episode_reward)
        win_rate = wins / (episode + 1)
        win_rates.append(win_rate)
        
        # Entraîner l'agent
        if len(agent.memory) >= 32:
            agent.train(batch_size=32)
        
        # Diminuer epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.9995)
        
        # Affichage des progrès
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Épisode {episode + 1}/{episodes} - "
                  f"Récompense moyenne: {avg_reward:.2f} - "
                  f"Taux de victoire: {win_rate:.3f} - "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Sauvegarder le modèle
        if (episode + 1) % save_interval == 0:
            torch.save(agent.network.state_dict(), f"marelle_agent_episode_{episode + 1}.pth")
            print(f"Modèle sauvegardé à l'épisode {episode + 1}")
    
    # Sauvegarder le modèle final
    torch.save(agent.network.state_dict(), "marelle_agent_final.pth")
    print("Entraînement terminé !")
    
    # Afficher les statistiques
    plot_training_stats(episode_rewards, win_rates)
    
    return agent

def plot_training_stats(rewards, win_rates):
    """Affiche les statistiques d'entraînement"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Récompenses
    ax1.plot(rewards)
    ax1.set_title("Récompenses par épisode")
    ax1.set_xlabel("Épisode")
    ax1.set_ylabel("Récompense")
    
    # Taux de victoire
    ax2.plot(win_rates)
    ax2.set_title("Taux de victoire")
    ax2.set_xlabel("Épisode")
    ax2.set_ylabel("Taux de victoire")
    ax2.axhline(y=0.5, color='r', linestyle='--', label='50%')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def test_trained_agent(agent_path, test_games=100):
    """Teste l'agent entraîné"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Charger l'agent
    agent = MarelleDualHeadAgent(player_id=1)
    agent.network.load_state_dict(torch.load(agent_path, map_location=device))
    agent.network = agent.network.to(device)
    agent.epsilon = 0.01  # Très peu d'exploration
    
    opponents = [
        ("RandomAgent", RandomAgent(-1)),
        ("GreedyAgent", GreedyAgent(-1))
    ]
    
    for opponent_name, opponent in opponents:
        wins = 0
        for _ in range(test_games):
            env = MarelleEnv()
            
            while not env.is_phase1_over():
                if env.current_player == agent.player_id:
                    action = agent.choose_move(env)
                    if action is not None:
                        if env.waiting_for_removal:
                            env.remove_pawn(action)
                        else:
                            env.play_move(action)
                else:
                    action = opponent.choose_move(env)
                    if action is not None:
                        env.play_move(action)
            
            # Déterminer le gagnant
            our_pawns = sum(1 for n in env.G.nodes() if env.G.nodes[n]["state"] == agent.player_id)
            opponent_pawns = sum(1 for n in env.G.nodes() if env.G.nodes[n]["state"] == -agent.player_id)
            
            if our_pawns > opponent_pawns:
                wins += 1
        
        win_rate = wins / test_games
        print(f"Contre {opponent_name}: {win_rate:.3f} ({wins}/{test_games})")

if __name__ == "__main__":
    # Lancer l'entraînement
    trained_agent = train_on_gpu(episodes=1000)
    
    # Tester l'agent entraîné
    test_trained_agent("marelle_agent_final.pth") 