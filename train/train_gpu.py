import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from environnement.marelle_env import MarelleEnv
from marelle_agents.agents import MarelleDualHeadAgent, RandomAgent, GreedyAgent
from environnement.visualisation import plot_marelle
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

def train_on_gpu(episodes=10000, save_interval=1000, log_interval=100):
    """Entraîne l'agent RL sur GPU"""
    
    # Créer le fichier de log
    log_filename = f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Vérifier si GPU disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")
    
    # Créer l'agent avec plus d'exploration
    agent = MarelleDualHeadAgent(player_id=1, learning_rate=0.001, epsilon=0.8)
    agent.network = agent.network.to(device)
    
    # Adversaire pour l'entraînement
    opponent = RandomAgent(player_id=-1)
    
    # Statistiques d'entraînement
    episode_rewards = []
    win_rates = []
    losses = []
    wins = 0
    
    print("Début de l'entraînement...")
    
    for episode in range(episodes):
        env = MarelleEnv()
        episode_reward = 0
        moves_count = 0
        episode_loss = 0
        
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
                    old_env = copy.deepcopy(env)
                    
                    if action_type == "capture":
                        env.remove_pawn(action)
                    else:
                        env.play_move_auto(action)
                    
                    reward = agent.reward_function(old_env, action, env)
                    episode_reward += reward
                    moves_count += 1
                    
                    next_state = agent.get_state_representation(env).to(device)
                    done = env.is_phase1_over()
                    agent.store_experience(state, action, reward, next_state, done, action_type)
            
            # Tour de l'adversaire
            else:
                opponent_move = opponent.choose_move(env)
                if opponent_move is not None:
                    env.play_move_auto(opponent_move)
        
        # Déterminer le gagnant
        our_pawns = sum(1 for n in env.G.nodes() if env.G.nodes[n]["state"] == agent.player_id)
        opponent_pawns = sum(1 for n in env.G.nodes() if env.G.nodes[n]["state"] == -agent.player_id)
        
        if our_pawns > opponent_pawns:
            wins += 1
        
        episode_rewards.append(episode_reward)
        win_rate = wins / (episode + 1)
        win_rates.append(win_rate)
        
        # Entraîner avec batch plus grand
        if len(agent.memory) >= 256:
            episode_loss = agent.train(batch_size=256)
            losses.append(episode_loss)
        else:
            losses.append(0.0)
        
        # Diminuer epsilon plus lentement
        agent.epsilon = max(0.05, agent.epsilon * 0.9999)
        
        # Log toutes les 100 époques
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_loss = np.mean(losses[-log_interval:]) if losses else 0.0
            
            log_message = (f"Épisode {episode + 1}/{episodes} - "
                          f"Récompense moyenne: {avg_reward:.2f} - "
                          f"Taux de victoire: {win_rate:.3f} - "
                          f"Epsilon: {agent.epsilon:.3f} - "
                          f"Loss moyenne: {avg_loss:.4f}")
            
            print(log_message)
            
            with open(log_filename, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {log_message}\n")
        
        # Test intermédiaire
        if (episode + 1) % 1000 == 0:
            test_wins = 0
            for _ in range(100):
                test_env = MarelleEnv()
                while not test_env.is_phase1_over():
                    if test_env.current_player == agent.player_id:
                        action = agent.choose_move(test_env)
                        if action is not None:
                            test_env.play_move_auto(action)
                    else:
                        action = opponent.choose_move(test_env)
                        if action is not None:
                            test_env.play_move_auto(action)
                
                our_pawns = sum(1 for n in test_env.G.nodes() if test_env.G.nodes[n]["state"] == agent.player_id)
                opponent_pawns = sum(1 for n in test_env.G.nodes() if test_env.G.nodes[n]["state"] == -agent.player_id)
                if our_pawns > opponent_pawns:
                    test_wins += 1
            
            test_win_rate = test_wins / 100
            print(f"Test intermédiaire: {test_win_rate:.3f}")
    
    # Sauvegarder le modèle final
    torch.save(agent.network.state_dict(), "marelle_agent_final.pth")
    print("Entraînement terminé !")
    
    # Afficher les statistiques
    plot_training_stats(episode_rewards, win_rates, losses)
    
    return agent

def plot_training_stats(rewards, win_rates, losses):
    """Affiche les statistiques d'entraînement avec loss"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Récompenses
    ax1.plot(rewards)
    ax1.set_title("Récompenses par épisode")
    ax1.set_xlabel("Épisode")
    ax1.set_ylabel("Récompense")
    ax1.grid(True, alpha=0.3)
    
    # Taux de victoire
    ax2.plot(win_rates)
    ax2.set_title("Taux de victoire")
    ax2.set_xlabel("Épisode")
    ax2.set_ylabel("Taux de victoire")
    ax2.axhline(y=0.5, color='r', linestyle='--', label='50%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss
    ax3.plot(losses)
    ax3.set_title("Loss d'entraînement")
    ax3.set_xlabel("Épisode")
    ax3.set_ylabel("Loss")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300, bbox_inches='tight')
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
                            env.play_move_auto(action)
                else:
                    action = opponent.choose_move(env)
                    if action is not None:
                        env.play_move_auto(action)
            
            # Déterminer le gagnant
            our_pawns = sum(1 for n in env.G.nodes() if env.G.nodes[n]["state"] == agent.player_id)
            opponent_pawns = sum(1 for n in env.G.nodes() if env.G.nodes[n]["state"] == -agent.player_id)
            
            if our_pawns > opponent_pawns:
                wins += 1
        
        win_rate = wins / test_games
        print(f"Contre {opponent_name}: {win_rate:.3f} ({wins}/{test_games})")

if __name__ == "__main__":
    # Lancer l'entraînement avec plus d'époques
    trained_agent = train_on_gpu(episodes=10000, save_interval=1000, log_interval=100)
    
    # Tester l'agent entraîné
    test_trained_agent("marelle_agent_final.pth") 