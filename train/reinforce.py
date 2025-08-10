import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import sys
import os
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marelle_agents.agents import BaseAgent
from marelle_agents.strategies import ModelStrategy, greedy_placement, SmartRemoval
from marelle_agents.modeles import MarelleDualHeadNet
from environnement.marelle_env import MarelleEnv


def train_reinforce(env, model, num_episodes=1000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gamma = 0.99

    loss_history = deque(maxlen=100)
    winner_history = deque(maxlen=100)

    best_loss = float("inf")
    best_model_state = None

    device_selected = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device_selected)

    for episode in range(num_episodes):
        env = MarelleEnv()

        # Agent ML
        placement_strategy = ModelStrategy(model, player_id=1, mode="placement", device=device_selected)
        removal_strategy = ModelStrategy(model, player_id=1, mode="removal", device=device_selected)
        agent_ml = BaseAgent(1, placement_strategy, removal_strategy)

        # Agent adverse (offensif)
        agent_contre = BaseAgent(
            player_id=-1,
            placement_strategy=greedy_placement,
            removal_strategy=SmartRemoval(-1),
            name="offensif"
        )

        agent_contre = BaseAgent(player_id=1, name = "dumb")


        log_probs, rewards = [], []

        while not env.is_phase1_over():
            current_agent = agent_ml if env.current_player == 1 else agent_contre
            move = current_agent.choose_move(env)

            if env.waiting_for_removal:
                if env.current_player == 1:
                    state = torch.tensor(
                        placement_strategy.encode_state(env),
                        dtype=torch.float32, device=device_selected
                    ).unsqueeze(0)
                    _, logits_remove = model(state)
                    mask = placement_strategy._create_action_mask(env.get_removable_pawns(), 24)
                    probs = F.softmax(logits_remove + mask, dim=-1)
                    log_prob = torch.log(probs[0, move] + 1e-8)
                    log_probs.append(log_prob)
                env.remove_pawn(move)

            else:
                if env.current_player == 1:
                    state = torch.tensor(
                        placement_strategy.encode_state(env),
                        dtype=torch.float32, device=device_selected
                    ).unsqueeze(0)
                    logits_place, _ = model(state)
                    mask = placement_strategy._create_action_mask(env.get_legal_moves(), 24)
                    probs = F.softmax(logits_place + mask, dim=-1)
                    log_prob = torch.log(probs[0, move] + 1e-8)
                    log_probs.append(log_prob)
                env.play_move(move)

        # Reward final basé sur get_winner()
        winner = env.get_winner()
        reward = 1 if winner == 1 else -1 if winner == -1 else 0
        rewards = [reward] * len(log_probs)

        # Calcul des retours (REINFORCE)
        returns, G = [], 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device_selected)

        # Normalisation (centrage-réduction)
        # Loss = -E[log_prob * return]
        loss = -torch.sum(torch.stack(log_probs) * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        winner_history.append(winner)

        if episode % 1000 == 0:
            avg_loss = sum(loss_history) / len(loss_history)
            avg_win_rate = sum(1 for w in winner_history if w == 1) / len(winner_history)

            print(f"[Episode {episode}] Avg Loss={avg_loss:.4f} Avg Win Rate={avg_win_rate:.2%}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict().copy()
                torch.save(best_model_state, "marelle_model_best.pth")
                print(f"🎯 Nouveau meilleur modèle ! Loss: {best_loss:.4f}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"🏆 Modèle final chargé avec la meilleure Loss: {best_loss:.4f}")

    return model


if __name__ == "__main__":
    model = MarelleDualHeadNet()
    trained_model = train_reinforce(env=MarelleEnv(), model=model, num_episodes=30000)
    torch.save(trained_model.state_dict(), "marelle_model_final.pth")
