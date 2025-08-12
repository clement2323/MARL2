import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environnement.symetries import SYMMETRY_MAPPINGS
from marelle_agents.agents import BaseAgent
from marelle_agents.strategies import ModelStrategy
from environnement.marelle_env import MarelleEnv
from marelle_agents.agent_configs import AGENTS
from marelle_agents.modeles import ActorCriticModel, ActorCriticModelLarge

# -------------------------
# Rewards
# -------------------------
def calculate_turn_reward(env_before, env_after, player_id):
    reward = 0.0
    moulins_j1_avant = sum(1 for mill in env_before.mills if all(env_before.G.nodes[n]["state"] == 1 for n in mill))
    moulins_j2_avant = sum(1 for mill in env_before.mills if all(env_before.G.nodes[n]["state"] == -1 for n in mill))
    moulins_j1_apres = sum(1 for mill in env_after.mills if all(env_after.G.nodes[n]["state"] == 1 for n in mill))
    moulins_j2_apres = sum(1 for mill in env_after.mills if all(env_after.G.nodes[n]["state"] == -1 for n in mill))

    if player_id == 1:
        if moulins_j1_apres > moulins_j1_avant:
            reward += 0.1 * (moulins_j1_apres - moulins_j1_avant)
        if moulins_j2_apres > moulins_j2_avant:
            reward -= 0.1 * (moulins_j2_apres - moulins_j2_avant)
    else:
        if moulins_j2_apres > moulins_j2_avant:
            reward += 0.1 * (moulins_j2_apres - moulins_j2_avant)
        if moulins_j1_apres > moulins_j1_avant:
            reward -= 0.1 * (moulins_j1_apres - moulins_j1_avant)
    return reward

def save_env_state(env):
    env_copy = MarelleEnv()
    env_copy.G = env.G.copy()
    env_copy.current_player = env.current_player
    env_copy.waiting_for_removal = env.waiting_for_removal
    env_copy.pawns_to_place = env.pawns_to_place.copy()
    return env_copy

def to_canonical(state_vec):
    """
    Retourne la forme canonique d'un état et le nom de la symétrie appliquée.
    """
    best_state = None
    best_sym = None
    
    for sym_name, mapping in SYMMETRY_MAPPINGS.items():
      
        transformed = [state_vec[i] for i in mapping]
        if (best_state is None) or (transformed < best_state):
            best_state = transformed
            best_sym = sym_name
            

    return best_state, best_sym


def map_moves_to_canonical(legal_moves, sym_name):
    """
    Applique la symétrie donnée aux coups légaux.
    """
    mapping = SYMMETRY_MAPPINGS[sym_name]
    return [mapping[m] for m in legal_moves]


def map_move_from_canonical(move_idx, sym_name):
    """
    Transforme un coup choisi dans la base canonique
    vers son index dans le plateau réel.
    """
    mapping = SYMMETRY_MAPPINGS[sym_name]
    inverse_mapping = {new: orig for orig, new in enumerate(mapping)}
    return inverse_mapping[move_idx]

# -------------------------
# Entraînement
# -------------------------
def train_actor_critic(
    model_path_train=None,
    model_path="save_models/marelle_model_actor_critic.pth",
    agents_to_train_against=None,
    total_episodes=30000,
    lr=1e-3,
    exploration_epsilon=0.1,
    switch_every=1,
    save_every=2000,
    log_every=500,
    add_self_every=30000,
    max_agents=8,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    #model = ActorCriticModel().to(device)
    model = ActorCriticModelLarge().to(device)
    # Charger un modèle existant si chemin fourni
    if model_path_train is not None and os.path.exists(model_path_train):
        print(f"[LOAD] Chargement du modèle depuis {model_path_train}")
        model.load_state_dict(torch.load(model_path_train, map_location=device))
    else:
        if model_path_train:
            print(f"[WARN] Aucun fichier trouvé à {model_path_train}, initialisation d'un nouveau modèle")
        else:
            print("[INIT] Nouveau modèle, pas de chemin de chargement fourni")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    gamma = 0.99

    if agents_to_train_against is None or "offensif" not in agents_to_train_against:
        raise ValueError("La liste agents_to_train_against doit contenir au moins 'offensif'")

    dynamic_agents = []
    stats_by_agent = {name: {"episodes": 0, "wins": 0, "losses": 0, "draws": 0}
                      for name in agents_to_train_against.keys()}
    overall = {"episodes": 0, "wins": 0, "losses": 0, "draws": 0}
    first_player = 1

    for ep in range(1, total_episodes + 1):
        if (ep - 1) % switch_every == 0:
            selected_name = random.choice(list(agents_to_train_against.keys()))
            agent_contre = agents_to_train_against[selected_name]()
            first_player *= -1

        env = MarelleEnv()
        env.current_player = first_player
        placement_strategy = ModelStrategy(model, player_id=1, mode="placement", device=device)
        removal_strategy = ModelStrategy(model, player_id=1, mode="removal", device=device)
        agent_ml = BaseAgent(1, placement_strategy, removal_strategy)

        log_probs, values, rewards = [], [], []
        reward_turn = 0.0

        while not env.is_phase1_over():
            env_before = save_env_state(env)
            current_agent = agent_ml if env.current_player == 1 else agent_contre

            if env.current_player == 1:
                canon_state, sym_used = to_canonical(placement_strategy.encode_state(env))
                
                canon_state = torch.tensor(
                    canon_state,
                    dtype=torch.float32, device=device
                ).unsqueeze(0)
                
                


                logits_place, logits_remove, value = model(canon_state)

                if env.waiting_for_removal:
                    legal_moves = env.get_removable_pawns()
                    canon_legal_moves = map_moves_to_canonical(legal_moves,sym_used)    
                    mask = torch.full((24,), float('-inf'), device=device)
                    mask[canon_legal_moves] = 0.0
                    probs = F.softmax(logits_remove + mask, dim=-1)
                else:
                    legal_moves = env.get_legal_moves()
                    canon_legal_moves = map_moves_to_canonical(legal_moves,sym_used)    
                    mask = torch.full((24,), float('-inf'), device=device)
                    mask[canon_legal_moves] = 0.0
                    probs = F.softmax(logits_place + mask, dim=-1)

                if random.random() < exploration_epsilon:
                    canon_move = random.choice(canon_legal_moves)
                else:
                    canon_move = torch.multinomial(probs, 1).item()

                log_prob = torch.log(probs[0, canon_move] + 1e-8)
                log_probs.append(log_prob)
                values.append(value.squeeze())
                real_move = map_move_from_canonical(canon_move, sym_used)

                if env.waiting_for_removal:
                    env.remove_pawn(real_move)
                    rewards.append(0.0)
                else:
                    env.play_move(real_move)
                    reward_turn += calculate_turn_reward(env_before, env, player_id=1)
                    rewards.append(reward_turn)
                    reward_turn = 0.0

            else:
                move = current_agent.choose_move(env)
                if env.waiting_for_removal:
                    env.remove_pawn(move)
                else:
                    env.play_move(move)
                    reward_turn += calculate_turn_reward(env_before, env, player_id=1)
                    if rewards:
                        rewards[-1] += reward_turn
                    reward_turn = 0.0

        winner = env.get_winner()
        final_reward = 1.0 if winner == 1 else -1.0 if winner == -1 else 0.5 # 0.5 for nul !!!! =signal
        rewards = [r + final_reward for r in rewards]

        # Compute returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        log_probs_t = torch.stack(log_probs)
        values_t = torch.stack(values)
        advantages = returns - values_t.detach()

        actor_loss = -(log_probs_t * advantages).mean()
        critic_loss = F.mse_loss(values_t, returns)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        overall["episodes"] += 1
        if winner == 1:
            overall["wins"] += 1
            stats_by_agent[selected_name]["wins"] += 1
        elif winner == -1:
            overall["losses"] += 1
            stats_by_agent[selected_name]["losses"] += 1
        else:
            overall["draws"] += 1
            stats_by_agent[selected_name]["draws"] += 1
        stats_by_agent[selected_name]["episodes"] += 1

        # Logging
        if ep % log_every == 0:
            w, l, d, total = overall["wins"], overall["losses"], overall["draws"], overall["episodes"]
            print(f"EP {ep}/{total_episodes} | W:{w/total*100:.1f}% L:{l/total*100:.1f}% D:{d/total*100:.1f}%")
            for name, s in stats_by_agent.items():
                e = s["episodes"]
                if e > 0:
                    print(f"   - {name}: ep={e} W:{s['wins']/e*100:.1f}% L:{s['losses']/e*100:.1f}% D:{s['draws']/e*100:.1f}%")

        # Sauvegarde périodique
        if ep % save_every == 0:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"[SAVE] Modèle sauvegardé -> {model_path} (ep {ep})")

        # Ajout auto-modèle
        if ep % add_self_every == 0:
            self_model_path = f"save_models/marelle_model_self_{ep}.pth"
            torch.save(model.state_dict(), self_model_path)
            new_agent_name = f"self_{ep}"
            agents_to_train_against[new_agent_name] = lambda p=self_model_path: AGENTS["ac_large"](model_path=p)
            stats_by_agent[new_agent_name] = {"episodes": 0, "wins": 0, "losses": 0, "draws": 0}
            dynamic_agents.append(new_agent_name)
            print(f"[ADD] Nouvel agent ajouté : {new_agent_name}")

            # Limiter à max_agents mais garder 'offensif'
            while len(agents_to_train_against) > max_agents:
                oldest = dynamic_agents.pop(0)
                if oldest != "offensif":
                    del agents_to_train_against[oldest]
                    del stats_by_agent[oldest]
                    print(f"[REMOVE] Agent le plus ancien retiré : {oldest}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("[DONE] Entraînement terminé. Modèle sauvegardé.")
    return model, stats_by_agent, overall

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    training_agents = {
        "offensif": AGENTS["offensif"],
        "ac6": lambda: AGENTS["ac_large"](model_path="save_models/marelle_model_self_45000.pth"),
        }

    model, stats, overall = train_actor_critic(
        model_path_train="save_models/marelle_model_actor_critic_large.pth",#None
        model_path="save_models/marelle_model_actor_critic_large.pth",
        agents_to_train_against=training_agents,
        total_episodes=5_000_000,
        lr=1e-3,
        exploration_epsilon=0.05,
        switch_every=10,
        save_every=30000,
        log_every=1000,
        add_self_every=15000,
        max_agents=3
    )
