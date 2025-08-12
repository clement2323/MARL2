import random
import torch
import torch.nn.functional as F

from environnement.symetries import map_move_from_canonical, map_moves_to_canonical, to_canonical
# ==== Stratégies de base ====
def random_placement(env, legal_moves):
    return random.choice(legal_moves)

def random_removal(env, removable):
    return random.choice(removable)

def greedy_placement(env, legal_moves):
    """Essaye de former un moulin sinon joue au hasard."""
    for move in legal_moves:
        original_state = env.G.nodes[move]["state"]
        env.G.nodes[move]["state"] = env.current_player
        if env._forms_mill(move, env.current_player):
            env.G.nodes[move]["state"] = original_state
            return move
        env.G.nodes[move]["state"] = original_state
    return random.choice(legal_moves)

def block_opponent(env, legal_moves):
    opponent = -env.current_player
    for move in legal_moves:
        original_state = env.G.nodes[move]["state"]
        env.G.nodes[move]["state"] = opponent
        if env._forms_mill(move, opponent):
            env.G.nodes[move]["state"] = original_state
            return move
        env.G.nodes[move]["state"] = original_state
    return random.choice(legal_moves)


# ==== Stratégies intelligentes ====
class SmartPlacement:
    def __init__(self, player_id):
        self.player_id = player_id

    def __call__(self, env, legal_moves):
        opponent = -self.player_id
        # Bloquer
        for move in legal_moves:
            env.G.nodes[move]["state"] = opponent
            if env._forms_mill(move, opponent):
                env.G.nodes[move]["state"] = 0
                return move
            env.G.nodes[move]["state"] = 0
        # Faire un moulin
        for move in legal_moves:
            env.G.nodes[move]["state"] = self.player_id
            if env._forms_mill(move, self.player_id):
                env.G.nodes[move]["state"] = 0
                return move
            env.G.nodes[move]["state"] = 0
        return random.choice(legal_moves)


class SmartRemoval:
    def __init__(self, player_id):
        self.player_id = player_id

    def __call__(self, env, removable):
        opponent = -self.player_id
        for pawn in removable:
            for neighbor in env.G.neighbors(pawn):
                env.G.nodes[neighbor]["state"] = opponent
                if env._forms_mill(neighbor, opponent):
                    env.G.nodes[neighbor]["state"] = 0
                    return pawn
                env.G.nodes[neighbor]["state"] = 0
        return random.choice(removable)


# ==== Stratégie basée sur un modèle ====
class ModelStrategy:
    def __init__(self, model, player_id, device="cpu", mode="placement"):
        self.model = model
        self.player_id = player_id
        self.device = device
        self.mode = mode  # "placement" ou "removal"

    def __call__(self, env, moves):
        state = self.encode_state(env)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits_place, logits_remove = self.model(state_t)
            logits = logits_place if self.mode == "placement" else logits_remove
            mask = self._create_action_mask(moves, 24)
            masked_logits = logits + mask
            probs = F.softmax(masked_logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
        return action

    def encode_state(self, env):
        return [env.G.nodes[i]["state"] * (1 if env.current_player == self.player_id else -1) for i in range(24)]

    def _create_action_mask(self, legal_moves, size):
        mask = torch.full((size,), -1e9, device=self.device)
        for move in legal_moves:
            mask[move] = 0
        return mask

# --- CanonicalModelStrategy : remplace ModelStrategy.__call__ (même signature) ---
class CanonicalModelStrategy(ModelStrategy):
    def __call__(self, env, moves):
        """
        env: environment
        moves: list of legal moves in env (original indices)
        Retourne action en indices réels (original index)
        """
        # 1) encode_state (hérité) -> état vu depuis la perspective du joueur (list)
        state = self.encode_state(env)  # déjà en orientation self.player_id
        # 2) canonicalize
        canon_state, sym_used = to_canonical(state)
        state_t = torch.tensor(canon_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        # 3) forward (gérer modèle actor-only ou actor+critic)
        with torch.no_grad():
            outputs = self.model(state_t)
            # outputs peut être (logits_place, logits_remove) ou (logits_place, logits_remove, value)
            if isinstance(outputs, (tuple, list)):
                if len(outputs) == 3:
                    logits_place, logits_remove, _ = outputs
                elif len(outputs) == 2:
                    logits_place, logits_remove = outputs
                else:
                    raise RuntimeError("Model returned unexpected number of outputs")
            else:
                raise RuntimeError("Model must return a tuple/list of tensors")

            logits = logits_place if self.mode == "placement" else logits_remove

            # 4) map legal moves -> canonical indices and mask
            canon_moves = map_moves_to_canonical(moves, sym_used)
            mask = self._create_action_mask(canon_moves, logits.shape[-1])  # uses existing helper
            masked_logits = logits + mask
            probs = F.softmax(masked_logits, dim=-1)

            canon_action = torch.multinomial(probs, 1).item()

        # 5) map canonical action back to original index and return
        real_action = map_move_from_canonical(canon_action, sym_used)
        return real_action
