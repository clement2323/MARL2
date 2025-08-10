import random
from collections import deque

class BaseAgent:
    """Agent de base avec stratégie de placement et de retrait séparées."""
    def __init__(self, player_id, placement_strategy=None, removal_strategy=None, name = None):
        self.player_id = player_id
        # Stratégies par défaut = aléatoires
        self.placement_strategy = placement_strategy or self.random_placement
        self.removal_strategy = removal_strategy or self.random_removal
        self.name = name if name is not None else self.__class__.__name__

    def choose_move(self, env):
        """Décide quoi faire selon le contexte du jeu."""
        if env.waiting_for_removal:
            # Mode suppression
            removable = env.get_removable_pawns()
            if removable:
                return self.removal_strategy(removable, env)
            return None
        else:
            # Mode placement
            legal_moves = env.get_legal_moves()
            if legal_moves:
                return self.placement_strategy(legal_moves, env)
            return None

    # ==== Stratégies de base ====
    def random_placement(self, env, legal_moves):
        return random.choice(legal_moves)

    def random_removal(self, env, removable):
        return random.choice(removable)


# ==== Exemple de stratégies spécialisées ====

def greedy_placement(env,legal_moves):
    """Essaye de former un moulin sinon joue au hasard."""
    for move in legal_moves:
        original_state = env.G.nodes[move]["state"]
        env.G.nodes[move]["state"] = env.current_player
        if env._forms_mill(move, env.current_player):
            env.G.nodes[move]["state"] = original_state
            return move
        env.G.nodes[move]["state"] = original_state
    return random.choice(legal_moves)


def block_opponent(env,legal_moves):
    """Essaye de bloquer un moulin adverse sinon joue au hasard."""
    opponent = -env.current_player
    for move in legal_moves:
        original_state = env.G.nodes[move]["state"]
        env.G.nodes[move]["state"] = opponent
        if env._forms_mill(move, opponent):
            env.G.nodes[move]["state"] = original_state
            return move
        env.G.nodes[move]["state"] = original_state
    return random.choice(legal_moves)


def smart_removal(env,removable):
    """Retire en priorité un pion qui casse un moulin adverse."""
    opponent = -env.current_player
    for pawn in removable:
        # Simuler la suppression
        env.G.nodes[pawn]["state"] = 0
        mills = any(env._forms_mill(pos, opponent) for pos in env.G.nodes if env.G.nodes[pos]["state"] == opponent)
        env.G.nodes[pawn]["state"] = opponent
        if mills:
            return pawn
    return random.choice(removable)


class SmartPlacementStrategy:
    def __init__(self, player_id):
        self.player_id = player_id

    def __call__(self, env, legal_moves):
        opponent = -self.player_id
        
        # 1️⃣ Former un moulin si possible
        for move in legal_moves:
            env.G.nodes[move]["state"] = self.player_id
            if env._forms_mill(move, self.player_id):
                env.G.nodes[move]["state"] = 0
                return move
            env.G.nodes[move]["state"] = 0
        
        # 2️⃣ Bloquer un moulin adverse
        for move in legal_moves:
            env.G.nodes[move]["state"] = opponent
            if env._forms_mill(move, opponent):
                env.G.nodes[move]["state"] = 0
                return move
            env.G.nodes[move]["state"] = 0
        
        # 3️⃣ Coup par défaut (aléatoire)
        return random.choice(legal_moves)


class SmartRemovalStrategy:
    def __init__(self, player_id):
        self.player_id = player_id

    def __call__(self, env, removable_pawns):
        opponent = -self.player_id
        
        # Supprimer un pion qui contribue à un moulin imminent
        for pawn in removable_pawns:
            for neighbor in env.G.neighbors(pawn):
                env.G.nodes[neighbor]["state"] = opponent
                if env._forms_mill(neighbor, opponent):
                    env.G.nodes[neighbor]["state"] = 0
                    return pawn
                env.G.nodes[neighbor]["state"] = 0
        
        # Sinon supprimer aléatoirement
        return random.choice(removable_pawns)

# ==== Exemple d'instanciation ====

# Agent totalement random
agent_random = BaseAgent(player_id=1, name = "dumb")

# Agent offensif qui retire intelligemment
agent_offensif = BaseAgent(
    player_id=1,
    placement_strategy=greedy_placement,
    removal_strategy=smart_removal,
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
    placement_strategy=SmartPlacementStrategy(1),
    removal_strategy=SmartRemovalStrategy(1),
    name = "smart"
)



