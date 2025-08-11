import random

class BaseAgent:
    def __init__(self, player_id, placement_strategy=None, removal_strategy=None, name=None):
        self.player_id = player_id # n'infmllue pas sur le tour de jeu etc*.. l'agzent joue quand c'est son tour géré dans environneme t
        self.placement_strategy = placement_strategy or self.random_placement
        self.removal_strategy = removal_strategy or self.random_removal
        self.name = name if name else self.__class__.__name__

    def choose_move(self, env):
        if env.waiting_for_removal:
            removable = env.get_removable_pawns()
            if removable:
                return self.removal_strategy(env, removable)
        else:
            legal_moves = env.get_legal_moves()
            if legal_moves:
                return self.placement_strategy(env, legal_moves)
        return None

    # Stratégies par défaut
    def random_placement(self, env, legal_moves):
        return random.choice(legal_moves)

    def random_removal(self, env, removable):
        return random.choice(removable)

