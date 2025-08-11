import networkx as nx
import random
from environnement.marelle_graph import create_marelle_graph

class MarelleEnv:
    def __init__(self):
        self.G = create_marelle_graph()
        nx.set_node_attributes(self.G, 0, "state")  # 0 = vide, 1 = joueur 1, -1 = joueur 2
        self.current_player = 1
        self.moves = []
        self.pawns_to_place = {1: 9, -1: 9}
        self.mills = self._generate_mills()
        self.waiting_for_removal = False
        self.removable_pawns = []

    def reset(self):
        nx.set_node_attributes(self.G, 0, "state")
        self.current_player = 1
        self.moves = []
        self.pawns_to_place = {1: 9, -1: 9}
        self.waiting_for_removal = False
        self.removable_pawns = []

    def _generate_mills(self):
        return [
            # extérieur horizontal
            [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0],
            # moyen horizontal
            [8, 9, 10], [10, 11, 12], [12, 13, 14], [14, 15, 8],
            # intérieur horizontal
            [16, 17, 18], [18, 19, 20], [20, 21, 22], [22, 23, 16],
            # verticales
            [1, 9, 17], [3, 11, 19], [5, 13, 21], [7, 15, 23]
        ]

    def _forms_mill(self, node, player):
        return any(
            node in mill and all(self.G.nodes[n]["state"] == player for n in mill)
            for mill in self.mills
        )

    def _get_removable_opponent_pawns(self, opponent):
        """Retourne les pions adverses à retirer (règle officielle incluse)."""
        opponent_pawns = [n for n in self.G.nodes if self.G.nodes[n]["state"] == opponent]

        # Pions hors moulins
        non_mill_pawns = [
            n for n in opponent_pawns
            if not any(
                all(self.G.nodes[x]["state"] == opponent for x in mill) and n in mill
                for mill in self.mills
            )
        ]

        # Si tous les pions sont dans des moulins, on les autorise tous
        return non_mill_pawns if non_mill_pawns else opponent_pawns

    def play_move(self, node):
        """Pose un pion et détecte si un retrait est nécessaire."""
        if self.waiting_for_removal:
            raise ValueError("Retrait de pion en attente.")

        if self.G.nodes[node]["state"] != 0:
            raise ValueError("Position déjà occupée.")

        self.G.nodes[node]["state"] = self.current_player
        self.moves.append((node, self.current_player))
        self.pawns_to_place[self.current_player] -= 1

        if self._forms_mill(node, self.current_player):
            self.removable_pawns = self._get_removable_opponent_pawns(-self.current_player)
            self.waiting_for_removal = True
            return True

        self.current_player *= -1
        return False

    def remove_pawn(self, node):
        """Retire un pion adverse après un moulin."""
        if not self.waiting_for_removal:
            raise ValueError("Aucun retrait en attente.")
        if node not in self.removable_pawns:
            raise ValueError("Ce pion ne peut pas être retiré.")

        self.G.nodes[node]["state"] = 0
        self.waiting_for_removal = False
        self.removable_pawns = []
        self.current_player *= -1
        return True

    def play_move_auto(self, node, auto_remove=True):
        """Version auto pour entraînement."""
        needs_removal = self.play_move(node)
        if needs_removal and auto_remove:
            choice = random.choice(self.removable_pawns)
            self.remove_pawn(choice)

    def get_removable_pawns(self):
        return self.removable_pawns if self.waiting_for_removal else []

    def get_legal_moves(self):
        return [n for n, data in self.G.nodes(data=True) if data["state"] == 0]

    def is_phase1_over(self):
        return self.pawns_to_place[1] == 0 and self.pawns_to_place[-1] == 0

    def get_winner(self):
        """
        Détermine le gagnant à tout moment.
        Retourne :
            1  -> joueur 1 gagne
            -1 -> joueur 2 gagne
            0  -> égalité
        """
        p1_pawns = sum(1 for _, data in self.G.nodes(data=True) if data["state"] == 1)
        p2_pawns = sum(1 for _, data in self.G.nodes(data=True) if data["state"] == -1)

        if p1_pawns > p2_pawns:
            return 1
        elif p2_pawns > p1_pawns:
            return -1
        else:
            return 0
