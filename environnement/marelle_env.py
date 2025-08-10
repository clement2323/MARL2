# marelle_env.py
import networkx as nx
import random
from environnement.marelle_graph import create_marelle_graph

class MarelleEnv:
    def __init__(self):
        self.G = create_marelle_graph()
        nx.set_node_attributes(self.G, 0, "state")  # 0 = vide, 1 = modèle, -1 = humain
        self.current_player = 1
        self.moves = []
        self.pawns_to_place = {1: 9, -1: 9}  # jetons à poser
        self.mills = self._generate_mills()  # listes de moulins
        self.waiting_for_removal = False
        self.removable_pawns = []

    def reset(self):
        nx.set_node_attributes(self.G, 0, "state")
        self.current_player = 1
        self.moves = []
        self.pawns_to_place = {1: 9, -1: 9}
        self.waiting_for_removal = False# marelle_env.py
import networkx as nx
from environnement.marelle_graph import create_marelle_graph, marelle_edges

class MarelleEnv:
    def __init__(self):
        self.G = create_marelle_graph()
        nx.set_node_attributes(self.G, 0, "state")  # 0 = vide, 1 = modèle, -1 = humain
        self.current_player = 1
        self.moves = []
        self.pawns_to_place = {1: 9, -1: 9}  # jetons à poser
        self.mills = self._generate_mills()  # listes de moulins
        self.waiting_for_removal = False  # Nouveau flag
        self.removable_pawns = []  # Liste des pions à retirer

    def reset(self):
        nx.set_node_attributes(self.G, 0, "state")
        self.current_player = 1
        self.moves = []
        self.pawns_to_place = {1: 9, -1: 9}

    def _generate_mills(self):
        """Retourne la liste des triplets de positions formant un moulin."""
        # Chaque moulin est une liste de 3 nœuds
        mills = [
            # extérieur horizontal
            [0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0],
            # moyen horizontal
            [8, 9, 10], [10, 11, 12], [12, 13, 14], [14, 15, 8],
            # intérieur horizontal
            [16, 17, 18], [18, 19, 20], [20, 21, 22], [22, 23, 16],
            # verticales
            [1, 9, 17], [3, 11, 19], [5, 13, 21], [7, 15, 23],
            [0, 8, 16], [2, 10, 18], [4, 12, 20], [6, 14, 22]
        ]
        return mills

    def _forms_mill(self, node, player):
        """Vérifie si la pose sur 'node' forme un moulin pour 'player'."""
        for mill in self.mills:
            if node in mill and all(self.G.nodes[n]["state"] == player for n in mill):
                return True
        return False

    def _get_removable_opponent_pawns(self, opponent):
        """Retourne les pions adverses pouvant être retirés (pas dans un moulin sauf si tous le sont)."""
        opponent_pawns = [n for n in self.G.nodes() if self.G.nodes[n]["state"] == opponent]
        # pions qui ne sont pas dans un moulin
        non_mill_pawns = [
            n for n in opponent_pawns
            if not any(all(self.G.nodes[x]["state"] == opponent for x in mill) and n in mill
            for mill in self.mills)
        ]
        return non_mill_pawns if non_mill_pawns else opponent_pawns  # si tous dans des moulins

    def play_move(self, node):
        """Jouer un coup (pose de pion dans la phase 1)."""
        if self.G.nodes[node]["state"] != 0:
            raise ValueError("Position déjà occupée")

        # Poser pion
        self.G.nodes[node]["state"] = self.current_player
        self.moves.append((node, self.current_player))
        self.pawns_to_place[self.current_player] -= 1

        # Vérifier moulin
        if self._forms_mill(node, self.current_player):
            removable = self._get_removable_opponent_pawns(-self.current_player)
            if removable:
                print(f"⚡ Joueur {self.current_player} a formé un moulin !")
                print(f"Cliquez sur un pion adverse à retirer")
                self.waiting_for_removal = True
                self.removable_pawns = removable
                return  # Ne pas changer de joueur encore
        
        # Changer de joueur seulement si pas en attente de suppression
        if not self.waiting_for_removal:
            self.current_player *= -1

    def play_move_auto(self, node, auto_remove=True):
        """Jouer un coup automatiquement (pour l'entraînement)"""
        if self.G.nodes[node]["state"] != 0:
            raise ValueError("Position déjà occupée")

        # Poser pion
        self.G.nodes[node]["state"] = self.current_player
        self.moves.append((node, self.current_player))
        self.pawns_to_place[self.current_player] -= 1

        # Vérifier moulin
        if self._forms_mill(node, self.current_player):
            removable = self._get_removable_opponent_pawns(-self.current_player)
            if removable and auto_remove:
                # Suppression automatique (choix aléatoire)
                import random
                choice = random.choice(removable)
                self.G.nodes[choice]["state"] = 0
                # Pas de print ni d'input ici
        
        # Changer de joueur
        self.current_player *= -1

    def remove_pawn(self, node):
        """Supprimer un pion adverse (mode interactif)."""
        if not self.waiting_for_removal:
            return False
        
        if node in self.removable_pawns:
            self.G.nodes[node]["state"] = 0
            self.waiting_for_removal = False
            self.removable_pawns = []
            self.current_player *= -1  # Maintenant on change de joueur
            return True
        return False

    def get_removable_pawns(self):
        """Retourne la liste des pions pouvant être retirés."""
        return self.removable_pawns if self.waiting_for_removal else []

    def get_legal_moves(self):
        return [n for n, data in self.G.nodes(data=True) if data["state"] == 0]

    def is_phase1_over(self):
        """Vrai si les deux joueurs n'ont plus de pion à poser."""
        return self.pawns_to_place[1] == 0 and self.pawns_to_place[-1] == 0


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
            [1, 9, 17], [3, 11, 19], [5, 13, 21], [7, 15, 23],
            [0, 8, 16], [2, 10, 18], [4, 12, 20], [6, 14, 22]
        ]

    def _forms_mill(self, node, player):
        return any(
            node in mill and all(self.G.nodes[n]["state"] == player for n in mill)
            for mill in self.mills
        )

    def _get_removable_opponent_pawns(self, opponent):
        opponent_pawns = [n for n in self.G.nodes() if self.G.nodes[n]["state"] == opponent]
        non_mill_pawns = [
            n for n in opponent_pawns
            if not any(all(self.G.nodes[x]["state"] == opponent for x in mill) and n in mill
                       for mill in self.mills)
        ]
        return non_mill_pawns if non_mill_pawns else opponent_pawns

    def play_move(self, node):
        """Pose un pion et retourne True si un retrait est nécessaire."""
        if self.waiting_for_removal:
            raise ValueError("Un retrait de pion est en attente, appelez remove_pawn() avant de continuer.")

        if self.G.nodes[node]["state"] != 0:
            raise ValueError("Position déjà occupée")

        # Poser pion
        self.G.nodes[node]["state"] = self.current_player
        self.moves.append((node, self.current_player))
        self.pawns_to_place[self.current_player] -= 1

        # Vérifier moulin
        if self._forms_mill(node, self.current_player):
            self.removable_pawns = self._get_removable_opponent_pawns(-self.current_player)
            self.waiting_for_removal = True
            return True  # Un retrait doit être effectué

        # Sinon on passe au joueur suivant
        self.current_player *= -1
        return False

    def remove_pawn(self, node):
        """Retire un pion adverse si un retrait est en attente."""
        if not self.waiting_for_removal:
            raise ValueError("Aucun retrait en attente.")

        if node not in self.removable_pawns:
            raise ValueError("Ce pion ne peut pas être retiré.")

        # Retirer le pion
        self.G.nodes[node]["state"] = 0
        self.waiting_for_removal = False
        self.removable_pawns = []
        self.current_player *= -1
        return True

    def play_move_auto(self, node, auto_remove=True):
        """Pose un pion automatiquement et retire si nécessaire."""
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

    def is_full(self):
        return all(data["state"] != 0 for _, data in self.G.nodes(data=True))
