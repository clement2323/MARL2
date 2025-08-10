def marelle_layout():
    return {
        # Carré extérieur (nœuds 0 à 7) - plus espacé
        0: (-4, 4),   # haut-gauche
        1: (0, 4),    # haut-milieu
        2: (4, 4),    # haut-droite
        3: (4, 0),    # droite-milieu haut
        4: (4, -4),   # bas-droite
        5: (0, -4),   # bas-milieu
        6: (-4, -4),  # bas-gauche
        7: (-4, 0),   # gauche-milieu haut

        # Carré moyen (nœuds 8 à 15) - espacement moyen
        8: (-2.5, 2.5),   # haut-gauche
        9: (0, 2.5),      # haut-milieu
        10: (2.5, 2.5),   # haut-droite
        11: (2.5, 0),     # droite-milieu
        12: (2.5, -2.5),  # bas-droite
        13: (0, -2.5),    # bas-milieu
        14: (-2.5, -2.5), # bas-gauche
        15: (-2.5, 0),    # gauche-milieu

        # Carré intérieur (nœuds 16 à 23) - plus compact
        16: (-1, 1),      # haut-gauche
        17: (0, 1),       # haut-milieu
        18: (1, 1),       # haut-droite
        19: (1, 0),       # droite-milieu
        20: (1, -1),      # bas-droite
        21: (0, -1),      # bas-milieu
        22: (-1, -1),     # bas-gauche
        23: (-1, 0)       # gauche-milieu
    }
