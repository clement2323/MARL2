import random
from environnement.marelle_layout import marelle_layout

def generate_symmetry_mappings():
    layout = marelle_layout()
    coords_to_node = {pos: node for node, pos in layout.items()}

    def transform_coords(pos, sym):
        x, y = pos
        if sym == "identity": return (x, y)
        if sym == "rot90": return (y, -x)
        if sym == "rot180": return (-x, -y)
        if sym == "rot270": return (-y, x)
        if sym == "flip_h": return (-x, y)
        if sym == "flip_v": return (x, -y)
        if sym == "flip_d1": return (y, x)
        if sym == "flip_d2": return (-y, -x)

    mappings = {}
    for sym in ["identity","rot90","rot180","rot270","flip_h","flip_v","flip_d1","flip_d2"]:
        perm = []
        for node in range(24):
            new_coords = transform_coords(layout[node], sym)
            perm.append(coords_to_node[new_coords])
        mappings[sym] = perm
    return mappings


# Définition des permutations de nœuds pour chaque symétrie
# Chaque liste donne, pour un index original, le nouvel index après transformation
# Ces mappings doivent être calculés une fois à partir de ton layout (marelle_layout)
SYMMETRY_MAPPINGS = generate_symmetry_mappings()

def apply_symmetry(state_vec, symmetry_name):
    """Applique la symétrie à un vecteur d'état (taille 24)."""
    mapping = SYMMETRY_MAPPINGS[symmetry_name]
    return [state_vec[i] for i in mapping]

def invert_action(action_idx, symmetry_name):
    """Convertit un index d'action transformé vers l'index original."""
    mapping = SYMMETRY_MAPPINGS[symmetry_name]
    inverse_mapping = {new: orig for orig, new in enumerate(mapping)}
    return inverse_mapping[action_idx]

def random_symmetry(state_vec, action_idx=None):
    """Applique une symétrie aléatoire. Retourne (nouvel état, nouvelle action, sym)"""
    sym = random.choice(list(SYMMETRY_MAPPINGS.keys()))
    new_state = apply_symmetry(state_vec, sym)
    if action_idx is not None:
        new_action = SYMMETRY_MAPPINGS[sym][action_idx]
        return new_state, new_action, sym
    return new_state, None, sym

def augment_with_symmetries(state_vec, action_idx=None):
    """Retourne la liste [(state_sym, action_sym, sym_name), ...] pour toutes les symétries."""
    augmented = []
    for sym_name, mapping in SYMMETRY_MAPPINGS.items():
        state_sym = [state_vec[i] for i in mapping]
        if action_idx is not None:
            action_sym = mapping[action_idx]
        else:
            action_sym = None
        augmented.append((state_sym, action_sym, sym_name))
    return augmented
