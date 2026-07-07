from __future__ import annotations

import numpy as np

METALS = {"Cu", "Au", "Ni", "Ag", "Pt", "Pd", "Ir", "Rh", "Al", "Fe", "Co", "Zn", "Ti", "V", "Cr", "Mn", "Mo", "W"}


def pair_cutoffs(el1: str, el2: str):
    pair = tuple(sorted([str(el1), str(el2)]))
    has_metal = el1 in METALS or el2 in METALS
    if el1 in METALS and el2 in METALS:
        return 1.90, 2.20, "metal-metal contact"
    if has_metal:
        nonmetal = el2 if el1 in METALS else el1
        if nonmetal == "H":
            return 1.20, 1.50, "metal-H contact"
        if nonmetal in {"C", "O", "N"}:
            return 1.55, 1.75, f"metal-{nonmetal} contact"
        return 1.45, 1.70, "metal-adsorbate contact"
    specific = {
        ("C", "O"): (0.95, 1.05, "C-O bond/contact"),
        ("C", "H"): (0.75, 0.90, "C-H bond/contact"),
        ("H", "O"): (0.65, 0.80, "O-H bond/contact"),
        ("C", "C"): (1.00, 1.20, "C-C bond/contact"),
        ("O", "O"): (1.00, 1.15, "O-O contact"),
        ("H", "H"): (0.45, 0.65, "H-H contact"),
    }
    return specific.get(pair, (0.80, 1.00, f"{el1}-{el2} contact"))


def minimum_pair(atoms, positions):
    pos = np.asarray(positions, dtype=float)
    dmin = float("inf")
    pair = None
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if d < dmin:
                dmin = d
                pair = (i, j)
    if pair is None:
        return {"severity": "info", "message": "No atom pairs found for distance check."}
    i, j = pair
    el1 = atoms[i] if i < len(atoms) else "?"
    el2 = atoms[j] if j < len(atoms) else "?"
    severe, close, label = pair_cutoffs(el1, el2)
    msg = (
        f"Minimum non-PBC atom distance: {dmin:.3f} Å between atoms {pair} "
        f"({el1}-{el2}, {label}; severe < {severe:.2f} Å, close < {close:.2f} Å)."
    )
    if dmin < severe:
        return {"severity": "error", "message": "Possible severe overlap: " + msg, "distance": dmin, "pair": pair}
    if dmin < close:
        return {"severity": "warning", "message": "Possible close contact: " + msg, "distance": dmin, "pair": pair}
    return {"severity": "info", "message": msg, "distance": dmin, "pair": pair}
