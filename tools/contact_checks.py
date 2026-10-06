"""Periodic contact and molecule checks. Distances are in Angstrom."""
from __future__ import annotations

from itertools import product
import numpy as np

METALS = {"Cu", "Au", "Ni", "Ag", "Pt", "Pd", "Ir", "Rh", "Al", "Fe", "Co", "Zn", "Ti", "V", "Cr", "Mn", "Mo", "W", "Ru"}
BOND_RANGES = {
    ("C", "C"): (1.10, 1.85), ("C", "O"): (1.00, 1.65),
    ("C", "H"): (0.85, 1.25), ("H", "O"): (0.75, 1.20),
    ("H", "H"): (0.60, 0.90), ("C", "N"): (1.10, 1.65),
}


def pair_cutoffs(el1: str, el2: str):
    pair = tuple(sorted([str(el1), str(el2)]))
    if el1 in METALS and el2 in METALS:
        return 1.90, 2.20, "metal-metal contact"
    if el1 in METALS or el2 in METALS:
        other = el2 if el1 in METALS else el1
        return (1.20, 1.50, "metal-H contact") if other == "H" else (1.55, 1.75, f"metal-{other} contact")
    return BOND_RANGES.get(pair, (0.80, 1.00))[:1] + (1.40, f"{el1}-{el2} contact")


def minimum_image_vectors(vectors, cell=None, pbc=(True, True, False)):
    """Exact closest images, including skew cells and partial periodicity.

    A singular-value bound limits the integer image search; rounding fractional
    coordinates alone is insufficient for skew cells. Cell vectors are rows.
    """
    v = np.asarray(vectors, float)
    shape = v.shape
    v = v.reshape(-1, 3)
    periodic = np.broadcast_to(np.asarray(pbc, bool), (3,))
    if cell is None or not periodic.any() or not len(v):
        return v.reshape(shape).copy()
    lattice = np.asarray(cell, float)[periodic]
    if not np.all(np.isfinite(lattice)):
        raise ValueError("Periodic lattice vectors must be finite.")
    gram = lattice @ lattice.T
    lengths2 = np.diag(gram)
    if np.min(lengths2) > 1e-20 and np.allclose(gram, np.diag(lengths2), atol=1e-12, rtol=1e-12):
        frac = (v @ lattice.T) / lengths2
        return (v - np.rint(frac) @ lattice).reshape(shape)
    singular = np.linalg.svd(lattice, compute_uv=False)
    if not np.all(np.isfinite(lattice)) or singular[-1] < 1e-10:
        raise ValueError("Periodic lattice vectors must be finite and independent.")
    frac = v @ np.linalg.pinv(lattice)
    nearest = np.rint(frac)
    best = v - nearest @ lattice
    best2 = np.sum(best * best, axis=1)
    # The nonperiodic perpendicular component is constant across images and
    # must not inflate the search radius for slabs with a large vacuum.
    projected = (frac - nearest) @ lattice
    radius = int(np.ceil(np.max(np.linalg.norm(projected, axis=1)) / singular[-1] + 0.5))
    for offset in product(range(-radius, radius + 1), repeat=len(lattice)):
        candidate = v - (nearest + offset) @ lattice
        d2 = np.sum(candidate * candidate, axis=1)
        mask = d2 < best2
        best[mask], best2[mask] = candidate[mask], d2[mask]
    return best.reshape(shape)


def _shortest_image(cell, pbc):
    basis = np.asarray(cell, float)[np.broadcast_to(np.asarray(pbc, bool), (3,))]
    if not len(basis):
        return None
    lower = np.linalg.svd(basis, compute_uv=False)[-1]
    if lower < 1e-10:
        raise ValueError("Periodic lattice vectors must be independent.")
    upper = min(np.linalg.norm(basis, axis=1))
    radius = int(np.ceil(upper / lower))
    return min(float(np.linalg.norm(np.asarray(n) @ basis))
               for n in product(range(-radius, radius + 1), repeat=len(basis)) if any(n))


def validate_geometry(atoms, positions, cell=None, pbc=(True, True, False), molecules=()):
    """Check every pair, self image, declared bond, and adsorption face.

    Molecules carry global ``atom_indices`` and optional ``expected_bonds`` as
    global pairs. Bond ranges are screening tolerances, not stability criteria.
    Missing molecule mappings are reported as unknown, never inferred from a
    requested species label after relaxation.
    """
    atoms = list(atoms)
    pos = np.asarray(positions, float)
    if pos.shape != (len(atoms), 3) or not np.all(np.isfinite(pos)):
        raise ValueError("Expected one finite Cartesian position per atom.")
    owner, bonds, molecule_checks = {}, set(), []
    for k, molecule in enumerate(molecules or ()):
        ids = [int(i) for i in molecule.get("atom_indices", [])]
        if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= len(atoms) for i in ids):
            raise ValueError("Molecule indices must be unique and within the structure.")
        for i in ids:
            if i in owner:
                raise ValueError("An atom cannot belong to two molecules.")
            owner[i] = k
        for pair in molecule.get("expected_bonds", []):
            if len(pair) != 2 or any(int(i) not in ids for i in pair):
                raise ValueError("Expected bonds must reference atoms in their molecule.")
            bonds.add(tuple(sorted(map(int, pair))))
        expected = molecule.get("formula_counts")
        observed = {el: sum(atoms[i] == el for i in ids) for el in set(atoms[i] for i in ids)}
        molecule_checks.append({"molecule_id": molecule.get("molecule_id", str(k)),
                                "species_requested": molecule.get("species"), "formula_counts": observed,
                                "formula_matches": expected is None or observed == expected,
                                "connectivity": "unchecked" if len(ids) > 1 and not molecule.get("expected_bonds") else "intact"})
    issues, bond_checks = [], []
    periodic_basis = np.asarray(cell, float)[np.broadcast_to(np.asarray(pbc, bool), (3,))] if cell is not None else np.empty((0, 3))
    periodic_inverse = np.linalg.pinv(periodic_basis) if len(periodic_basis) else None
    periodic_lower = np.linalg.svd(periodic_basis, compute_uv=False)[-1] if len(periodic_basis) else None
    ii, jj = np.triu_indices(len(pos), 1)
    distances = np.linalg.norm(minimum_image_vectors(pos[jj] - pos[ii], cell, pbc), axis=1)
    for i, j, d in zip(ii.tolist(), jj.tolist(), distances.tolist()):
        pair = (i, j)
        if pair in bonds:
            lo, hi = BOND_RANGES.get(tuple(sorted((atoms[i], atoms[j]))), (0.60, 2.00))
            intact = lo <= d <= hi
            bond_checks.append({"pair": list(pair), "distance": d, "range": [lo, hi], "intact": intact})
            if not intact:
                molecule_checks[owner[i]]["connectivity"] = "changed"
                issues.append({"severity": "error", "kind": "bond_geometry", "pair": list(pair), "distance": d})
            if len(periodic_basis):
                delta = pos[j] - pos[i]
                chosen_vector = minimum_image_vectors(delta, cell, pbc)
                chosen_image = np.rint((delta - chosen_vector) @ periodic_inverse).astype(int)
                center = np.rint(delta @ periodic_inverse).astype(int)
                radius = int(np.ceil(hi / periodic_lower + 0.5))
                for offset in product(range(-radius, radius + 1), repeat=len(periodic_basis)):
                    candidate_image = center + offset
                    if np.array_equal(candidate_image, chosen_image):
                        continue
                    image_distance = float(np.linalg.norm(delta - candidate_image @ periodic_basis))
                    if image_distance <= hi:
                        molecule_checks[owner[i]]["connectivity"] = "changed"
                        issues.append({"severity": "error", "kind": "unexpected_periodic_bond", "pair": list(pair),
                                       "distance": image_distance, "image": candidate_image.tolist()})
            continue
        severe, close, _ = pair_cutoffs(atoms[i], atoms[j])
        if i in owner and j in owner and owner[i] != owner[j]:
            # A short contact between separate molecules cannot be excused as
            # a normal C-H or O-H covalent bond.
            severe, close = max(severe, 1.20), max(close, 1.50)
            limits = BOND_RANGES.get(tuple(sorted((atoms[i], atoms[j]))))
            if limits and d <= limits[1]:
                for atom in (i, j):
                    molecule_checks[owner[atom]]["connectivity"] = "changed"
                issues.append({"severity": "error", "kind": "possible_intermolecular_bond", "pair": list(pair), "distance": d})
        elif i in owner and j in owner and owner[i] == owner[j]:
            limits = BOND_RANGES.get(tuple(sorted((atoms[i], atoms[j]))))
            if limits and d <= limits[1]:
                molecule_checks[owner[i]]["connectivity"] = "changed"
                issues.append({"severity": "error", "kind": "unexpected_bond", "pair": list(pair), "distance": d})
        if d < close:
            issues.append({"severity": "error" if d < severe else "warning",
                           "kind": "contact", "pair": list(pair), "elements": [atoms[i], atoms[j]], "distance": d})
    angle_checks = []
    for molecule in molecules or ():
        for angle in molecule.get("expected_angles", []):
            a, center, b = angle["indices"]
            if any(i not in molecule["atom_indices"] for i in (a, center, b)):
                raise ValueError("Expected angles must reference atoms in their molecule.")
            va, vb = minimum_image_vectors(np.array([pos[a] - pos[center], pos[b] - pos[center]]), cell, pbc)
            denominator = np.linalg.norm(va) * np.linalg.norm(vb)
            value = float(np.degrees(np.arccos(np.clip(np.dot(va, vb) / denominator, -1, 1)))) if denominator else 0.
            lo, hi = angle["range"]
            intact = lo <= value <= hi
            angle_checks.append({"indices": [a, center, b], "angle_degrees": value, "range": [lo, hi], "valid": intact})
            if not intact:
                issues.append({"severity": "error", "kind": "angle_geometry", "indices": [a, center, b], "angle_degrees": value})
    for check in molecule_checks:
        if not check["formula_matches"]:
            issues.append({"severity": "error", "kind": "molecule_formula", "molecule_id": check["molecule_id"]})
    metals = [i for i, el in enumerate(atoms) if el in METALS]
    if metals:
        surface_z = max(pos[i, 2] for i in metals)
        for i in owner:
            if pos[i, 2] < surface_z - 0.5:
                issues.append({"severity": "warning", "kind": "below_selected_face", "atom_index": i})
    image_distance = _shortest_image(cell, pbc) if cell is not None else None
    if image_distance is not None:
        for i, el in enumerate(atoms):
            severe, close, _ = pair_cutoffs(el, el)
            if image_distance < close:
                issues.append({"severity": "error" if image_distance < severe else "warning", "kind": "self_image", "pair": [i, i], "distance": image_distance})
    severity = "error" if any(i["severity"] == "error" for i in issues) else "warning" if issues else "info"
    idx = int(np.argmin(distances)) if len(distances) else None
    worst = next((i for i in issues if i["severity"] == severity), None)
    pair = worst.get("pair") if worst else ([int(ii[idx]), int(jj[idx])] if idx is not None else None)
    distance = worst.get("distance") if worst else (float(distances[idx]) if idx is not None else None)
    return {"severity": severity, "message": f"Geometry check: {len(issues)} flagged contacts/bonds across {len(distances)} atom pairs.",
            "pair": pair, "distance": distance, "minimum_distance": float(distances.min()) if len(distances) else None,
            "periodic_axes": np.broadcast_to(np.asarray(pbc, bool), (3,)).tolist() if cell is not None else [False] * 3,
            "issues": issues, "bonds": bond_checks, "angles": angle_checks, "molecules": molecule_checks,
            "molecule_mapping": "declared" if molecules else "unavailable"}


def minimum_pair(atoms, positions, cell=None, pbc=(True, True, False), molecules=()):
    """Compatibility wrapper: severity reflects all pairs, not the shortest bond."""
    return validate_geometry(atoms, positions, cell, pbc, molecules)
