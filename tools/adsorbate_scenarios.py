"""Coverage- and distribution-aware adsorbate scenario generation.

Deterministic dataset-oriented structure generation for Cu/Au CO2RR:
coverage control, uniform/interface/interior bias, cross-interface motifs,
boundary avoidance, maximin spacing, and species-specific starting geometries.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import ADSORBATES

DISTRIBUTIONS = {"uniform", "interface_biased", "interior_biased", "cross_interface"}

METAL_C_HEIGHT = {"Cu": 1.85, "Au": 1.98}
CO_BOND = 1.15
CHO_CO = 1.23
CHO_CH = 1.10
COH_CO = 1.33
COH_OH = 0.98
COOH_CARBONYL = 1.26
COOH_COH = 1.33
COOH_OH = 0.99
OCCO_CC = 1.45
OCCO_CO = 1.25


def _frac(position: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return np.asarray(position, float) @ np.linalg.inv(np.asarray(cell, float))


def _margin(position: np.ndarray, cell: np.ndarray) -> float:
    f = _frac(position, cell)
    x, y = float(f[0] % 1.0), float(f[1] % 1.0)
    return min(x, 1.0 - x, y, 1.0 - y)


def _pbc_vector(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> np.ndarray:
    d = np.asarray(b, float) - np.asarray(a, float)
    lx = float(np.linalg.norm(cell[0, :2]))
    ly = float(np.linalg.norm(cell[1, :2]))
    if lx > 1e-12:
        d[0] = (d[0] + 0.5 * lx) % lx - 0.5 * lx
    if ly > 1e-12:
        d[1] = (d[1] + 0.5 * ly) % ly - 0.5 * ly
    return d


def _pbc_xy_distance(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> float:
    return float(np.linalg.norm(_pbc_vector(a, b, cell)[:2]))


def _top_indices(atoms: Sequence[str], positions: np.ndarray, tol: float = 0.5) -> np.ndarray:
    metal = [i for i, a in enumerate(atoms) if a not in {"H", "C", "N", "O"}]
    if not metal:
        metal = list(range(len(atoms)))
    zmax = max(float(positions[i, 2]) for i in metal)
    return np.array([i for i in metal if abs(float(positions[i, 2]) - zmax) < tol], int)


def _second_indices(atoms: Sequence[str], positions: np.ndarray, top: np.ndarray, tol: float = 0.5) -> np.ndarray:
    top_set = set(top.tolist())
    ztop = float(np.mean(positions[top, 2]))
    metal = [i for i, a in enumerate(atoms) if a not in {"H", "C", "N", "O"} and i not in top_set]
    levels = sorted({round(float(positions[i, 2]), 6) for i in metal if positions[i, 2] < ztop - tol})
    if not levels:
        return np.array([], int)
    z2 = levels[-1]
    return np.array([i for i in metal if abs(float(positions[i, 2]) - z2) < tol], int)


def _interface_distance(position: np.ndarray, metadata: Dict[str, Any], cell: np.ndarray) -> float:
    interface = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}
    if interface.get("type") != "lateral":
        return 0.0
    axis = 1 if interface.get("split_axis", "y") == "y" else 0
    split = float(interface.get("split_coordinate", 0.5 * np.linalg.norm(cell[axis, :2])))
    return abs(float(position[axis]) - split)


def _top_sites(structure: Dict[str, Any], element: Optional[str], boundary_margin: float) -> List[Dict[str, Any]]:
    atoms = structure["atoms"]
    pos = np.asarray(structure["positions"], float)
    cell = np.asarray(structure["cell"], float)
    metadata = structure.get("metadata", {})
    sites = []
    for idx in _top_indices(atoms, pos):
        if element and atoms[idx] != element:
            continue
        if _margin(pos[idx], cell) < boundary_margin:
            continue
        sites.append({
            "kind": "top",
            "indices": (int(idx),),
            "element": atoms[idx],
            "composition": atoms[idx],
            "position": pos[idx].copy(),
            "interface_distance": _interface_distance(pos[idx], metadata, cell),
        })
    return sites


def _nearest_neighbor_distance(pos: np.ndarray, cell: np.ndarray) -> float:
    values = []
    for i in range(len(pos)):
        for j in range(i):
            d = _pbc_xy_distance(pos[i], pos[j], cell)
            if d > 0.5:
                values.append(d)
    return min(values) if values else 2.5


def _hollows(structure: Dict[str, Any], element: Optional[str], boundary_margin: float) -> List[Dict[str, Any]]:
    atoms = structure["atoms"]
    pos = np.asarray(structure["positions"], float)
    cell = np.asarray(structure["cell"], float)
    metadata = structure.get("metadata", {})
    top = _top_indices(atoms, pos)
    second = _second_indices(atoms, pos, top)
    nn = _nearest_neighbor_distance(pos[top], cell)
    sites, seen = [], set()
    for ai in range(len(top)):
        for bi in range(ai + 1, len(top)):
            for ci in range(bi + 1, len(top)):
                ids = (int(top[ai]), int(top[bi]), int(top[ci]))
                edges = [
                    _pbc_xy_distance(pos[ids[0]], pos[ids[1]], cell),
                    _pbc_xy_distance(pos[ids[0]], pos[ids[2]], cell),
                    _pbc_xy_distance(pos[ids[1]], pos[ids[2]], cell),
                ]
                if min(edges) < 0.85 * nn or max(edges) > 1.15 * nn:
                    continue
                ref = pos[ids[0]].copy()
                p2 = ref + _pbc_vector(ref, pos[ids[1]], cell)
                p3 = ref + _pbc_vector(ref, pos[ids[2]], cell)
                center = (ref + p2 + p3) / 3.0
                f = _frac(center, cell)
                f[0] %= 1.0
                f[1] %= 1.0
                center = f @ cell
                if _margin(center, cell) < boundary_margin:
                    continue
                if any(_pbc_xy_distance(center, pos[idx], cell) < 0.25 * nn for idx in second):
                    continue
                comp = "".join(sorted(atoms[i] for i in ids))
                if element and element not in comp:
                    continue
                key = tuple(np.round(f[:2], 6))
                if key in seen:
                    continue
                seen.add(key)
                sites.append({
                    "kind": "fcc",
                    "indices": ids,
                    "element": element,
                    "composition": comp,
                    "position": center,
                    "interface_distance": _interface_distance(center, metadata, cell),
                })
    return sites


def _pairs(structure: Dict[str, Any], first: str, second: Optional[str], boundary_margin: float) -> List[Dict[str, Any]]:
    atoms = structure["atoms"]
    pos = np.asarray(structure["positions"], float)
    cell = np.asarray(structure["cell"], float)
    metadata = structure.get("metadata", {})
    top = _top_indices(atoms, pos)
    nn = _nearest_neighbor_distance(pos[top], cell)
    sites = []
    for a_i, a in enumerate(top):
        for b in top[a_i + 1:]:
            if second is None:
                if atoms[a] != first or atoms[b] != first:
                    continue
            else:
                if {atoms[a], atoms[b]} != {first, second}:
                    continue
            d = _pbc_xy_distance(pos[a], pos[b], cell)
            if not (0.85 * nn <= d <= 1.15 * nn):
                continue
            ref = pos[a].copy()
            other = ref + _pbc_vector(ref, pos[b], cell)
            center = 0.5 * (ref + other)
            f = _frac(center, cell)
            f[0] %= 1.0
            f[1] %= 1.0
            center = f @ cell
            if _margin(center, cell) < boundary_margin:
                continue
            sites.append({
                "kind": "pair",
                "indices": (int(a), int(b)),
                "element": f"{atoms[a]}-{atoms[b]}",
                "composition": "".join(sorted([atoms[a], atoms[b]])),
                "position": center,
                "interface_distance": _interface_distance(center, metadata, cell),
            })
    return sites


def _select(candidates: List[Dict[str, Any]], n: int, cell: np.ndarray, distribution: str) -> List[Dict[str, Any]]:
    if n > len(candidates):
        raise ValueError(f"Requested {n} adsorbates but only {len(candidates)} eligible sites exist.")
    if not candidates:
        raise ValueError("No eligible adsorption sites after boundary/site filters.")
    if n == 1:
        if distribution == "interface_biased":
            return [min(candidates, key=lambda x: (x["interface_distance"], -_margin(x["position"], cell)))]
        if distribution == "interior_biased":
            return [max(candidates, key=lambda x: (x["interface_distance"], _margin(x["position"], cell)))]
        return [max(candidates, key=lambda x: _margin(x["position"], cell))]

    best = None
    for seed in candidates[:80]:
        selected = [seed]
        remaining = [x for x in candidates if x is not seed]
        while len(selected) < n:
            def score(site):
                sep = min(_pbc_xy_distance(site["position"], s["position"], cell) for s in selected)
                dint = float(site["interface_distance"])
                bias = -dint if distribution == "interface_biased" else dint if distribution == "interior_biased" else 0.0
                return sep, bias
            chosen = max(remaining, key=score)
            selected.append(chosen)
            remaining.remove(chosen)
        ds = [_pbc_xy_distance(selected[i]["position"], selected[j]["position"], cell)
              for i in range(len(selected)) for j in range(i)]
        minsep = min(ds)
        avgsep = float(np.mean(ds))
        avgint = float(np.mean([x["interface_distance"] for x in selected]))
        obj = (minsep, -avgint, avgsep) if distribution == "interface_biased" else               (minsep, avgint, avgsep) if distribution == "interior_biased" else               (minsep, avgsep)
        if best is None or obj > best[0]:
            best = (obj, selected)
    return best[1]


def _perp(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    v /= np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    p = z - np.dot(z, v) * v
    if np.linalg.norm(p) < 1e-8:
        p = np.array([1.0, 0.0, 0.0])
    return p / np.linalg.norm(p)


def _single_geometry(name: str, site: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    anchor = np.asarray(site["position"], float)
    element = site.get("element") or "Cu"
    if site["kind"] == "fcc" and name == "H":
        return ["H"], np.array([anchor + [0.0, 0.0, 1.05]])
    if site["kind"] == "fcc" and name == "OH":
        O = anchor + np.array([0.0, 0.0, 1.15])
        return ["O", "H"], np.array([O, O + [0.0, 0.0, 0.98]])

    C = anchor + np.array([0.0, 0.0, METAL_C_HEIGHT.get(str(element), 1.90)])
    if name == "CO":
        return ["C", "O"], np.array([C, C + [0.0, 0.0, CO_BOND]])
    if name == "CHO":
        co = np.array([0.22, 0.0, math.sqrt(CHO_CO**2 - 0.22**2)])
        O = C + co
        u = co / np.linalg.norm(co)
        ch = math.cos(math.radians(120.0)) * u + math.sin(math.radians(120.0)) * _perp(u)
        return ["C", "O", "H"], np.array([C, O, C + CHO_CH * ch])
    if name == "COH":
        co = np.array([0.30, 0.0, math.sqrt(COH_CO**2 - 0.30**2)])
        O = C + co
        oc = -co / np.linalg.norm(co)
        oh = math.cos(math.radians(110.0)) * oc + math.sin(math.radians(110.0)) * _perp(oc)
        return ["C", "O", "H"], np.array([C, O, O + COH_OH * oh])
    if name == "COOH":
        # Monodentate Au-like starting motif: carbon binds the metal; retain
        # one carbonyl O and one hydroxyl O-H.
        co = np.array([0.28, 0.0, math.sqrt(COOH_CARBONYL**2 - 0.28**2)])
        Ocar = C + co
        uco = co / np.linalg.norm(co)
        uoh = math.cos(math.radians(125.0)) * uco + math.sin(math.radians(125.0)) * _perp(uco)
        Oh = C + COOH_COH * uoh
        oc = (C - Oh) / np.linalg.norm(C - Oh)
        uh = math.cos(math.radians(108.0)) * oc + math.sin(math.radians(108.0)) * _perp(oc)
        H = Oh + COOH_OH * uh
        return ["C", "O", "O", "H"], np.array([C, Ocar, Oh, H])

    data = ADSORBATES[name]
    local = np.asarray(data["positions"], float).copy()
    binding = int(data["binding_atom"])
    local -= local[binding]
    local += C
    return list(data["atoms"]), local


def _pair_geometry(name: str, site: Dict[str, Any], structure: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
    atoms = structure["atoms"]
    pos = np.asarray(structure["positions"], float)
    cell = np.asarray(structure["cell"], float)
    i, j = site["indices"]
    A = pos[i].copy()
    B = A + _pbc_vector(A, pos[j], cell)
    ea, eb = atoms[i], atoms[j]
    u = B - A
    u[2] = 0.0
    u /= np.linalg.norm(u)
    z = np.array([0.0, 0.0, 1.0])

    if name == "OCCO":
        C1 = A + [0.0, 0.0, METAL_C_HEIGHT.get(ea, 1.90)]
        C2 = B + [0.0, 0.0, METAL_C_HEIGHT.get(eb, 1.90)]
        dz = float(C2[2] - C1[2])
        target_h = math.sqrt(max(OCCO_CC**2 - dz**2, 0.5**2))
        inward = max(0.0, float(np.linalg.norm((C2 - C1)[:2])) - target_h)
        C1 = C1 + 0.5 * inward * u
        C2 = C2 - 0.5 * inward * u
        cc = (C2 - C1) / np.linalg.norm(C2 - C1)
        v1 = -0.5 * cc + 0.866 * z
        v2 = 0.5 * cc + 0.866 * z
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        return ["C", "O", "C", "O"], np.array([C1, C1 + OCCO_CO * v1, C2, C2 + OCCO_CO * v2])

    if name != "COOH":
        raise ValueError(f"No pair geometry for {name}.")

    if {ea, eb} == {"Au", "Cu"}:
        if ea == "Au":
            Au, Cu, axis = A, B, u
        else:
            Au, Cu, axis = B, A, -u
        C = Au + 0.55 * axis + np.array([0.0, 0.0, 2.02])
        direction = Cu - C
        Ocar = C + COOH_CARBONYL * direction / np.linalg.norm(direction)
    else:
        Cu1, Cu2 = A, B
        C = Cu1 - 0.15 * u + np.array([0.0, 0.0, 1.94])
        target = Cu1 + 0.75 * u + np.array([0.0, 0.0, 1.80])
        Ocar = C + COOH_CARBONYL * (target - C) / np.linalg.norm(target - C)

    co = (Ocar - C) / np.linalg.norm(Ocar - C)
    uoh = math.cos(math.radians(125.0)) * co + math.sin(math.radians(125.0)) * _perp(co)
    Oh = C + COOH_COH * uoh
    oc = (C - Oh) / np.linalg.norm(C - Oh)
    uh = math.cos(math.radians(108.0)) * oc - math.sin(math.radians(108.0)) * _perp(oc)
    H = Oh + COOH_OH * uh
    return ["C", "O", "O", "H"], np.array([C, Ocar, Oh, H])


def _preferred(structure: Dict[str, Any], requested: Optional[str]) -> Optional[str]:
    if requested:
        return requested
    metadata = structure.get("metadata", {})
    elems = [metadata.get("element1"), metadata.get("element2"), metadata.get("element")]
    elems = [x for x in elems if x]
    return "Cu" if "Cu" in elems else elems[0] if elems else None


def _denominator(structure: Dict[str, Any], preferred: Optional[str], basis: Optional[str]) -> Tuple[int, str]:
    atoms = structure["atoms"]
    pos = np.asarray(structure["positions"], float)
    top = _top_indices(atoms, pos)
    basis = basis or preferred or "all_surface"
    if basis == "all_surface":
        return len(top), basis
    count = sum(1 for i in top if atoms[i] == basis)
    if not count:
        raise ValueError(f"No top-layer atoms for coverage basis {basis}.")
    return count, basis


def _scenario_specs(structure: Dict[str, Any], adsorbate: str, distribution: str,
                    preferred: Optional[str], all_scenarios: bool) -> List[Dict[str, Any]]:
    interface = structure.get("metadata", {}).get("interface", {})
    is_interface = isinstance(interface, dict) and interface.get("type") == "lateral"
    if not all_scenarios:
        if adsorbate == "OCCO":
            mode = "cross" if distribution == "cross_interface" else "same"
        elif adsorbate == "COOH":
            mode = "cross" if distribution == "cross_interface" else ("top" if preferred == "Au" else "same")
        else:
            mode = None
        return [{"name": distribution, "distribution": distribution, "element": preferred, "pair_mode": mode}]
    if not is_interface:
        if adsorbate in {"OCCO", "COOH"} and preferred != "Au":
            return [{"name": "uniform", "distribution": "uniform", "element": preferred, "pair_mode": "same"}]
        return [{"name": "uniform", "distribution": "uniform", "element": preferred, "pair_mode": "top" if adsorbate == "COOH" else None}]
    if adsorbate == "OCCO":
        return [
            {"name": "CuCu_uniform", "distribution": "uniform", "element": "Cu", "pair_mode": "same"},
            {"name": "CuCu_interface", "distribution": "interface_biased", "element": "Cu", "pair_mode": "same"},
            {"name": "AuCu_interface", "distribution": "cross_interface", "element": None, "pair_mode": "cross"},
        ]
    if adsorbate == "COOH":
        return [
            {"name": "Cu_uniform", "distribution": "uniform", "element": "Cu", "pair_mode": "same"},
            {"name": "Cu_interface", "distribution": "interface_biased", "element": "Cu", "pair_mode": "same"},
            {"name": "Au_uniform", "distribution": "uniform", "element": "Au", "pair_mode": "top"},
            {"name": "AuCu_interface", "distribution": "cross_interface", "element": None, "pair_mode": "cross"},
        ]
    return [
        {"name": "uniform", "distribution": "uniform", "element": preferred, "pair_mode": None},
        {"name": "interface_biased", "distribution": "interface_biased", "element": preferred, "pair_mode": None},
        {"name": "interior_biased", "distribution": "interior_biased", "element": preferred, "pair_mode": None},
    ]


def generate_adsorbate_scenarios(
    structure: Dict[str, Any],
    adsorbate: str,
    coverage: float,
    distribution: str = "uniform",
    preferred_element: Optional[str] = None,
    coverage_basis: Optional[str] = None,
    all_scenarios: bool = False,
    boundary_margin: float = 0.12,
) -> List[Dict[str, Any]]:
    """Generate coverage/distribution-controlled adsorbate structures."""
    distribution = str(distribution or "uniform").lower()
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"Unknown distribution {distribution}.")
    if adsorbate not in ADSORBATES and adsorbate not in {"COH", "COOH"}:
        raise ValueError(f"Unknown adsorbate {adsorbate}.")
    if float(coverage) <= 0:
        raise ValueError("Coverage must be positive.")

    preferred = _preferred(structure, preferred_element)
    denominator, basis = _denominator(structure, preferred, coverage_basis)
    target = max(1, int(math.floor(float(coverage) * denominator + 0.5)))
    n_ads = max(1, int(math.floor(target / 2.0 + 0.5))) if adsorbate == "OCCO" else target
    actual_equiv = 2 * n_ads if adsorbate == "OCCO" else n_ads
    cell = np.asarray(structure["cell"], float)

    outputs = []
    for spec in _scenario_specs(structure, adsorbate, distribution, preferred, all_scenarios):
        mode = spec.get("pair_mode")
        dist = spec["distribution"]
        element = spec.get("element")
        pair_mode = adsorbate in {"OCCO", "COOH"} and mode in {"same", "cross"}

        if pair_mode:
            candidates = _pairs(structure, "Cu", "Au", boundary_margin) if mode == "cross" else                          _pairs(structure, element or "Cu", None, boundary_margin)
            selected = _select(candidates, n_ads, cell, "interface_biased" if dist == "cross_interface" else dist)
        else:
            candidates = _hollows(structure, element, boundary_margin) if adsorbate in {"H", "OH"} else                          _top_sites(structure, element, boundary_margin)
            selected = _select(candidates, n_ads, cell, dist)

        new_atoms = list(structure["atoms"])
        new_pos = [np.asarray(x, float).copy() for x in structure["positions"]]
        groups = []
        site_meta = []

        for site in selected:
            add_atoms, add_pos = _pair_geometry(adsorbate, site, structure) if pair_mode else                                  _single_geometry(adsorbate, site)
            if any(_margin(p, cell) < boundary_margin for p in add_pos):
                raise ValueError(f"{adsorbate} scenario {spec['name']} crosses the periodic boundary margin.")
            new_atoms.extend(add_atoms)
            new_pos.extend(add_pos)
            groups.append(add_pos)
            site_meta.append({
                "kind": site["kind"],
                "indices": list(site["indices"]),
                "composition": site["composition"],
                "interface_distance": float(site["interface_distance"]),
                "fractional_xy": [float(x) for x in (_frac(site["position"], cell)[:2] % 1.0)],
            })

        min_inter = float("inf")
        for i in range(len(groups)):
            for j in range(i):
                for a in groups[i]:
                    for b in groups[j]:
                        d = _pbc_vector(a, b, cell)
                        d[2] = b[2] - a[2]
                        min_inter = min(min_inter, float(np.linalg.norm(d)))
        if min_inter < 1.4:
            raise ValueError(f"{adsorbate} scenario {spec['name']} has a {min_inter:.3f} A inter-adsorbate contact.")

        metadata = dict(structure.get("metadata", {}))
        metadata.update({
            "adsorbate": adsorbate,
            "coverage_requested": float(coverage),
            "coverage_actual": actual_equiv / float(denominator),
            "coverage_basis": basis,
            "coverage_denominator": denominator,
            "coverage_definition": "CO-equivalent C count / denominator" if adsorbate == "OCCO" else "adsorbate count / denominator",
            "n_adsorbates": n_ads,
            "distribution": dist,
            "scenario_name": spec["name"],
            "boundary_margin_fractional": float(boundary_margin),
            "selected_sites": site_meta,
            "min_inter_adsorbate_distance": None if math.isinf(min_inter) else min_inter,
        })
        outputs.append({
            "atoms": new_atoms,
            "positions": np.asarray(new_pos, float),
            "cell": cell.copy(),
            "metadata": metadata,
        })

    return outputs
