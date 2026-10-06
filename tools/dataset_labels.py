"""Auditable configuration IDs, molecule mappings, coverage, and geometry labels."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

from .contact_checks import BOND_RANGES, METALS, minimum_image_vectors, validate_geometry
from .utils import ADSORBATES, ADSORBATE_ALIASES, read_structure_file

LABEL_SCHEMA_VERSION = "2.0"


def geometry_hash(structure):
    """Order-independent hash of a decorated cell (six decimal places)."""
    cell = np.asarray(structure["cell"], float)
    frac = np.asarray(structure["positions"], float) @ np.linalg.inv(cell)
    frac = np.round(frac, 6)
    frac[:, :2] %= 1.0
    rows = sorted((str(el), *np.round(f, 6).tolist()) for el, f in zip(structure["atoms"], frac))
    payload = json.dumps([np.round(cell, 6).tolist(), rows], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def molecule_record(species, atom_indices, atoms, positions, molecule_id=None, binding_atom_indices=None):
    """Declare intended connectivity from a template, not relaxed coordinates."""
    species = ADSORBATE_ALIASES.get(species, species)
    ids = list(map(int, atom_indices))
    symbols = [atoms[i] for i in ids]
    template = ADSORBATES.get(species)
    if template is None or Counter(template["atoms"]) != Counter(symbols):
        raise ValueError(f"Molecule {species} does not match the declared atom indices.")
    # Match element occurrences so templates and scenario generators may use
    # different element orders while the saved POSCAR map remains explicit.
    available = list(range(len(ids)))
    order = []
    for element in template["atoms"]:
        local = next(i for i in available if symbols[i] == element)
        available.remove(local)
        order.append(local)
    bonds = []
    reference = np.asarray(template["positions"], float)
    for i in range(len(order)):
        for j in range(i):
            limits = BOND_RANGES.get(tuple(sorted((template["atoms"][i], template["atoms"][j]))))
            if limits and np.linalg.norm(reference[i] - reference[j]) <= limits[1]:
                bonds.append([ids[order[j]], ids[order[i]]])
    angles = []
    adjacency = {i: [] for i in ids}
    for a, b in bonds:
        adjacency[a].append(b); adjacency[b].append(a)
    reference_by_id = {ids[order[i]]: reference[i] for i in range(len(order))}
    for center, neighbors in adjacency.items():
        for j, a in enumerate(neighbors):
            for b in neighbors[j + 1:]:
                va, vb = reference_by_id[a] - reference_by_id[center], reference_by_id[b] - reference_by_id[center]
                angle = float(np.degrees(np.arccos(np.clip(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)), -1, 1))))
                limits = [85., 180.] if species == "CO2" else [max(45., angle - 35.), min(180., angle + 35.)]
                angles.append({"indices": [a, center, b], "range": limits})
    if binding_atom_indices is None:
        binding_atom_indices = ([ids[i] for i, el in enumerate(symbols) if el == "C"] if species == "OCCO"
                                else [ids[order[int(template["binding_atom"])]]])
    return {"molecule_id": molecule_id or f"molecule_{ids[0]}", "species": species,
            "atom_indices": ids, "binding_atom_indices": list(map(int, binding_atom_indices)),
            "formula_counts": dict(Counter(template["atoms"])), "expected_bonds": bonds, "expected_angles": angles}


def remap_metadata(metadata, permutation):
    """Map old indices to element-grouped POSCAR indices without changing IDs."""
    result = deepcopy(metadata)
    lookup = {old: new for new, old in enumerate(permutation)}
    for molecule in result.get("molecules", []):
        for key in ("atom_indices", "binding_atom_indices"):
            molecule[key] = [lookup[i] for i in molecule.get(key, [])]
        molecule["expected_bonds"] = [[lookup[i] for i in pair] for pair in molecule.get("expected_bonds", [])]
        for angle in molecule.get("expected_angles", []):
            angle["indices"] = [lookup[i] for i in angle["indices"]]
    for site in result.get("selected_sites", []):
        site["indices"] = [lookup[i] for i in site.get("indices", [])]
    for key in ("atom_ids", "atom_layers"):
        if key in result:
            result[key] = [result[key][i] for i in permutation]
    # These measurements are recomputed in the on-disk index order.
    result.pop("geometry_qc", None)
    result.pop("observed_geometry", None)
    return result


def annotate_structure(structure, stage="generated"):
    """Attach measured labels. Relaxed transformations require review.

    Geometry labels use a declared distance heuristic: nearest top-face metal
    atoms within 0.25 A of the nearest contact and within 3 A of the binding
    atom. No species assignment is invented for changed connectivity.
    """
    meta = deepcopy(structure.get("metadata", {}))
    atoms = list(structure["atoms"])
    pos, cell = np.asarray(structure["positions"], float), np.asarray(structure["cell"], float)
    molecules = meta.get("molecules", [])
    qc = validate_geometry(atoms, pos, cell, meta.get("pbc", [True, True, False]), molecules)
    measured_hash = geometry_hash(structure)
    meta.setdefault("configuration_id", "cfg_" + measured_hash[:24])
    meta.setdefault("geometry_hash_initial", measured_hash)
    meta["geometry_hash"] = measured_hash
    if "atom_ids" not in meta:
        meta["atom_ids"] = [f"{meta['configuration_id']}:atom_{i:05d}" for i in range(len(atoms))]
    if len(meta["atom_ids"]) != len(atoms):
        raise ValueError("Atom ID mapping does not match the structure.")
    meta["label_schema_version"] = LABEL_SCHEMA_VERSION
    meta["label_stage"] = stage
    meta.setdefault("species_requested", "+".join(sorted(set(m["species"] for m in molecules))) if molecules else meta.get("adsorbate", "clean"))
    meta.setdefault("geometry_requested", meta.get("adsorption_region") or meta.get("adsorption_site"))
    meta.setdefault("distribution_requested", meta.get("distribution"))
    metals = [i for i, el in enumerate(atoms) if el in METALS]
    top = [i for i in metals if pos[i, 2] >= max(pos[j, 2] for j in metals) - 0.6] if metals else []
    meta["atom_layers"] = [-1] * len(atoms)
    layer_z = []
    for i in sorted(metals, key=lambda j: pos[j, 2]):
        if not layer_z or pos[i, 2] - layer_z[-1] > 0.6:
            layer_z.append(float(pos[i, 2]))
        meta["atom_layers"][i] = len(layer_z) - 1
    meta["surface_atom_ids"] = [meta["atom_ids"][i] for i in top]
    observed = []
    occupied = set()
    for molecule, check in zip(molecules, qc["molecules"]):
        contacts, motifs = [], []
        for anchor in molecule.get("binding_atom_indices", []):
            distances = np.linalg.norm(minimum_image_vectors(pos[top] - pos[anchor], cell), axis=1) if top else np.array([])
            near = [top[j] for j, d in enumerate(distances) if d <= 3.0 and d <= distances.min() + 0.25]
            occupied.update(near)
            motif = {0: "unbound", 1: "atop", 2: "bridge", 3: "hollow"}.get(len(near), "multicoordinated")
            composition = "-".join(sorted(set(atoms[i] for i in near))) or "none"
            motifs.append(f"{composition}:{motif}")
            contacts.append({"binding_atom_index": anchor, "metal_indices": near,
                             "metal_elements": [atoms[i] for i in near],
                             "distances": [float(np.linalg.norm(minimum_image_vectors(pos[i] - pos[anchor], cell))) for i in near]})
        intact = check["formula_matches"] and check["connectivity"] == "intact"
        observed.append({"molecule_id": molecule["molecule_id"], "species": molecule["species"] if intact else None,
                         "connectivity": check["connectivity"], "motif": "+".join(motifs), "contacts": contacts})
    clean = not molecules and all(a in METALS for a in atoms)
    integrity = "verified" if (clean or molecules and all(o["species"] for o in observed)) and qc["severity"] != "error" else "review_required"
    meta["label_status"] = integrity
    meta["geometry_qc"] = qc
    meta["observed_geometry"] = observed
    meta["species_observed"] = "+".join(sorted(set(o["species"] for o in observed))) if observed and all(o["species"] for o in observed) else "clean" if clean else None
    meta["geometry_observed"] = "+".join(sorted(set(o["motif"] for o in observed))) if observed else "clean" if clean else None
    counts = dict(Counter(m["species"] for m in molecules))
    basis = meta.get("coverage_basis") or meta.get("coverage_reference", "").removesuffix("_top_sites") or meta.get("binding_element") or "all_surface"
    denominator = meta.get("coverage_denominator") or meta.get("coverage_reference_count")
    denominator = int(denominator) if denominator is not None else sum(basis == "all_surface" or atoms[i] == basis for i in top)
    if denominator < 1:
        raise ValueError(f"No surface sites for coverage basis {basis}.")
    definition = meta.get("coverage_definition", "molecules / reference surface sites")
    numerator = sum(counts.values())
    if "C-equivalent" in definition:
        numerator = sum(m["formula_counts"].get("C", 0) for m in molecules)
    elif meta.get("coverage_site_count") is not None:
        numerator = sum(len(m.get("binding_atom_indices", [])) for m in molecules)
        definition = "binding anchors / reference surface sites"
    known_counts = bool(molecules) or clean
    meta["coverage_labels"] = {"requested": meta.get("coverage_requested", meta.get("coverage_ml")),
        "actual": numerator / denominator if known_counts else None, "numerator": numerator if known_counts else None,
        "denominator": denominator, "basis": basis, "definition": definition,
        "molecule_counts": counts, "occupied_surface_sites": len(occupied),
        "per_species_molecular_coverage": {key: value / denominator for key, value in counts.items()}}
    interface = meta.get("interface", {}) or {}
    anchor_distances = []
    if interface.get("type") == "lateral" and "split_coordinate" in interface:
        axis = 1 if interface.get("split_axis", "y") == "y" else 0
        length = float(np.linalg.norm(cell[axis]))
        split = float(interface["split_coordinate"])
        for molecule in molecules:
            for index in molecule.get("binding_atom_indices", []):
                coordinate = float(pos[index, axis]) % length
                anchor_distances.append(min(coordinate, length - coordinate, abs(coordinate - split), length - abs(coordinate - split)))
    meta["distribution_observed"] = {"mean_interface_distance": float(np.mean(anchor_distances)) if anchor_distances else None,
                                     "binding_atom_interface_distances": anchor_distances, "occupied_surface_sites": sorted(occupied)}
    # Requested choices remain in provenance. ML targets come from measured
    # labels only and are withheld for structures requiring review.
    meta["ml_targets"] = {"adsorbate_identity": meta["species_observed"] if integrity == "verified" else None,
                          "local_geometry": meta["geometry_observed"] if integrity == "verified" else None,
                          "coverage": meta["coverage_labels"]["actual"] if integrity == "verified" else None}
    meta["geometry_label_method"] = "top metal contact <=3A, nearest+0.25A; connectivity screening"
    structure["metadata"] = meta
    return structure


def relaxed_metadata(structure, metadata, initial_structure=None):
    """Preserve decorated group identity across relaxation and refresh labels."""
    if initial_structure is not None and not metadata.get("configuration_id"):
        initial_structure["metadata"] = deepcopy(metadata)
        metadata = annotate_structure(initial_structure)["metadata"]
    if initial_structure is not None and list(initial_structure["atoms"]) != list(structure["atoms"]):
        raise ValueError("Relaxed atom order/composition differs from the saved POSCAR mapping.")
    structure["metadata"] = deepcopy(metadata)
    return annotate_structure(structure, stage="post_relaxation")["metadata"]


def write_dataset_manifest(root):
    """Build a reproducible JSON/CSV manifest for saved structure_info files."""
    import csv
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("structure_info.json")):
        meta = json.loads(path.read_text())
        rows.append({"path": str(path.relative_to(root)), "configuration_id": meta.get("configuration_id"),
                     "parent_structure_id": meta.get("parent_structure_id"), "geometry_hash": meta.get("geometry_hash"),
                     "stage": meta.get("label_stage"), "label_status": meta.get("label_status"),
                     "species_requested": meta.get("species_requested"), "geometry_requested": meta.get("geometry_requested"),
                     "species_observed": meta.get("species_observed"), "geometry_observed": meta.get("geometry_observed"),
                     "coverage": meta.get("coverage_labels"), "ml_targets": meta.get("ml_targets"),
                     "qc_severity": meta.get("geometry_qc", {}).get("severity")})
    (root / "dataset_manifest.json").write_text(json.dumps({"label_schema_version": LABEL_SCHEMA_VERSION, "structures": rows}, indent=2) + "\n")
    if rows:
        with (root / "dataset_manifest.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
            writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, dict) else v for k, v in row.items()} for row in rows)
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit saved structures and write a dataset manifest.")
    parser.add_argument("root")
    args = parser.parse_args()
    for path in sorted(Path(args.root).rglob("structure_info.json")):
        structure = read_structure_file(str(path.with_name("CONTCAR") if path.with_name("CONTCAR").exists() else path.with_name("POSCAR")))
        meta = json.loads(path.read_text())
        stage = "post_relaxation" if path.with_name("CONTCAR").exists() else "generated"
        initial = read_structure_file(str(path.with_name("POSCAR"))) if stage == "post_relaxation" else None
        updated = relaxed_metadata(structure, meta, initial) if initial else annotate_structure({**structure, "metadata": meta}, stage)["metadata"]
        path.write_text(json.dumps(updated, indent=2) + "\n")
    rows = write_dataset_manifest(args.root)
    print(f"Audited {len(rows)} structures")


def sample_label_fields(metadata):
    """Shared ISAAC sample fields for all record generation routes."""
    if not metadata:
        return {}
    keys = ("label_schema_version", "label_status", "label_stage", "geometry_hash", "species_requested",
            "species_observed", "geometry_requested", "geometry_observed", "distribution_requested",
            "distribution_observed", "coverage_labels", "observed_geometry", "geometry_qc", "atom_ids", "atom_layers", "surface_atom_ids")
    return {"configuration_id": metadata.get("configuration_id"), "parent_structure_id": metadata.get("parent_structure_id"),
            "catalyst": metadata.get("catalyst", {}), "adsorbate": metadata.get("adsorbate_metadata", {}),
            "ml_labels": metadata.get("ml_targets", {}), "structure_descriptors": {k: metadata.get(k) for k in keys}}
