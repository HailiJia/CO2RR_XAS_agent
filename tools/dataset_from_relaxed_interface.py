"""Create a coverage-resolved Cu K-edge showcase set from a relaxed parent.

Coverage is reported as adsorbate binding sites divided by the number of
relaxed top-layer sites of the binding element. For the attached parent,
N(Cu_top)=36 and N(Au_top)=32. This convention is preferable to dividing by
all 68 top-layer atoms when the target is Cu K-edge local environments.
"""
from pathlib import Path
import argparse, csv, json, shutil
import numpy as np
from .relaxed_interface import load_relaxed_interface
from .utils import ADSORBATES
ADSORBATES.setdefault("O", {"atoms": ["O"], "positions": np.array([[0., 0., 0.]]), "binding_atom": 0})

COVERAGE_GRID = (0.25, 0.50, 0.75)

def _write_poscar(path, atoms, positions, cell):
    elements = []
    for atom in atoms:
        if atom not in elements:
            elements.append(atom)
    lines = ["Cu/Au relaxed-parent coverage showcase", "1.0"]
    lines += [" %.12f %.12f %.12f" % tuple(v) for v in cell]
    lines += [" ".join(elements), " ".join(str(atoms.count(e)) for e in elements)]
    lines += ["Selective dynamics", "Cartesian"]
    lines += [" %.12f %.12f %.12f T T T" % tuple(p) for p in positions]
    Path(path).write_text("\n".join(lines) + "\n")

def _substrate_top_indices(structure, element):
    positions = np.asarray(structure["positions"], dtype=float)
    element_indices = np.asarray([i for i, a in enumerate(structure["atoms"]) if a == element], dtype=int)
    zmax = positions[element_indices, 2].max()
    return element_indices[positions[element_indices, 2] >= zmax - 0.6]

def _spread_sites(structure, element, count, preferred_y=None, exclude=()):
    """Select well separated top-layer sites, starting near a requested row."""
    positions = np.asarray(structure["positions"], dtype=float)
    pool = [int(i) for i in _substrate_top_indices(structure, element) if int(i) not in set(exclude)]
    if count > len(pool):
        raise ValueError(f"{count} {element} sites requested, only {len(pool)} top sites available")
    if preferred_y is not None:
        first = min(pool, key=lambda i: abs(float(positions[i, 1]) - preferred_y))
    else:
        first = pool[0]
    chosen = [first]
    while len(chosen) < count:
        candidates = [i for i in pool if i not in chosen]
        # Maximin selection in the lateral plane; this avoids placing all
        # high-coverage adsorbates in one row of the stripe.
        nxt = max(candidates, key=lambda i: min(np.linalg.norm(positions[i, :2] - positions[j, :2]) for j in chosen))
        chosen.append(nxt)
    return chosen

def _add_species(structure, adsorbate, site_ids, coverage, region, binding_element):
    atoms = list(structure["atoms"])
    positions = np.asarray(structure["positions"], dtype=float).copy()
    cell = np.asarray(structure["cell"], dtype=float)
    data = ADSORBATES[adsorbate]
    z_surface = max(float(positions[i, 2]) for i, a in enumerate(atoms) if a == binding_element)
    used = set(site_ids)
    top_pool = list(_substrate_top_indices(structure, binding_element))
    for site_index in site_ids:
        reference = positions[site_index].copy()
        ads_positions = np.asarray(data["positions"], dtype=float).copy()
        binding = int(data["binding_atom"])
        if adsorbate in {"OCCO", "COCO"}:
            # Use two neighboring Cu sites as the two C anchors.  The nominal
            # molecular coverage therefore occupies two surface sites.
            partner = min((i for i in top_pool if i not in used and i != site_index), key=lambda i: np.linalg.norm(positions[i, :2] - reference[:2]))
            used.add(partner)
            c_indices = [1, 2] if adsorbate == "OCCO" else [0, 2]
            c0, c1 = ads_positions[c_indices]
            v = positions[partner, :2] - reference[:2]
            length = max(np.linalg.norm(c1[:2] - c0[:2]), 1e-8)
            unit = v / max(np.linalg.norm(v), 1e-8)
            ads_positions[:, :2] -= c0[:2]
            ads_positions[:, :2] += reference[:2] + 0.5 * v
            ads_positions[c_indices[0], :2] = reference[:2]
            ads_positions[c_indices[1], :2] = positions[partner, :2]
            ads_positions[:, 2] += 2.0 - ads_positions[c_indices[0], 2]
        else:
            ads_positions -= ads_positions[binding]
            ads_positions += reference + np.array([0.0, 0.0, 2.0])
        atoms.extend(list(data["atoms"]))
        positions = np.vstack([positions, ads_positions])
    metadata = dict(structure.get("metadata", {}))
    site_count = len(_substrate_top_indices(structure, binding_element))
    site_occupancy = len(site_ids) * (2 if adsorbate in {"OCCO", "COCO"} else 1)
    metadata.update({
        "adsorbate": adsorbate,
        "adsorbate_count": len(site_ids),
        "binding_element": binding_element,
        "coverage_ml": float(coverage),
        "coverage_ml_total": float(coverage),
        "coverage_site_count": int(site_occupancy),
        "coverage_reference": f"{binding_element}_top_sites",
        "coverage_reference_count": int(site_count),
        "adsorption_region": region,
        "local_geometry_label": region,
        "absorber_element": "Cu",
        "xas_edge": "Cu K",
        "label_schema_version": "1.1",
        "structure_status": "parent_relaxed_plus_adsorbate_unrelaxed",
    })
    return {"atoms": atoms, "positions": positions, "cell": cell, "metadata": metadata}

def _coverage_count(coverage, site_count, adsorbate):
    occupancy = 2 if adsorbate in {"OCCO", "COCO"} else 1
    return max(1, int(round(coverage * site_count / occupancy)))

def _manifest_row(sample_id, structure):
    m = structure["metadata"]
    return {
        "id": sample_id,
        "adsorbate": m.get("adsorbate", "clean"),
        "coverage_ml": m.get("coverage_ml_total", m.get("coverage_ml", 0.0)),
        "co_coverage_ml": m.get("co_coverage_ml", ""),
        "h_coverage_ml": m.get("h_coverage_ml", ""),
        "coverage_site_count": m.get("coverage_site_count", 0),
        "coverage_reference": m.get("coverage_reference", "Cu_top_sites"),
        "region": m.get("adsorption_region", "clean"),
        "local_geometry_label": m.get("local_geometry_label", "clean"),
        "absorber": "Cu",
        "edge": "K",
        "parent_relaxation": "relaxed",
        "structure_status": m.get("structure_status", "parent_relaxed_plus_adsorbate_unrelaxed"),
    }

def build(parent, output):
    output = Path(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copy2(parent, output / "parent_CONTCAR")
    base = load_relaxed_interface(parent, cu_rows=4, au_rows=4)
    cu_sites = len(_substrate_top_indices(base, "Cu"))
    au_sites = len(_substrate_top_indices(base, "Au"))
    samples = [("clean", base, "clean")]

    # Pure adsorbate families span the coverage range used in Cu CO2RR
    # surface calculations. H includes 1 ML because the literature samples it.
    for adsorbate, max_cov in [("CO", 0.75), ("H", 1.0), ("OH", 0.75), ("O", 0.75)]:
        for coverage in COVERAGE_GRID + ((1.0,) if adsorbate == "H" else ()):
            count = _coverage_count(coverage, cu_sites, adsorbate)
            ids = _spread_sites(base, "Cu", count, preferred_y=7.3)
            samples.append((f"{adsorbate}_{coverage:.2f}ML_Cu", _add_species(base, adsorbate, ids, coverage, "Cu_top_surface", "Cu"), "Cu_top_surface"))

    # C1 intermediates are sampled at low and intermediate coverage.
    for adsorbate in ("CHO", "CH2", "CH3", "CO2"):
        for coverage in (0.25, 0.50):
            count = _coverage_count(coverage, cu_sites, adsorbate)
            ids = _spread_sites(base, "Cu", count, preferred_y=7.3)
            samples.append((f"{adsorbate}_{coverage:.2f}ML_Cu", _add_species(base, adsorbate, ids, coverage, "Cu_top_surface", "Cu"), "Cu_top_surface"))

    # C2 species occupy two Cu sites per molecule. Their coverage is therefore
    # reported as occupied Cu sites, while adsorbate_count remains molecular.
    for adsorbate in ("OCCO", "COCO"):
        for coverage in (0.25, 0.50):
            count = _coverage_count(coverage, cu_sites, adsorbate)
            ids = _spread_sites(base, "Cu", count, preferred_y=7.3)
            samples.append((f"{adsorbate}_{coverage:.2f}ML_Cu", _add_species(base, adsorbate, ids, coverage, "Cu_Au_boundary_bridge", "Cu"), "Cu_Au_boundary_bridge"))

    # Mixed CO/H states target the coverage window associated with restructuring.
    for co_cov, h_cov in ((0.25, 0.50), (0.50, 0.25)):
        co_ids = _spread_sites(base, "Cu", _coverage_count(co_cov, cu_sites, "CO"), preferred_y=7.3)
        co_structure = _add_species(base, "CO", co_ids, co_cov, "mixed_Cu_surface", "Cu")
        h_ids = _spread_sites(co_structure, "Cu", _coverage_count(h_cov, cu_sites, "H"), preferred_y=5.1, exclude=co_ids)
        mixed = _add_species(co_structure, "H", h_ids, h_cov, "mixed_Cu_surface", "Cu")
        mixed["metadata"].update({"adsorbate": "CO+H", "coadsorbates": "CO+H", "co_coverage_ml": co_cov, "h_coverage_ml": h_cov, "coverage_ml_total": co_cov + h_cov, "coverage_site_count": int(round((co_cov + h_cov) * cu_sites)), "local_geometry_label": "mixed_Cu_surface"})
        samples.append((f"CO_{co_cov:.2f}ML_H_{h_cov:.2f}ML_Cu", mixed, "mixed_Cu_surface"))

    rows = []
    for index, (name, structure, _region) in enumerate(samples):
        sample_id = f"{index:03d}_{name}"
        folder = output / sample_id
        folder.mkdir()
        _write_poscar(folder / "POSCAR", structure["atoms"], structure["positions"], structure["cell"])
        (folder / "structure_info.json").write_text(json.dumps(structure["metadata"], indent=2, default=str))
        rows.append(_manifest_row(sample_id, structure))
    fields = list(rows[0])
    with (output / "dataset_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (output / "dataset_manifest.json").write_text(json.dumps(rows, indent=2))
    (output / "README.md").write_text(
        "# Coverage-resolved Cu K-edge showcase dataset\n\n"
        "This dataset starts from the attached relaxed plain Cu/Au(111) 1:1 parent. "
        "The parent contains 36 relaxed top-layer Cu sites and 32 top-layer Au sites. "
        "Coverage is reported per top-layer binding-element site; C2 molecules count as two occupied Cu sites. "
        "The coverage grid is 0.25, 0.50, and 0.75 ML, with 1.0 ML added for H. "
        "These structures retain the relaxed parent and append unrelaxed adsorbates, so they require geometry relaxation before production XAS.\n"
    )
    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = build(args.parent, args.output)
    print(f"generated {len(rows)} structures in {args.output}")
