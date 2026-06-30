"""
Structure Generator Tool for CO2RR XAS Agent
Skill 1 Implementation

Updated for web_xas_agent:
- single-metal and lateral interface generation
- Cu/Au auto/manual interface repeat matching with strain metadata
- interface-aware OCCO placement
- caller-controlled CO placement through site_index
"""

import os
import json
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union
INTERFACE_GENERATOR_UPDATE_TAG = "v4_2026-06-29_adsorbate_default_sites_generator_level"

from .utils import (
    METAL_DATA, ADSORBATES, ADSORPTION_SITES, CO2RR_PATHWAY,
    write_poscar, read_poscar, read_structure_file, ensure_dir, generate_uuid, get_timestamp
)


# Adsorbate-specific defaults used by the generator itself, not only by the web UI.
# This keeps command-line/generated structures consistent with the visualization app.
DEFAULT_ADSORBATE_SITES = {
    "CO": "top",
    "CO2": "top",
    "CHO": "top",
    "CHOH": "top",
    "CH3": "top",
    "CH4": "top",
    "OH": "top",
    "H2O": "top",
    "H": "fcc",
    "CH": "fcc",
    "CH2": "fcc",
    "OCCO": "bridge_interface",
    "COCO": "bridge_interface",
}

# Default interface-side binding preference for single-binding adsorbates.
# If the requested element is not present, the generator falls back to Cu if present,
# then to element1 of the interface.
DEFAULT_INTERFACE_BINDING_ELEMENTS = {
    "CO": "Cu",
    "CO2": "Cu",
    "CHO": "Cu",
    "CHOH": "Cu",
    "CH": "Cu",
    "CH2": "Cu",
    "CH3": "Cu",
    "CH4": "Cu",
    "H": "Cu",
    "OH": "Cu",
    "H2O": "Cu",
    "OCCO": "Cu",
    "COCO": "Cu",
}


class StructureGenerator:
    """
    Generate atomic structures for CO2RR simulations.

    Supports:
    - FCC, BCC, HCP metal surfaces
    - Various facets (111, 100, 110, 0001)
    - Lateral bimetallic interfaces
    - CO2RR adsorbate placement
    """

    def __init__(self):
        self.metal_data = METAL_DATA
        self.adsorbates = ADSORBATES
        self.sites = ADSORPTION_SITES

    def _resolve_adsorbate_site(self, adsorbate_name: Optional[str], site: Optional[str]) -> str:
        """Return the actual site used for placement and metadata."""
        if not adsorbate_name or adsorbate_name == "clean":
            return "clean"
        if site is None or str(site).lower() == "auto":
            return DEFAULT_ADSORBATE_SITES.get(adsorbate_name, "top")
        return str(site)

    def _top_layer_indices(self, positions: np.ndarray, tol: float = 0.5) -> np.ndarray:
        z_max = positions[:, 2].max()
        return np.where(np.abs(positions[:, 2] - z_max) < tol)[0]

    @staticmethod
    def _periodic_delta(value: float, ref: float, length: Optional[float]) -> float:
        delta = abs(float(value) - float(ref))
        if length and length > 1e-12:
            delta = min(delta, length - delta)
        return delta

    def _center_top_site_index(
        self,
        atoms: List[str],
        positions: np.ndarray,
        cell: np.ndarray,
        top_indices: np.ndarray,
        preferred_element: Optional[str] = None,
    ) -> int:
        """Return a site_index for a top atom near the slab center."""
        candidates = list(top_indices)
        if preferred_element:
            preferred = [idx for idx in top_indices if atoms[idx] == preferred_element]
            if preferred:
                candidates = preferred
        center_xy = 0.5 * (cell[0, :2] + cell[1, :2])
        chosen = min(candidates, key=lambda idx: float(np.linalg.norm(positions[idx, :2] - center_xy)))
        return list(top_indices).index(chosen)

    def _interface_binding_element(
        self,
        adsorbate_name: str,
        metadata: Dict[str, Any],
        requested: Optional[str] = None,
    ) -> Optional[str]:
        elements = [metadata.get("element1"), metadata.get("element2")]
        elements = [el for el in elements if el]
        if requested in elements:
            return requested
        preferred = DEFAULT_INTERFACE_BINDING_ELEMENTS.get(adsorbate_name)
        if preferred in elements:
            return preferred
        if "Cu" in elements:
            return "Cu"
        return elements[0] if elements else requested

    def _interface_top_site_index(
        self,
        atoms: List[str],
        positions: np.ndarray,
        cell: np.ndarray,
        top_indices: np.ndarray,
        metadata: Dict[str, Any],
        adsorbate_name: str,
        preferred_element: Optional[str] = None,
    ) -> int:
        """Return a site_index for a top atom on/nearest a lateral interface line."""
        interface_meta = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}
        if interface_meta.get("type") != "lateral":
            return self._center_top_site_index(atoms, positions, cell, top_indices, preferred_element)

        split_axis_name = interface_meta.get("split_axis", "y")
        split_axis = 1 if split_axis_name == "y" else 0
        line_axis = 0 if split_axis == 1 else 1
        split_coord = float(interface_meta.get("split_coordinate", cell[split_axis, split_axis] / 2.0))
        line_length = float(cell[line_axis, line_axis]) if cell[line_axis, line_axis] else None
        ref_line = 0.5 * line_length if line_length else 0.0

        preferred = self._interface_binding_element(adsorbate_name, metadata, preferred_element)
        candidates = list(top_indices)
        if preferred:
            preferred_candidates = [idx for idx in top_indices if atoms[idx] == preferred]
            if preferred_candidates:
                candidates = preferred_candidates

        def score(idx: int) -> float:
            return (
                abs(float(positions[idx, split_axis]) - split_coord)
                + 0.25 * self._periodic_delta(float(positions[idx, line_axis]), ref_line, line_length)
            )

        chosen = min(candidates, key=score)
        return list(top_indices).index(chosen)

    def _adsorbate_formula(self, adsorbate_name: Optional[str]) -> Optional[str]:
        if not adsorbate_name or adsorbate_name not in self.adsorbates:
            return None
        counts: Dict[str, int] = {}
        for atom in self.adsorbates[adsorbate_name]["atoms"]:
            counts[atom] = counts.get(atom, 0) + 1
        parts = []
        for element in ("C", "H", "O"):
            count = counts.pop(element, 0)
            if count:
                parts.append(f"{element}{count if count > 1 else ''}")
        for element in sorted(counts):
            count = counts[element]
            parts.append(f"{element}{count if count > 1 else ''}")
        return "".join(parts)

    def _adsorbate_intermediate_class(self, adsorbate_name: Optional[str]) -> Optional[str]:
        if not adsorbate_name or adsorbate_name not in self.adsorbates:
            return None
        carbon_count = sum(1 for atom in self.adsorbates[adsorbate_name]["atoms"] if atom == "C")
        return f"C{carbon_count}" if carbon_count else "C0"

    def _binding_atom_symbol(self, adsorbate_name: Optional[str]) -> Optional[str]:
        if not adsorbate_name or adsorbate_name not in self.adsorbates:
            return None
        ads_data = self.adsorbates[adsorbate_name]
        return ads_data["atoms"][int(ads_data["binding_atom"])]

    def _with_ml_metadata(
        self,
        metadata: Dict[str, Any],
        adsorbate_name: Optional[str] = None,
        site: Optional[str] = None,
        metadata_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attach catalyst/adsorbate metadata intended for downstream XAS ML records."""
        enriched = dict(metadata)
        overrides = metadata_overrides or {}

        if "element1" in enriched:
            elements = [enriched.get("element1"), enriched.get("element2")]
            facet = enriched.get("facet1") if enriched.get("facet1") == enriched.get("facet2") else f"{enriched.get('facet1')}/{enriched.get('facet2')}"
            site_type = "interface"
        else:
            elements = [enriched.get("element")]
            facet = enriched.get("facet")
            site_type = "surface"

        elements = [element for element in elements if element]
        composition = "".join(elements)
        site_value = site or enriched.get("adsorption_site")
        ads_value = adsorbate_name or enriched.get("adsorbate")

        catalyst = {
            "elements": elements,
            "composition": composition,
            "surface_facet": str(facet) if facet is not None else None,
            "site_type": site_type,
        }

        if ads_value:
            adsorbate = {
                "identity": ads_value,
                "formula": self._adsorbate_formula(ads_value),
                "intermediate_class": self._adsorbate_intermediate_class(ads_value),
                "binding_mode": site_value or "top",
                "adsorption_site": (
                    f"{'-'.join(elements)} interface"
                    if site_type == "interface" and len(elements) > 1
                    else f"{composition}({facet}) {site_value or 'top'}"
                ),
                "binding_atom": self._binding_atom_symbol(ads_value),
            }
        else:
            adsorbate = {
                "identity": "clean",
                "formula": None,
                "intermediate_class": None,
                "binding_mode": None,
                "adsorption_site": None,
                "binding_atom": None,
            }

        catalyst.update(overrides.get("catalyst", {}))
        adsorbate.update(overrides.get("adsorbate", {}))

        structure_id = overrides.get("structure_id") or enriched.get("structure_id")
        if not structure_id:
            id_parts = [composition or "structure", str(facet or "facet")]
            if ads_value:
                id_parts.extend([ads_value, str(site_value or "top")])
            else:
                id_parts.append("clean")
            id_parts.append("001")
            structure_id = "_".join(part.replace("/", "-").replace(" ", "_") for part in id_parts if part)

        catalyst["structure_id"] = structure_id
        enriched["structure_id"] = structure_id
        enriched["catalyst"] = catalyst
        enriched["adsorbate_metadata"] = adsorbate
        return enriched

    def generate_fcc_surface(
        self,
        element: str,
        facet: str = "111",
        supercell: Tuple[int, int] = (3, 3),
        layers: int = 4,
        vacuum: float = 15.0,
        lattice_constant: Optional[float] = None,
    ) -> Dict:
        """Generate FCC metal surface slab."""
        a = lattice_constant or self.metal_data[element]["a"]

        if facet == "111":
            return self._generate_fcc_111(element, a, supercell, layers, vacuum)
        if facet == "100":
            return self._generate_fcc_100(element, a, supercell, layers, vacuum)
        if facet == "110":
            return self._generate_fcc_110(element, a, supercell, layers, vacuum)
        raise ValueError(f"Unsupported facet: {facet}")

    def _generate_fcc_111(
        self,
        element: str,
        a: float,
        supercell: Tuple[int, int],
        layers: int,
        vacuum: float,
    ) -> Dict:
        """Generate FCC(111) surface."""
        a1 = a / np.sqrt(2) * np.array([1.0, 0.0, 0.0])
        a2 = a / np.sqrt(2) * np.array([0.5, np.sqrt(3) / 2, 0.0])
        layer_spacing = a / np.sqrt(3)
        stacking = [
            np.array([0.0, 0.0]),
            np.array([1 / 3, 1 / 3]),
            np.array([2 / 3, 2 / 3]),
        ]

        atoms = []
        positions = []
        for layer in range(layers):
            shift = stacking[layer % 3]
            z = layer * layer_spacing
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    pos = (i + shift[0]) * a1 + (j + shift[1]) * a2
                    pos[2] = z
                    atoms.append(element)
                    positions.append(pos)

        positions = np.array(positions, dtype=float)
        positions[:, 2] -= positions[:, 2].min()
        slab_height = positions[:, 2].max()
        cell_height = slab_height + vacuum

        cell = np.array([
            supercell[0] * a1,
            supercell[1] * a2,
            [0.0, 0.0, cell_height],
        ])

        return {
            "atoms": atoms,
            "positions": positions,
            "cell": cell,
            "metadata": {
                "element": element,
                "structure": "fcc",
                "facet": "111",
                "supercell": list(supercell),
                "layers": layers,
                "vacuum": vacuum,
                "lattice_constant": a,
            },
        }

    def _generate_fcc_100(
        self,
        element: str,
        a: float,
        supercell: Tuple[int, int],
        layers: int,
        vacuum: float,
    ) -> Dict:
        """Generate FCC(100) surface."""
        a1 = a / np.sqrt(2) * np.array([1.0, 0.0, 0.0])
        a2 = a / np.sqrt(2) * np.array([0.0, 1.0, 0.0])
        layer_spacing = a / 2
        stacking = [np.array([0.0, 0.0]), np.array([0.5, 0.5])]

        atoms = []
        positions = []
        for layer in range(layers):
            shift = stacking[layer % 2]
            z = layer * layer_spacing
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    pos = (i + shift[0]) * a1 + (j + shift[1]) * a2
                    pos[2] = z
                    atoms.append(element)
                    positions.append(pos)

        positions = np.array(positions, dtype=float)
        positions[:, 2] -= positions[:, 2].min()
        slab_height = positions[:, 2].max()
        cell_height = slab_height + vacuum

        cell = np.array([
            supercell[0] * a1,
            supercell[1] * a2,
            [0.0, 0.0, cell_height],
        ])

        return {
            "atoms": atoms,
            "positions": positions,
            "cell": cell,
            "metadata": {
                "element": element,
                "structure": "fcc",
                "facet": "100",
                "supercell": list(supercell),
                "layers": layers,
                "vacuum": vacuum,
                "lattice_constant": a,
            },
        }

    def _generate_fcc_110(
        self,
        element: str,
        a: float,
        supercell: Tuple[int, int],
        layers: int,
        vacuum: float,
    ) -> Dict:
        """Generate FCC(110) surface."""
        a1 = a / np.sqrt(2) * np.array([1.0, 0.0, 0.0])
        a2 = a * np.array([0.0, 1.0, 0.0])
        layer_spacing = a / (2 * np.sqrt(2))
        stacking = [np.array([0.0, 0.0]), np.array([0.5, 0.5])]

        atoms = []
        positions = []
        for layer in range(layers):
            shift = stacking[layer % 2]
            z = layer * layer_spacing
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    pos = (i + shift[0]) * a1 + (j + shift[1]) * a2
                    pos[2] = z
                    atoms.append(element)
                    positions.append(pos)

        positions = np.array(positions, dtype=float)
        positions[:, 2] -= positions[:, 2].min()
        slab_height = positions[:, 2].max()
        cell_height = slab_height + vacuum

        cell = np.array([
            supercell[0] * a1,
            supercell[1] * a2,
            [0.0, 0.0, cell_height],
        ])

        return {
            "atoms": atoms,
            "positions": positions,
            "cell": cell,
            "metadata": {
                "element": element,
                "structure": "fcc",
                "facet": "110",
                "supercell": list(supercell),
                "layers": layers,
                "vacuum": vacuum,
                "lattice_constant": a,
            },
        }

    def generate_bcc_surface(
        self,
        element: str,
        facet: str = "110",
        supercell: Tuple[int, int] = (3, 3),
        layers: int = 4,
        vacuum: float = 15.0,
        lattice_constant: Optional[float] = None,
    ) -> Dict:
        """Generate BCC metal surface slab."""
        a = lattice_constant or self.metal_data[element]["a"]

        if facet == "110":
            return self._generate_bcc_110(element, a, supercell, layers, vacuum)
        if facet == "100":
            return self._generate_bcc_100(element, a, supercell, layers, vacuum)
        raise ValueError(f"Unsupported BCC facet: {facet}")

    def _generate_bcc_110(
        self,
        element: str,
        a: float,
        supercell: Tuple[int, int],
        layers: int,
        vacuum: float,
    ) -> Dict:
        """Generate BCC(110) surface."""
        a1 = a * np.array([1.0, 0.0, 0.0])
        a2 = a / np.sqrt(2) * np.array([0.0, 1.0, 0.0])
        layer_spacing = a / np.sqrt(2)
        stacking = [np.array([0.0, 0.0]), np.array([0.5, 0.5])]

        atoms = []
        positions = []
        for layer in range(layers):
            shift = stacking[layer % 2]
            z = layer * layer_spacing
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    pos = (i + shift[0]) * a1 + (j + shift[1]) * a2
                    pos[2] = z
                    atoms.append(element)
                    positions.append(pos)

        positions = np.array(positions, dtype=float)
        positions[:, 2] -= positions[:, 2].min()
        slab_height = positions[:, 2].max()
        cell_height = slab_height + vacuum

        cell = np.array([
            supercell[0] * a1,
            supercell[1] * a2,
            [0.0, 0.0, cell_height],
        ])

        return {
            "atoms": atoms,
            "positions": positions,
            "cell": cell,
            "metadata": {
                "element": element,
                "structure": "bcc",
                "facet": "110",
                "supercell": list(supercell),
                "layers": layers,
                "vacuum": vacuum,
                "lattice_constant": a,
            },
        }

    def _generate_bcc_100(
        self,
        element: str,
        a: float,
        supercell: Tuple[int, int],
        layers: int,
        vacuum: float,
    ) -> Dict:
        """Generate BCC(100) surface."""
        a1 = a * np.array([1.0, 0.0, 0.0])
        a2 = a * np.array([0.0, 1.0, 0.0])
        layer_spacing = a / 2
        stacking = [np.array([0.0, 0.0]), np.array([0.5, 0.5])]

        atoms = []
        positions = []
        for layer in range(layers):
            shift = stacking[layer % 2]
            z = layer * layer_spacing
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    pos = (i + shift[0]) * a1 + (j + shift[1]) * a2
                    pos[2] = z
                    atoms.append(element)
                    positions.append(pos)

        positions = np.array(positions, dtype=float)
        positions[:, 2] -= positions[:, 2].min()
        slab_height = positions[:, 2].max()
        cell_height = slab_height + vacuum

        cell = np.array([
            supercell[0] * a1,
            supercell[1] * a2,
            [0.0, 0.0, cell_height],
        ])

        return {
            "atoms": atoms,
            "positions": positions,
            "cell": cell,
            "metadata": {
                "element": element,
                "structure": "bcc",
                "facet": "100",
                "supercell": list(supercell),
                "layers": layers,
                "vacuum": vacuum,
                "lattice_constant": a,
            },
        }

    def generate_surface(
        self,
        element: str,
        facet: str = "111",
        supercell: Tuple[int, int] = (3, 3),
        layers: int = 4,
        vacuum: float = 15.0,
    ) -> Dict:
        """Generate metal surface based on crystal structure."""
        if element not in self.metal_data:
            raise ValueError(f"Unknown element: {element}. Supported: {list(self.metal_data.keys())}")

        structure = self.metal_data[element]["structure"]
        if structure == "fcc":
            return self.generate_fcc_surface(element, facet, supercell, layers, vacuum)
        if structure == "bcc":
            return self.generate_bcc_surface(element, facet, supercell, layers, vacuum)
        if structure == "hcp":
            raise NotImplementedError("HCP surface generation not yet implemented")
        raise ValueError(f"Unknown structure type: {structure}")

    def _match_lateral_repeats(self, element1: str, element2: str, max_repeat: int = 10) -> Dict[str, Any]:
        """Find a small lateral repeat match for two fcc(111) surface periods."""
        period1 = self.metal_data[element1]["a"] / np.sqrt(2)
        period2 = self.metal_data[element2]["a"] / np.sqrt(2)
        best = None

        for n1 in range(1, max_repeat + 1):
            for n2 in range(1, max_repeat + 1):
                length1 = n1 * period1
                length2 = n2 * period2
                matched = 0.5 * (length1 + length2)
                strain1 = (matched - length1) / length1
                strain2 = (matched - length2) / length2
                score = max(abs(strain1), abs(strain2))
                candidate = {
                    "match_mode": "auto",
                    "element1_repeats": n1,
                    "element2_repeats": n2,
                    "element1_natural_length": length1,
                    "element2_natural_length": length2,
                    "matched_block_length": matched,
                    "element1_strain": strain1,
                    "element2_strain": strain2,
                    "max_abs_strain": score,
                }
                if best is None or (score, n1 + n2) < (
                    best["max_abs_strain"],
                    best["element1_repeats"] + best["element2_repeats"],
                ):
                    best = candidate
        return best or {}

    def generate_interface(
        self,
        element1: str,
        element2: str,
        facet1: str = "111",
        facet2: str = "111",
        supercell: Tuple[int, int] = (3, 3),
        layers1: int = 3,
        layers2: int = 3,
        vacuum: float = 15.0,
        interface_distance: float = 2.5,
        element1_repeats: Optional[int] = None,
        element2_repeats: Optional[int] = None,
        match_mode: str = "auto",
    ) -> Dict:
        """Generate a lateral bimetallic fcc(111) surface interface.

        Geometry convention used by web_xas_agent:
        - x is the interface direction.
        - y is perpendicular to the interface.
        - auto mode minimizes strain along x, so a Cu/Au interface usually has
          9 Cu repeats matched to 8 Au repeats along the interface.
        - supercell[1] is the number of atomic rows away from the interface on
          each side. The periodic cell therefore contains two interfaces:
          one internal interface at y = split_coordinate and one at the y PBC.
        - supercell[0] is only a requested value in auto mode; actual repeats
          come from the lattice match. In manual mode, element1_repeats and
          element2_repeats are used directly.
        """
        if facet1 != facet2:
            raise ValueError("Lateral interfaces currently require matching facets")
        if facet1 != "111":
            raise ValueError("Lateral interface generation currently supports fcc(111) facets")
        if element1 not in self.metal_data or element2 not in self.metal_data:
            raise ValueError(f"Unknown interface elements: {element1}, {element2}")
        if self.metal_data[element1]["structure"] != "fcc" or self.metal_data[element2]["structure"] != "fcc":
            raise ValueError("Lateral interface generation currently supports fcc/fcc interfaces")

        requested_nx = int(supercell[0]) if supercell else 3
        rows_per_side = int(supercell[1]) if supercell else 3
        rows_per_side = max(1, rows_per_side)
        layers = max(int(layers1), int(layers2))

        period1 = self.metal_data[element1]["a"] / np.sqrt(2)
        period2 = self.metal_data[element2]["a"] / np.sqrt(2)

        if match_mode == "manual" or element1_repeats is not None or element2_repeats is not None:
            n1 = max(1, int(element1_repeats or requested_nx))
            n2 = max(1, int(element2_repeats or requested_nx))
            length1 = n1 * period1
            length2 = n2 * period2
            matched = 0.5 * (length1 + length2)
            strain1 = (matched - length1) / length1
            strain2 = (matched - length2) / length2
            match = {
                "match_mode": "manual",
                "match_direction": "x/interface",
                "element1_repeats": n1,
                "element2_repeats": n2,
                "element1_natural_length": length1,
                "element2_natural_length": length2,
                "matched_block_length": matched,
                "element1_strain": strain1,
                "element2_strain": strain2,
                "max_abs_strain": max(abs(strain1), abs(strain2)),
            }
        else:
            match = self._match_lateral_repeats(element1, element2)
            match["match_mode"] = "auto"
            match["match_direction"] = "x/interface"
            n1 = int(match["element1_repeats"])
            n2 = int(match["element2_repeats"])
            matched = float(match["matched_block_length"])

        common_x = float(match["matched_block_length"])

        row_spacing1 = period1 * np.sqrt(3) / 2
        row_spacing2 = period2 * np.sqrt(3) / 2
        block1_y = rows_per_side * row_spacing1
        block2_y = rows_per_side * row_spacing2
        total_y = block1_y + block2_y
        split_y = block1_y

        layer_spacing = 0.5 * (
            self.metal_data[element1]["a"] / np.sqrt(3)
            + self.metal_data[element2]["a"] / np.sqrt(3)
        )
        stacking = [
            np.array([0.0, 0.0]),
            np.array([1 / 3, 1 / 3]),
            np.array([2 / 3, 2 / 3]),
        ]

        atoms: List[str] = []
        positions: List[np.ndarray] = []

        for layer in range(layers):
            shift = stacking[layer % 3]
            z = layer * layer_spacing

            # element1 block: below the internal interface, y in [0, split_y)
            for j in range(rows_per_side):
                row_shift = 0.5 * (j % 2)
                y = (j + shift[1]) * row_spacing1
                for i in range(n1):
                    x_frac = (i + shift[0] + row_shift) / n1
                    x = (x_frac % 1.0) * common_x
                    atoms.append(element1)
                    positions.append(np.array([x, y, z]))

            # element2 block: above the internal interface, y in [split_y, total_y)
            for j in range(rows_per_side):
                row_shift = 0.5 * (j % 2)
                y = split_y + (j + shift[1]) * row_spacing2
                for i in range(n2):
                    x_frac = (i + shift[0] + row_shift) / n2
                    x = (x_frac % 1.0) * common_x
                    atoms.append(element2)
                    positions.append(np.array([x, y, z]))

        all_positions = np.array(positions, dtype=float)
        all_positions[:, 2] -= all_positions[:, 2].min()
        slab_height = all_positions[:, 2].max()

        cell = np.array([
            [common_x, 0.0, 0.0],
            [0.0, total_y, 0.0],
            [0.0, 0.0, slab_height + vacuum],
        ])

        return {
            "atoms": atoms,
            "positions": all_positions,
            "cell": cell,
            "metadata": {
                "element1": element1,
                "element2": element2,
                "structure_type": "interface",
                "interface_type": "lateral",
                "facet1": facet1,
                "facet2": facet2,
                "supercell": [max(n1, n2), 2 * rows_per_side],
                "requested_supercell": list(supercell),
                "layers1": layers,
                "layers2": layers,
                "vacuum": vacuum,
                "interface_distance": interface_distance,
                "interface": {
                    "type": "lateral",
                    "interface_axis": "x",
                    "split_axis": "y",
                    "split_coordinate": split_y,
                    "rows_per_side": rows_per_side,
                    "element1_block": {"y_min": 0.0, "y_max": split_y},
                    "element2_block": {"y_min": split_y, "y_max": total_y},
                    "actual_repeats": {
                        element1: n1,
                        element2: n2,
                        "rows_per_side": rows_per_side,
                    },
                    "match": match,
                },
            },
        }

    def add_adsorbate(
        self,
        structure: Dict,
        adsorbate_name: str,
        site: Optional[str] = "auto",
        height: float = 2.0,
        site_index: Optional[int] = None,
        metadata_overrides: Optional[Dict[str, Any]] = None,
        binding_element: Optional[str] = None,
    ) -> Dict:
        """Add adsorbate to a surface or interface structure.

        Generator-level defaults are used when site="auto" and site_index=None.
        This is important because structures generated from scripts should match
        the Streamlit visualization behavior.

        Defaults:
        - Single-metal surfaces: adsorbate-specific site near slab center.
        - Lateral interfaces: adsorbate-specific site at the interface.
        - OCCO/COCO on lateral interfaces: bridge the two interface sides.
        """
        if adsorbate_name not in self.adsorbates:
            raise ValueError(f"Unknown adsorbate: {adsorbate_name}. Available: {list(self.adsorbates.keys())}")

        effective_site = self._resolve_adsorbate_site(adsorbate_name, site)

        ads_data = self.adsorbates[adsorbate_name]
        ads_atoms = list(ads_data["atoms"])
        ads_positions = np.array(ads_data["positions"], dtype=float).copy()
        binding_idx = int(ads_data["binding_atom"])

        atoms = structure["atoms"]
        positions = np.array(structure["positions"], dtype=float)
        cell = np.array(structure["cell"], dtype=float)
        metadata = structure.get("metadata", {})
        interface_meta = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}

        z_max = positions[:, 2].max()
        top_indices = self._top_layer_indices(positions)
        if len(top_indices) == 0:
            raise ValueError("No top-layer atoms found for adsorbate placement")

        if site_index is None:
            if interface_meta.get("type") == "lateral":
                site_index = self._interface_top_site_index(
                    atoms=atoms,
                    positions=positions,
                    cell=cell,
                    top_indices=top_indices,
                    metadata=metadata,
                    adsorbate_name=adsorbate_name,
                    preferred_element=binding_element,
                )
            else:
                site_index = self._center_top_site_index(
                    atoms=atoms,
                    positions=positions,
                    cell=cell,
                    top_indices=top_indices,
                    preferred_element=binding_element,
                )

        if site_index >= len(top_indices) or site_index < 0:
            site_index = 0

        ref_pos = positions[top_indices[site_index]].copy()

        if adsorbate_name in {"OCCO", "COCO"} and interface_meta.get("type") == "lateral":
            split_axis_name = interface_meta.get("split_axis", "y")
            split_axis = 1 if split_axis_name == "y" else 0
            line_axis = 0 if split_axis == 1 else 1
            split_coord = float(interface_meta.get("split_coordinate", cell[split_axis, split_axis] / 2.0))
            element1 = metadata.get("element1")
            element2 = metadata.get("element2")
            line_length = float(cell[line_axis, line_axis]) if cell[line_axis, line_axis] else None
            ref_line = float(ref_pos[line_axis])

            # Choose one top atom from each side near the selected position along the interface.
            side1_candidates = [
                idx for idx in top_indices
                if atoms[idx] == element1 and positions[idx, split_axis] <= split_coord + 1e-8
            ]
            side2_candidates = [
                idx for idx in top_indices
                if atoms[idx] == element2 and positions[idx, split_axis] >= split_coord - 1e-8
            ]
            if not side1_candidates:
                side1_candidates = [idx for idx in top_indices if positions[idx, split_axis] <= split_coord + 1e-8]
            if not side2_candidates:
                side2_candidates = [idx for idx in top_indices if positions[idx, split_axis] >= split_coord - 1e-8]

            def score(idx: int) -> float:
                return (
                    abs(float(positions[idx, split_axis]) - split_coord)
                    + 0.25 * self._periodic_delta(float(positions[idx, line_axis]), ref_line, line_length)
                )

            left_idx = min(side1_candidates or list(top_indices), key=score)
            right_idx = min(side2_candidates or list(top_indices), key=score)
            left_pos = positions[left_idx].copy()
            right_pos = positions[right_idx].copy()

            c_indices = [idx for idx, atom in enumerate(ads_atoms) if atom == "C"]
            if len(c_indices) >= 2:
                c_left, c_right = c_indices[:2]
                original_c_left = ads_positions[c_left].copy()
                original_c_right = ads_positions[c_right].copy()

                target_left = left_pos.copy()
                target_right = right_pos.copy()
                target_left[2] = z_max + height
                target_right[2] = z_max + height

                # Preserve each atom's local vector relative to the nearest C.
                new_ads_positions = ads_positions.copy()
                for k in range(len(ads_atoms)):
                    d_left = np.linalg.norm(ads_positions[k] - original_c_left)
                    d_right = np.linalg.norm(ads_positions[k] - original_c_right)
                    if d_left <= d_right:
                        new_ads_positions[k] = target_left + (ads_positions[k] - original_c_left)
                    else:
                        new_ads_positions[k] = target_right + (ads_positions[k] - original_c_right)
                new_ads_positions[c_left] = target_left
                new_ads_positions[c_right] = target_right
                ads_positions = new_ads_positions
                effective_site = "bridge_interface"
            else:
                target = ref_pos.copy()
                target[split_axis] = split_coord
                target[2] = z_max + height
                ads_positions -= ads_positions[binding_idx]
                ads_positions += target
                effective_site = "bridge_interface"
        else:
            placement_site = "top" if effective_site == "bridge_interface" else effective_site
            site_offset = self.sites.get(placement_site, np.array([0.0, 0.0]))
            supercell = metadata.get("supercell", [3, 3])
            a1 = cell[0] / max(float(supercell[0]), 1.0)
            a2 = cell[1] / max(float(supercell[1]), 1.0)

            base_pos = ref_pos.copy()
            base_pos[:2] += site_offset[0] * a1[:2] + site_offset[1] * a2[:2]
            base_pos[2] = z_max + height

            # Align the declared binding atom to the selected site. For CO2 this
            # means C binds to the surface if tools/utils.py has binding_atom=0.
            ads_positions -= ads_positions[binding_idx]
            ads_positions += base_pos

        new_atoms = atoms.copy() + ads_atoms
        new_positions = np.vstack([positions, ads_positions])

        new_metadata = metadata.copy()
        new_metadata["adsorbate"] = adsorbate_name
        new_metadata["adsorption_site"] = effective_site
        new_metadata["adsorption_height"] = height
        if binding_element is not None:
            new_metadata["adsorption_binding_element"] = binding_element
        new_metadata = self._with_ml_metadata(new_metadata, adsorbate_name, effective_site, metadata_overrides)

        return {
            "atoms": new_atoms,
            "positions": new_positions,
            "cell": cell.copy(),
            "metadata": new_metadata,
        }

    def generate_co2rr_pathway(
        self,
        element: str,
        facet: str = "111",
        adsorbates: Optional[List[str]] = None,
        supercell: Tuple[int, int] = (3, 3),
        layers: int = 4,
        site: str = "auto",
        element2: Optional[str] = None,
        facet2: Optional[str] = None,
    ) -> List[Dict]:
        """Generate complete CO2RR reaction pathway structures."""
        if adsorbates is None:
            adsorbates = CO2RR_PATHWAY

        structures = []
        if element2:
            base_surface = self.generate_interface(element, element2, facet, facet2 or facet, supercell, layers, layers)
        else:
            base_surface = self.generate_surface(element, facet, supercell, layers)

        clean = base_surface.copy()
        clean["metadata"] = base_surface["metadata"].copy()
        clean["metadata"]["step"] = "clean"
        structures.append(clean)

        for ads in adsorbates:
            if element2:
                surface = self.generate_interface(element, element2, facet, facet2 or facet, supercell, layers, layers)
            else:
                surface = self.generate_surface(element, facet, supercell, layers)

            structure = self.add_adsorbate(surface, ads, site=site)
            structure["metadata"]["step"] = ads
            structures.append(structure)

        return structures

    def save_structure(self, structure: Dict, output_dir: str, name: Optional[str] = None) -> Dict[str, str]:
        """Save structure to POSCAR and metadata JSON."""
        ensure_dir(output_dir)

        if name is None:
            meta = structure["metadata"]
            parts = []
            if "element" in meta:
                parts.append(meta["element"])
            if "element1" in meta:
                parts.append(f"{meta['element1']}-{meta['element2']}")
            if "facet" in meta:
                parts.append(meta["facet"])
            if "adsorbate" in meta:
                parts.append(meta["adsorbate"])
            name = "_".join(parts) if parts else "structure"

        poscar_path = os.path.join(output_dir, "POSCAR")
        write_poscar(
            structure["atoms"],
            structure["positions"],
            structure["cell"],
            poscar_path,
            comment=f"CO2RR: {name}",
        )

        metadata_path = os.path.join(output_dir, "structure_info.json")
        metadata = structure["metadata"].copy()
        if "catalyst" not in metadata:
            metadata = self._with_ml_metadata(metadata)
        metadata["n_atoms"] = len(structure["atoms"])
        metadata["composition"] = {}
        for atom in structure["atoms"]:
            metadata["composition"][atom] = metadata["composition"].get(atom, 0) + 1

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {"poscar": poscar_path, "metadata": metadata_path}


# =============================================================================
# Skill Interface Function
# =============================================================================

def execute_structure_generation(
    mode: str = "generate",
    structure_file: Optional[str] = None,
    metal1: Optional[str] = None,
    metal2: Optional[str] = None,
    facet1: str = "111",
    facet2: Optional[str] = None,
    adsorbate: Optional[Union[str, List[str]]] = None,
    site: str = "auto",
    supercell: Tuple[int, int] = (3, 3),
    layers: int = 4,
    vacuum: float = 15.0,
    output_dir: str = "output",
    full_pathway: bool = False,
    metadata_overrides: Optional[Dict[str, Any]] = None,
    interface_match_mode: str = "auto",
    element1_repeats: Optional[int] = None,
    element2_repeats: Optional[int] = None,
) -> Dict:
    """Execute Skill 1: Structure Generation."""
    generator = StructureGenerator()
    results = {"status": "success", "structures": [], "files": []}

    if mode == "read_file":
        if not structure_file or not os.path.exists(structure_file):
            return {"status": "error", "message": f"Structure file not found: {structure_file}"}

        structure = read_structure_file(structure_file)
        structure["metadata"] = {"source_file": structure_file}
        struct_dir = os.path.join(output_dir, "structure")
        files = generator.save_structure(structure, struct_dir)
        results["structures"].append(structure["metadata"])
        results["files"].append(files)

    elif mode == "generate":
        if not metal1:
            return {"status": "error", "message": "metal1 is required for generate mode"}

        if full_pathway:
            structures = generator.generate_co2rr_pathway(
                element=metal1,
                facet=facet1,
                supercell=supercell,
                layers=layers,
                site=site,
                element2=metal2,
                facet2=facet2,
            )

            for struct in structures:
                step = struct["metadata"].get("step", "unknown")
                struct_dir = os.path.join(output_dir, step, "structure")
                files = generator.save_structure(struct, struct_dir, name=step)
                results["structures"].append(struct["metadata"])
                results["files"].append(files)
        else:
            if metal2:
                structure = generator.generate_interface(
                    metal1,
                    metal2,
                    facet1,
                    facet2 or facet1,
                    supercell,
                    layers,
                    layers,
                    vacuum,
                    match_mode=interface_match_mode,
                    element1_repeats=element1_repeats,
                    element2_repeats=element2_repeats,
                )
            else:
                structure = generator.generate_surface(metal1, facet1, supercell, layers, vacuum)

            if adsorbate:
                if isinstance(adsorbate, str):
                    adsorbate = [adsorbate]

                for ads in adsorbate:
                    struct = generator.add_adsorbate(
                        structure.copy(),
                        ads,
                        site=site,
                        metadata_overrides=metadata_overrides,
                    )
                    struct_dir = os.path.join(output_dir, ads, "structure")
                    files = generator.save_structure(struct, struct_dir)
                    results["structures"].append(struct["metadata"])
                    results["files"].append(files)
            else:
                structure["metadata"] = generator._with_ml_metadata(
                    structure["metadata"],
                    metadata_overrides=metadata_overrides,
                )
                struct_dir = os.path.join(output_dir, "clean", "structure")
                files = generator.save_structure(structure, struct_dir)
                results["structures"].append(structure["metadata"])
                results["files"].append(files)

    return results
