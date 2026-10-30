#!/bin/bash
set -euo pipefail

SCRIPT_VERSION="2026-10-20.2"

# Convert four layer-resolved Cu K-edge *_conv.txt spectra into ONE ISAAC JSON
# containing four measurement.series blocks for ONE relaxed structure.
#
# Structure metadata are read directly from the nearest relaxed CONTCAR.
# structure_info.json is intentionally NOT required or used in the final record.
#
# Usage:
#   # One completed structure
#   ./convert_layers_to_isaac.sh /path/to/XAS/FDMNES/green
#   ./convert_layers_to_isaac.sh /path/to/XAS/FDMNES/green /path/to/output.json
#
#   # Batch-convert all finished calculations below a CO2RR root
#   ./convert_layers_to_isaac.sh --all /pscratch/sd/h/hjia/CO2RR
#   ./convert_layers_to_isaac.sh --all /pscratch/sd/h/hjia/CO2RR --skip-existing
#
# Batch completion criterion:
#   - directory path matches */XAS/FDMNES/green
#   - four distinct layer IDs contain a usable final *_conv.txt spectrum
#   - each selected spectrum contains >= 50 numeric (energy,intensity) rows
#   - restart/convolution auxiliary outputs are ignored; equal row counts are NOT required
# Existing isaac_record_layers.json files are overwritten by default.
# Use --skip-existing only if you explicitly want to keep an old record.
#
# Expected somewhere below XAS_folder:
#   four *_conv.txt files whose path/name identifies four distinct integer layer IDs.
# Examples accepted by the layer detector include:
#   layer_2, layer2, Layer_2, layer-2, l2 (any integer layer ID is accepted)
#
# CONTCAR lookup:
#   Starting from XAS_folder and walking upward, the script checks both
#     <dir>/CONTCAR
#     <dir>/01_structure/CONTCAR
#   and uses the first one found.
#
# Output default:
#   <XAS_folder>/isaac_record_layers.json

usage() {
    cat >&2 <<EOF
Usage:
  $0 /path/to/XAS/FDMNES/green [output.json]
  $0 --all [/pscratch/sd/h/hjia/CO2RR] [--skip-existing]
EOF
}

echo "convert_layers_to_isaac.sh version ${SCRIPT_VERSION}" >&2

# -----------------------------------------------------------------------------
# Batch mode: find completed layer-resolved FDMNES calculations and invoke this
# same script once per finished green/ directory.
# -----------------------------------------------------------------------------
if [[ "${1:-}" == "--all" ]]; then
    shift
    BATCH_ROOT="/pscratch/sd/h/hjia/CO2RR"
    SKIP_EXISTING=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force)
                # Backward-compatible no-op: overwrite is now the default.
                SKIP_EXISTING=0
                shift
                ;;
            --skip-existing)
                SKIP_EXISTING=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                echo "ERROR: Unknown option in batch mode: $1" >&2
                usage
                exit 2
                ;;
            *)
                BATCH_ROOT="$1"
                shift
                ;;
        esac
    done

    if [[ ! -d "$BATCH_ROOT" ]]; then
        echo "ERROR: Batch root does not exist: $BATCH_ROOT" >&2
        exit 1
    fi
    BATCH_ROOT="$(cd "$BATCH_ROOT" && pwd)"

    # Return success only when a green/ directory has exactly four distinct
    # layer spectra and they look complete.  Equal point counts across layers
    # is a useful guard against a partially written *_conv.txt.
    batch_finished_green() {
        local d="$1"
        python - "$d" <<'PY_BATCH_CHECK'
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

def layer_id(path):
    text = str(path.relative_to(root)).lower()
    pats = [
        r"(?:^|[/_.-])layer[_-]?(\d+)(?:[/_.-]|$)",
        r"(?:^|[/_.-])l[_-]?(\d+)(?:[/_.-]|$)",
        r"layer\s*(\d+)",
    ]
    found = set()
    for pat in pats:
        found.update(int(x) for x in re.findall(pat, text))
    return next(iter(found)) if len(found) == 1 else None

def numeric_rows(path):
    n = 0
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return 0
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(("#", "!", "*", "%")):
            continue
        vals = []
        for token in line.replace(",", " ").split():
            try:
                v = float(token.replace("D", "E").replace("d", "e"))
            except ValueError:
                continue
            if math.isfinite(v):
                vals.append(v)
        if len(vals) >= 2:
            n += 1
    return n

def is_auxiliary_conv(path):
    name = path.name.lower()
    return "resume" in name or "convolution" in name

def preference(path, layer):
    # Prefer the canonical layer-averaged output, e.g. Cu_layer4_conv.txt.
    # Fall back deterministically to another non-auxiliary *_conv.txt only when
    # it is the sole usable candidate for that layer.
    name = path.name.lower()
    canonical = re.fullmatch(rf".*layer[_-]?{layer}_conv\.txt", name) is not None
    return (0 if canonical else 1, len(name), name, str(path))

candidates = {}
for p in sorted(root.rglob("*_conv.txt")):
    if not p.is_file() or is_auxiliary_conv(p):
        continue
    layer = layer_id(p)
    if layer is None:
        continue
    n = numeric_rows(p)
    if n < 50:
        continue
    candidates.setdefault(layer, []).append((p, n))

if len(candidates) != 4:
    print(f"reason=found {len(candidates)} usable layer(s): {sorted(candidates)}", file=sys.stderr)
    raise SystemExit(1)

by_layer = {}
counts = {}
for layer, items in sorted(candidates.items()):
    items.sort(key=lambda item: preference(item[0], layer))
    by_layer[layer], counts[layer] = items[0]

print("layers=" + ",".join(map(str, sorted(by_layer))))
print("points=" + ",".join(f"L{k}:{counts[k]}" for k in sorted(counts)))
PY_BATCH_CHECK
    }

    converted=0
    skipped_existing=0
    skipped_unfinished=0
    failed=0
    found=0

    echo "Scanning for layer-resolved FDMNES calculations below:"
    echo "  $BATCH_ROOT"
    echo

    while IFS= read -r -d '' d; do
        ((++found))
        rel="${d#$BATCH_ROOT/}"
        out="$d/isaac_record_layers.json"

        check_output="$(batch_finished_green "$d" 2>&1)" || {
            echo "SKIP unfinished: $rel  [$check_output]"
            ((++skipped_unfinished))
            continue
        }

        if [[ -f "$out" && "$SKIP_EXISTING" -eq 1 ]]; then
            echo "SKIP existing:   $rel"
            ((++skipped_existing))
            continue
        fi

        echo "CONVERT:         $rel  [$check_output]"
        if "$0" "$d"; then
            ((++converted))
        else
            echo "FAILED:          $rel" >&2
            ((++failed))
        fi
        echo
    done < <(find "$BATCH_ROOT" -type d -path '*/XAS/FDMNES/green' -print0 | sort -z)

    echo "============================================================"
    echo "ISAAC batch conversion summary"
    echo "  green directories found : $found"
    echo "  converted               : $converted"
    echo "  skipped (already exists): $skipped_existing"
    echo "  skipped (unfinished)    : $skipped_unfinished"
    echo "  failed conversion       : $failed"
    echo "============================================================"

    [[ "$failed" -eq 0 ]]
    exit $?
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

XAS_DIR="$(cd "$1" && pwd)"
OUTPUT="${2:-$XAS_DIR/isaac_record_layers.json}"

# Resolve repository root from this script location first, then current folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/tools/convert_simulation_xas_run_to_isaac.py" ]]; then
    REPO_ROOT="$SCRIPT_DIR"
elif [[ -f "$PWD/tools/convert_simulation_xas_run_to_isaac.py" ]]; then
    REPO_ROOT="$PWD"
else
    echo "ERROR: Cannot find tools/convert_simulation_xas_run_to_isaac.py" >&2
    echo "Put this script in the CO2RR_XAS_agent repository root, or run it from there." >&2
    exit 1
fi

# Find the relaxed structure without relying on structure_info.json.
CONTCAR=""
search_dir="$XAS_DIR"
for _ in {1..8}; do
    if [[ -f "$search_dir/CONTCAR" ]]; then
        CONTCAR="$search_dir/CONTCAR"
        break
    fi
    if [[ -f "$search_dir/01_structure/CONTCAR" ]]; then
        CONTCAR="$search_dir/01_structure/CONTCAR"
        break
    fi

    parent="$(dirname "$search_dir")"
    [[ "$parent" == "$search_dir" ]] && break
    search_dir="$parent"
done

if [[ -z "$CONTCAR" ]]; then
    echo "ERROR: Could not find a relaxed CONTCAR for $XAS_DIR" >&2
    echo "Checked parent directories for CONTCAR and 01_structure/CONTCAR." >&2
    exit 1
fi

echo "Using relaxed structure: $CONTCAR"

BASE_CONVERTER="$REPO_ROOT/tools/convert_simulation_xas_run_to_isaac.py"
TMP_JSON="$(mktemp "${TMPDIR:-/tmp}/isaac_base.XXXXXX.json")"
trap 'rm -f "$TMP_JSON"' EXIT

# If replacing an existing local record, preserve its record_id so regeneration
# updates the same scientific record identity instead of creating a duplicate.
EXISTING_RECORD_ID=""
if [[ -f "$OUTPUT" ]]; then
    EXISTING_RECORD_ID="$(python - "$OUTPUT" <<'PY_RECORD_ID'
import json, sys
try:
    rid = json.load(open(sys.argv[1])).get("record_id", "")
    print(rid if isinstance(rid, str) else "")
except Exception:
    print("")
PY_RECORD_ID
)"
    if [[ -n "$EXISTING_RECORD_ID" ]]; then
        echo "Replacing existing record while preserving record_id: $EXISTING_RECORD_ID"
    fi
fi

RECORD_ID_ARGS=()
if [[ -n "$EXISTING_RECORD_ID" ]]; then
    RECORD_ID_ARGS=(--record-id "$EXISTING_RECORD_ID")
fi

# Build the generic ISAAC record using the existing converter.  Force the known
# representation here because the layer container may not itself contain a
# top-level FDMNES input deck from which the converter can infer Cu K-edge.
ISAAC_ABSORBER=Cu \
ISAAC_EDGE=K \
ISAAC_PROCESSING_TYPE=simulated_spectrum_broadened \
python "$BASE_CONVERTER" \
    --input-dir "$XAS_DIR" \
    --output "$TMP_JSON" \
    --record-id "$EXISTING_RECORD_ID"

# Replace the generic sample/structure metadata with values extracted directly
# from CONTCAR, and merge the four layer spectra into measurement.series.
python - "$XAS_DIR" "$CONTCAR" "$TMP_JSON" "$OUTPUT" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

xas_dir = Path(sys.argv[1]).resolve()
contcar = Path(sys.argv[2]).resolve()
base_json = Path(sys.argv[3]).resolve()
out_json = Path(sys.argv[4]).resolve()


def identify_layer(path: Path):
    """Infer an integer layer index from path/name; return None when ambiguous."""
    text = str(path.relative_to(xas_dir)).lower()
    patterns = [
        r"(?:^|[/_.-])layer[_-]?(\d+)(?:[/_.-]|$)",
        r"(?:^|[/_.-])l[_-]?(\d+)(?:[/_.-]|$)",
        r"layer\s*(\d+)",
    ]
    found = set()
    for pat in patterns:
        found.update(int(m) for m in re.findall(pat, text))
    if len(found) == 1:
        return next(iter(found))
    return None


def read_conv(path: Path):
    energy = []
    intensity = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "!", "*", "%")):
            continue
        cols = line.replace(",", " ").split()
        nums = []
        for token in cols:
            try:
                val = float(token)
            except ValueError:
                continue
            if math.isfinite(val):
                nums.append(val)
        if len(nums) >= 2:
            energy.append(nums[0])
            intensity.append(nums[1])
    if len(energy) < 3:
        raise RuntimeError(f"No usable two-column spectrum found in {path}")
    return energy, intensity


def edge_position_first_derivative(x, y):
    best_i = None
    best_d = None
    for i in range(1, min(len(x), len(y))):
        dx = x[i] - x[i - 1]
        if dx == 0:
            continue
        d = (y[i] - y[i - 1]) / dx
        if best_d is None or d > best_d:
            best_d = d
            best_i = i
    return x[best_i] if best_i is not None else None


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _vadd(a, b):
    return [a[i] + b[i] for i in range(3)]


def _vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def _vscale(a, s):
    return [a[i] * s for i in range(3)]


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def _cross(a, b):
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(a):
    return math.sqrt(max(0.0, _dot(a, a)))


def _unit(a):
    n = _norm(a)
    if n == 0:
        raise RuntimeError("Zero-length vector")
    return [x / n for x in a]


def parse_contcar(path: Path):
    """Parse a VASP5/6 POSCAR/CONTCAR, including atomic coordinates."""
    lines = path.read_text(errors="ignore").splitlines()
    if len(lines) < 8:
        raise RuntimeError(f"Structure file is too short: {path}")

    comment = lines[0].strip()
    try:
        raw_scale = float(lines[1].split()[0])
        lattice_raw = [[float(v) for v in lines[i].split()[:3]] for i in range(2, 5)]
    except Exception as exc:
        raise RuntimeError(f"Could not parse lattice from {path}: {exc}") from exc

    raw_volume = abs(_dot(lattice_raw[0], _cross(lattice_raw[1], lattice_raw[2])))
    if raw_scale < 0:
        if raw_volume <= 0:
            raise RuntimeError(f"Invalid lattice volume in {path}")
        scale = (abs(raw_scale) / raw_volume) ** (1.0 / 3.0)
    else:
        scale = raw_scale
    lattice = [[scale * v for v in vec] for vec in lattice_raw]

    symbols = lines[5].split()
    if not symbols or all(re.fullmatch(r"[+-]?\d+", token) for token in symbols):
        raise RuntimeError(
            "Structure file does not contain VASP5/6 element symbols on line 6; "
            "cannot infer composition reliably."
        )
    try:
        counts = [int(v) for v in lines[6].split()]
    except Exception as exc:
        raise RuntimeError(f"Could not parse atom counts from {path}: {exc}") from exc
    if len(symbols) != len(counts):
        raise RuntimeError(
            f"Species/count mismatch in {path}: {len(symbols)} species but {len(counts)} counts"
        )

    n_atoms = sum(counts)
    coord_i = 7
    if lines[coord_i].strip().lower().startswith("s"):
        coord_i += 1
    mode = lines[coord_i].strip().lower()
    direct = mode.startswith("d")
    cartesian = mode.startswith(("c", "k"))
    if not (direct or cartesian):
        raise RuntimeError(f"Unknown coordinate mode '{lines[coord_i]}' in {path}")
    coord_i += 1
    if len(lines) < coord_i + n_atoms:
        raise RuntimeError(f"Not enough coordinate lines in {path}")

    elements = []
    for symbol, count in zip(symbols, counts):
        elements.extend([symbol] * count)

    frac_positions = []
    cart_positions = []
    for atom_i in range(n_atoms):
        fields = lines[coord_i + atom_i].split()
        if len(fields) < 3:
            raise RuntimeError(f"Malformed coordinate line {coord_i + atom_i + 1} in {path}")
        xyz = [float(fields[j]) for j in range(3)]
        if direct:
            frac = xyz
            cart = [
                frac[0] * lattice[0][j] + frac[1] * lattice[1][j] + frac[2] * lattice[2][j]
                for j in range(3)
            ]
        else:
            cart = [scale * x for x in xyz]
            frac = None
        frac_positions.append(frac)
        cart_positions.append(cart)

    composition = dict(zip(symbols, counts))
    formula = "".join(symbol if count == 1 else f"{symbol}{count}" for symbol, count in zip(symbols, counts))
    volume = abs(_dot(lattice[0], _cross(lattice[1], lattice[2])))
    parent = path.parent
    sample_name = parent.parent.name if parent.name == "01_structure" else parent.name

    return {
        "comment": comment,
        "symbols": symbols,
        "counts": counts,
        "composition": composition,
        "formula": formula,
        "n_atoms": n_atoms,
        "lattice_vectors_A": lattice,
        "cell_volume_A3": volume,
        "sample_name": sample_name,
        "elements": elements,
        "cart_positions_A": cart_positions,
        "frac_positions": frac_positions,
        "coordinate_mode": "direct" if direct else "cartesian",
    }


def parse_structure_path_labels(contcar_path: Path):
    """Parse identity hints and spatial-distribution labels from the dataset path.

    Folder names are used for *distribution/region* semantics only.  Atomic site
    geometry (atop/bridge/fcc-hollow/hcp-hollow) is determined independently
    from the relaxed CONTCAR coordinates.
    """
    structure_dir = contcar_path.parent.parent if contcar_path.parent.name == "01_structure" else contcar_path.parent

    rel_parts = None
    cur = structure_dir
    while True:
        if cur.name == "CO2RR":
            rel_parts = list(structure_dir.relative_to(cur).parts)
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    if rel_parts is None:
        rel_parts = list(structure_dir.parts[-3:])

    relative_structure_path = "/".join(rel_parts)
    coverage_re = re.compile(r"(?<![A-Za-z0-9])0p(\d+)(?![A-Za-z0-9])", re.IGNORECASE)
    coverage_part_index = None
    coverage_token = None
    identity_raw = None
    suffix_parts = []
    for idx in range(len(rel_parts) - 1, -1, -1):
        part = rel_parts[idx]
        m = coverage_re.search(part)
        if not m:
            continue
        coverage_part_index = idx
        coverage_token = m.group(0)
        prefix = part[:m.start()].strip("_- ")
        suffix = part[m.end():].strip("_- ")
        identity_raw = prefix or None
        if suffix:
            suffix_parts.append(suffix)
        suffix_parts.extend(rel_parts[idx + 1:])
        break

    category = rel_parts[0] if rel_parts else None
    if not identity_raw and coverage_part_index is not None and coverage_part_index > 0:
        identity_raw = rel_parts[coverage_part_index - 1]
    if not identity_raw:
        identity_raw = category

    config_label = "/".join(p for p in suffix_parts if p) or None
    text = (config_label or "").lower().replace("-", "_")

    # Distribution means how multiple adsorbates are arranged; it is NOT the
    # adsorption site geometry.  Keep both a canonical class and the exact label.
    distribution = None
    if "uniform" in text or "even" in text:
        distribution = "uniform"
    elif "interface_biased" in text or "interface_weighted" in text:
        distribution = "interface_biased"
    elif "interior_biased" in text or "interior_weighted" in text:
        distribution = "interior_biased"
    elif "adjacent" in text:
        distribution = "adjacent"
    elif "separated" in text or "separate" in text:
        distribution = "separated"
    elif "interface" in text:
        distribution = "interface"

    # Region is a coarse lateral location and remains separate from site geometry.
    region = None
    if "aucu" in text and "interface" in text:
        region = "AuCu_interface"
    elif "interface" in text or "boundary" in text:
        region = "interface"
    elif re.search(r"(?:^|[_/])au(?:$|[_/])", text):
        region = "Au"
    elif re.search(r"(?:^|[_/])cu(?:$|[_/])", text):
        region = "Cu"
    elif "interior" in text:
        region = "interior"
    elif "terrace" in text:
        region = "terrace"

    # Adsorbate identity hints. Prefer the top-level dataset category when it is
    # chemically informative because names such as CHO_CO and OCCO_CO encode mixtures.
    identity_hint = identity_raw
    if category and not coverage_re.search(category) and category.lower() not in {"plain", "cuo", "cu2o"}:
        identity_hint = category
    identity_hint = re.sub(r"_plus_", "+", identity_hint or "", flags=re.IGNORECASE).strip("_ ") or None

    return {
        "identity_hint": identity_hint,
        "category": category,
        "distribution": distribution,
        "distribution_label": config_label,
        "region": region,
        "coverage_token": coverage_token,
        "relative_structure_path": relative_structure_path,
    }


KNOWN_ADSORBATES = {
    "H": {"H": 1},
    "O": {"O": 1},
    "OH": {"O": 1, "H": 1},
    "CO": {"C": 1, "O": 1},
    "CO2": {"C": 1, "O": 2},
    "CHO": {"C": 1, "H": 1, "O": 1},
    "COH": {"C": 1, "H": 1, "O": 1},
    "OCHO": {"C": 1, "H": 1, "O": 2},
    "COOH": {"C": 1, "H": 1, "O": 2},
    "OCCO": {"C": 2, "O": 2},
    "COCO": {"C": 2, "O": 2},
    "OCCOH": {"C": 2, "H": 1, "O": 2},
    "COCHO": {"C": 2, "H": 1, "O": 2},
    "CH3": {"C": 1, "H": 3},
}


def _formula_dict(elements):
    out = {}
    for el in elements:
        out[el] = out.get(el, 0) + 1
    return out


def _formula_string(comp):
    order = []
    if "C" in comp:
        order.append("C")
    if "H" in comp:
        order.append("H")
    order.extend(sorted(k for k in comp if k not in {"C", "H"}))
    return "".join(k if comp[k] == 1 else f"{k}{comp[k]}" for k in order)


def _expected_species_from_path(path_labels):
    hint = path_labels.get("identity_hint") or ""
    raw = hint.replace("+", "_")
    raw = re.sub(r"_(adjacent|uniform|interface.*|interior.*|terrace.*)$", "", raw, flags=re.IGNORECASE)

    # Explicit leading multiplicity, e.g. 2CO_adjacent.
    m = re.match(r"^(\d+)(CO2|CO|CHO|COH|OCCOH|OCCO|COCO|COCHO|COOH|OCHO|OH|H|O)(?:_|$)", raw, re.IGNORECASE)
    if m:
        species = m.group(2).upper().replace("COH", "COH")
        # restore mixed-case-free canonical spelling by lookup
        species = next((k for k in KNOWN_ADSORBATES if k.upper() == species), species)
        return [species], {species: int(m.group(1))}

    if raw in KNOWN_ADSORBATES:
        return [raw], {}

    parts = [p for p in raw.split("_") if p]
    if parts and all(p in KNOWN_ADSORBATES for p in parts):
        return parts, {}

    # Try exact known names contained in a + expression.
    plus_parts = [p for p in hint.split("+") if p]
    if plus_parts and all(p in KNOWN_ADSORBATES for p in plus_parts):
        return plus_parts, {}

    return ([raw] if raw in KNOWN_ADSORBATES else []), {}


def _periodic_delta_xy(r1, r2, lattice, normal):
    """Minimum lateral vector r1-r2 using periodic translations in slab a/b."""
    best = None
    best_norm = None
    a, b = lattice[0], lattice[1]
    base = _vsub(r1, r2)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            d = _vadd(base, _vadd(_vscale(a, i), _vscale(b, j)))
            lateral = _vsub(d, _vscale(normal, _dot(d, normal)))
            n = _norm(lateral)
            if best_norm is None or n < best_norm:
                best, best_norm = lateral, n
    return best, best_norm


def _periodic_distance(r1, r2, lattice):
    best = None
    a, b = lattice[0], lattice[1]
    base = _vsub(r1, r2)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            d = _vadd(base, _vadd(_vscale(a, i), _vscale(b, j)))
            n = _norm(d)
            if best is None or n < best:
                best = n
    return best


def _surface_layers(structure, metal_elements={"Cu", "Au"}):
    lattice = structure["lattice_vectors_A"]
    normal = _unit(_cross(lattice[0], lattice[1]))
    metals = []
    for idx, (el, pos) in enumerate(zip(structure["elements"], structure["cart_positions_A"])):
        if el in metal_elements:
            metals.append({"index": idx, "element": el, "pos": pos, "height": _dot(pos, normal)})
    if not metals:
        return normal, []
    metals.sort(key=lambda x: x["height"], reverse=True)
    layers = []
    for atom in metals:
        if not layers:
            layers.append([atom])
            continue
        mean_h = sum(x["height"] for x in layers[-1]) / len(layers[-1])
        if abs(atom["height"] - mean_h) <= 0.80:
            layers[-1].append(atom)
        else:
            layers.append([atom])
    return normal, layers


def _top_nn_distance(top_layer, lattice, normal):
    vals = []
    for i, ai in enumerate(top_layer):
        best = None
        for j, aj in enumerate(top_layer):
            if i == j:
                continue
            _, d = _periodic_delta_xy(ai["pos"], aj["pos"], lattice, normal)
            if d > 1e-5 and (best is None or d < best):
                best = d
        if best is not None:
            vals.append(best)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def _adsorbate_components(structure, metal_elements={"Cu", "Au"}):
    ads = [
        {"index": i, "element": el, "pos": pos}
        for i, (el, pos) in enumerate(zip(structure["elements"], structure["cart_positions_A"]))
        if el not in metal_elements
    ]
    if not ads:
        return []
    radii = {"H": 0.31, "C": 0.76, "O": 0.66, "N": 0.71, "S": 1.05}
    adjacency = {a["index"]: set() for a in ads}
    by_index = {a["index"]: a for a in ads}
    for ii in range(len(ads)):
        for jj in range(ii + 1, len(ads)):
            a, b = ads[ii], ads[jj]
            rsum = radii.get(a["element"], 0.75) + radii.get(b["element"], 0.75)
            cutoff = 1.25 * rsum + 0.10
            if _periodic_distance(a["pos"], b["pos"], structure["lattice_vectors_A"]) <= cutoff:
                adjacency[a["index"]].add(b["index"])
                adjacency[b["index"]].add(a["index"])
    components = []
    seen = set()
    for atom in ads:
        root = atom["index"]
        if root in seen:
            continue
        stack = [root]
        comp = []
        seen.add(root)
        while stack:
            cur = stack.pop()
            comp.append(by_index[cur])
            for nb in adjacency[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(comp)
    return components


def _hydrogen_attachment(comp, lattice):
    """Return C or O when a component's H is covalently attached to one unambiguously."""
    hs = [a for a in comp if a["element"] == "H"]
    heavy = [a for a in comp if a["element"] in {"C", "O"}]
    if len(hs) != 1 or not heavy:
        return None
    h = hs[0]
    # Covalent cutoffs chosen to distinguish an internal X-H bond from separate
    # coadsorbed H on the metal surface.
    bonded = []
    for a in heavy:
        d = _periodic_distance(h["pos"], a["pos"], lattice)
        cutoff = 1.25 if a["element"] == "O" else 1.30
        if d <= cutoff:
            bonded.append((d, a["element"]))
    if not bonded:
        return None
    bonded.sort()
    return bonded[0][1]


def _species_from_structure(comp, formula, structure, expected_unique):
    """Classify from relaxed composition/connectivity; path only breaks true formula ties."""
    matching_known = [sp for sp, f in KNOWN_ADSORBATES.items() if f == formula]

    # Formula is already unique: the structure is authoritative.
    if len(matching_known) == 1:
        return matching_known[0]

    # Same-formula protonation/isomer pairs are resolved by where H is bonded.
    attachment = _hydrogen_attachment(comp, structure["lattice_vectors_A"])
    if formula == {"C": 1, "H": 1, "O": 1}:
        if attachment == "C":
            return "CHO"
        if attachment == "O":
            return "COH"
    if formula == {"C": 1, "H": 1, "O": 2}:
        if attachment == "C":
            return "OCHO"
        if attachment == "O":
            return "COOH"
    if formula == {"C": 2, "H": 1, "O": 2}:
        if attachment == "C":
            return "COCHO"
        if attachment == "O":
            return "OCCOH"

    # For formula-isomeric species without an H attachment discriminator
    # (e.g. OCCO vs COCO), the directory chemistry may disambiguate, but only
    # when its formula agrees with the relaxed component.
    compatible_path = [sp for sp in expected_unique if sp in matching_known]
    if len(compatible_path) == 1:
        return compatible_path[0]

    # Do not invent an identity when the relaxed geometry cannot distinguish it.
    return _formula_string(formula)


def _assign_component_species(components, expected_species, multiplicity_hints, structure):
    expected_unique = list(dict.fromkeys(expected_species))
    assigned = []
    for comp in components:
        formula = _formula_dict([a["element"] for a in comp])
        label = _species_from_structure(comp, formula, structure, expected_unique)

        # A path hint is NEVER allowed to override an incompatible relaxed formula.
        if label in KNOWN_ADSORBATES and KNOWN_ADSORBATES[label] != formula:
            raise RuntimeError(
                f"Internal adsorbate classification mismatch: {label} has formula "
                f"{KNOWN_ADSORBATES[label]} but relaxed component is {formula}"
            )
        assigned.append((label, formula, comp))

    counts = {}
    for label, _, _ in assigned:
        counts[label] = counts.get(label, 0) + 1

    hint_status = "not_available"
    if multiplicity_hints:
        hint_status = "match" if all(counts.get(k) == v for k, v in multiplicity_hints.items()) else "mismatch"
    return assigned, counts, hint_status


def _path_identity_status(expected_species, counts):
    expected = set(expected_species)
    actual = set(counts)
    if not expected:
        return "not_available"
    return "match" if expected == actual else f"mismatch:path={sorted(expected)};CONTCAR={sorted(actual)}"


def _classify_site(anchor_pos, layers, lattice, normal):
    if not layers or not layers[0]:
        return {"site_type": "unresolved", "site_metals": [], "local_surface_environment": "unresolved"}
    top = layers[0]
    dnn = _top_nn_distance(top, lattice, normal)
    if not dnn or dnn <= 0:
        return {"site_type": "unresolved", "site_metals": [], "local_surface_environment": "unresolved"}

    dists = []
    for atom in top:
        _, d = _periodic_delta_xy(anchor_pos, atom["pos"], lattice, normal)
        dists.append((d, atom))
    dists.sort(key=lambda x: x[0])
    nearest = dists[:3]
    ds = [x[0] for x in nearest]

    site_type = "off_high_symmetry"
    nsite = 1
    if ds and ds[0] <= 0.28 * dnn:
        site_type = "atop"
        nsite = 1
    elif len(ds) >= 3 and ds[1] <= 0.64 * dnn and ds[2] >= 0.72 * dnn and abs(ds[1] - ds[0]) <= 0.25 * dnn:
        site_type = "bridge"
        nsite = 2
    elif len(ds) >= 3 and ds[2] <= 0.78 * dnn and (ds[2] - ds[0]) <= 0.30 * dnn:
        site_type = "hollow"
        nsite = 3
        # hcp hollow projects onto a second-layer atom; fcc hollow onto a third-layer atom.
        second_d = None
        third_d = None
        if len(layers) > 1:
            second_d = min(_periodic_delta_xy(anchor_pos, a["pos"], lattice, normal)[1] for a in layers[1])
        if len(layers) > 2:
            third_d = min(_periodic_delta_xy(anchor_pos, a["pos"], lattice, normal)[1] for a in layers[2])
        tol = 0.32 * dnn
        if second_d is not None and second_d <= tol and (third_d is None or second_d <= third_d):
            site_type = "hcp_hollow"
        elif third_d is not None and third_d <= tol:
            site_type = "fcc_hollow"

    site_metals = [x[1]["element"] for x in nearest[:nsite]]
    local3 = [x[1]["element"] for x in nearest[:3]]
    env_counts = {}
    for el in local3:
        env_counts[el] = env_counts.get(el, 0) + 1
    local_env = "".join(f"{el}{env_counts[el]}" for el in ("Cu", "Au") if env_counts.get(el)) or "unresolved"
    return {
        "site_type": site_type,
        "site_metals": site_metals,
        "local_surface_environment": local_env,
        "nearest_surface_distance_A": round(ds[0], 4) if ds else None,
    }


def analyze_adsorbates(structure, path_labels, poscar_structure=None):
    """Count adsorbates and infer relaxed adsorption sites from atomic geometry."""
    expected, multiplicity_hints = _expected_species_from_path(path_labels)
    components = _adsorbate_components(structure)
    if not components:
        return {
            "adsorbate_counts": {},
            "adsorbate_count_total": 0,
            "adsorption_sites": [],
            "adsorption_site_counts": {},
            "path_count_hint_status": "not_applicable",
            "path_identity_hint_status": "not_applicable",
            "poscar_count_crosscheck": "not_applicable",
        }
    assigned, counts, hint_status = _assign_component_species(components, expected, multiplicity_hints, structure)

    normal, layers = _surface_layers(structure)
    top_h = None
    if layers and layers[0]:
        top_h = sum(a["height"] for a in layers[0]) / len(layers[0])

    sites = []
    per_species_index = {}
    for label, formula, comp in assigned:
        # The atom closest vertically to the slab is the adsorption anchor.
        if top_h is not None:
            comp_sorted = sorted(comp, key=lambda a: abs(_dot(a["pos"], normal) - top_h))
        else:
            comp_sorted = list(comp)
        anchor = comp_sorted[0]
        site = _classify_site(anchor["pos"], layers, structure["lattice_vectors_A"], normal)
        per_species_index[label] = per_species_index.get(label, 0) + 1
        item = {
            "adsorbate": label,
            "index": per_species_index[label],
            "formula": _formula_string(formula),
            "anchor_element": anchor["element"],
            **site,
        }
        if top_h is not None:
            item["anchor_height_above_top_metal_A"] = round(_dot(anchor["pos"], normal) - top_h, 4)
        sites.append(item)

    site_counts = {}
    site_species_counts = {}
    for item in sites:
        st = item["site_type"]
        site_counts[st] = site_counts.get(st, 0) + 1
        sp = item["adsorbate"]
        site_species_counts[sp] = site_species_counts.get(sp, 0) + 1

    # Hard consistency gate: the aggregate identity/count block and the per-site
    # entries are two views of the same relaxed adsorbates and must agree exactly.
    if site_species_counts != counts:
        raise RuntimeError(
            f"Adsorbate identity/count inconsistency: counts={counts}, "
            f"site_entries={site_species_counts}"
        )
    if sum(counts.values()) != len(sites):
        raise RuntimeError(
            f"Adsorbate total/count inconsistency: total={sum(counts.values())}, "
            f"site_entries={len(sites)}"
        )

    # POSCAR is used as an independent composition/count cross-check.  CONTCAR
    # remains the site source because it is the relaxed geometry.
    poscar_crosscheck = "POSCAR_not_found"
    if poscar_structure is not None:
        pos_components = _adsorbate_components(poscar_structure)
        _, pos_counts, _ = _assign_component_species(pos_components, expected, multiplicity_hints, poscar_structure)
        poscar_crosscheck = "match" if pos_counts == counts else f"mismatch:POSCAR={pos_counts};CONTCAR={counts}"

    return {
        "adsorbate_counts": counts,
        "adsorbate_count_total": sum(counts.values()),
        "adsorption_sites": sites,
        "adsorption_site_counts": site_counts,
        "path_count_hint_status": hint_status,
        "path_identity_hint_status": _path_identity_status(expected, counts),
        "poscar_count_crosscheck": poscar_crosscheck,
    }


FDMNES_VALUE_KEYS = {
    "filout": "filout",
    "range": "energy_range",
    "radius": "radius_A",
    "z_absorber": "z_absorber",
    "zabsorber": "z_absorber",
    "edge": "edge",
    "efermi": "efermi_eV",
    "rpotmax": "rpotmax_A",
    "gamma_hole": "gamma_hole",
    "gamma_max": "gamma_max",
    "ecent": "ecent",
    "elarg": "elarg",
    "eimag": "eimag",
    "lmax": "lmax",
}

FDMNES_FLAG_KEYS = {
    "green": "green",
    "scf": "scf",
    "energpho": "energpho",
    "memory_save": "memory_save",
    "perdew": "perdew",
    "convolution": "convolution",
    "quadrupole": "quadrupole",
    "spinorbit": "spinorbit",
    "tddft": "tddft",
    "nonexc": "nonexc",
    "notscf": "notscf",
}


def _strip_fdmnes_comment(line: str) -> str:
    # FDMNES examples commonly use !; tolerate # as well.
    return line.split("!", 1)[0].split("#", 1)[0].strip()


def _coerce_fdmnes_value(text: str):
    tokens = text.replace(",", " ").split()
    if not tokens:
        return ""
    vals = []
    all_numeric = True
    for token in tokens:
        try:
            value = float(token)
            if value.is_integer() and not any(c in token.lower() for c in (".", "e")):
                value = int(value)
            vals.append(value)
        except ValueError:
            all_numeric = False
            break
    if all_numeric:
        return vals[0] if len(vals) == 1 else vals
    return text.strip()


def parse_fdmnes_input(path: Path):
    """Extract reusable calculation settings from one FDMNES *_in.txt file.

    This intentionally summarizes calculation controls and does not duplicate the
    Crystal coordinate block; the relaxed geometry is already represented by CONTCAR.
    """
    raw_lines = path.read_text(errors="ignore").splitlines()
    lines = [_strip_fdmnes_comment(line) for line in raw_lines]
    settings = {}

    known_keys = set(FDMNES_VALUE_KEYS) | set(FDMNES_FLAG_KEYS)
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        parts = line.split()
        key_raw = parts[0].lower().replace("-", "_")
        key_raw = re.sub(r"[^a-z0-9_]", "", key_raw)

        if key_raw in FDMNES_FLAG_KEYS:
            settings[FDMNES_FLAG_KEYS[key_raw]] = True
            i += 1
            continue

        if key_raw in FDMNES_VALUE_KEYS:
            inline = " ".join(parts[1:]).strip()
            value_text = inline
            if not value_text:
                j = i + 1
                while j < len(lines) and not lines[j]:
                    j += 1
                if j < len(lines):
                    next_parts = lines[j].split()
                    next_key = next_parts[0].lower().replace("-", "_") if next_parts else ""
                    next_key = re.sub(r"[^a-z0-9_]", "", next_key)
                    if next_key not in known_keys and next_key not in {"crystal", "end"}:
                        value_text = lines[j]
                        i = j
            if value_text:
                settings[FDMNES_VALUE_KEYS[key_raw]] = _coerce_fdmnes_value(value_text)
            i += 1
            continue

        i += 1

    # Add an explicit solver summary from the actual input keyword.
    if settings:
        settings["calculation_mode"] = (
            "green_function_multiple_scattering" if settings.get("green")
            else "finite_difference"
        )
    return settings


def is_fdmnes_convolution_input(path: Path) -> bool:
    """Identify auxiliary convolution input decks, not the main FDMNES calculation deck."""
    name = path.name.lower()
    return (
        "convolution" in name
        or name.endswith("_conv_in.txt")
        or name.endswith("-conv_in.txt")
    )


def is_fdmnes_restart_input(path: Path) -> bool:
    """Identify generated restart/resume decks, not the canonical main input.

    Restart files are useful provenance assets, but their absorber list may be a
    subset of the original calculation.  They must therefore never replace or
    compete with the original layer input when reconstructing calculation setup.
    """
    name = path.name.lower()
    stem = name[:-7] if name.endswith("_in.txt") else name
    tokens = stem.replace("-", "_").split("_")
    return any(tok in {"resume", "restart", "rerun", "retry"} for tok in tokens)


def is_fdmnes_auxiliary_input(path: Path) -> bool:
    return is_fdmnes_convolution_input(path) or is_fdmnes_restart_input(path)


def collect_fdmnes_layer_inputs():
    """Return one canonical main FDMNES *_in.txt file per layer when available.

    Auxiliary convolution and restart/resume decks are deliberately excluded.
    They remain available as record assets, but only the original calculation
    input is used to reconstruct the layer setup.
    """
    files_by_layer = {}
    unassigned = []
    for path in sorted(p for p in xas_dir.rglob("*_in.txt") if p.is_file()):
        if is_fdmnes_auxiliary_input(path):
            continue
        layer = identify_layer(path)
        if layer is None:
            unassigned.append(path)
            continue
        files_by_layer.setdefault(layer, []).append(path)

    selected = {}
    for layer, paths in files_by_layer.items():
        if len(paths) == 1:
            selected[layer] = paths[0]
            continue
        # Prefer the main input in the same directory as the final spectrum.
        same_dir = [p for p in paths if layer in layer_files and p.parent == layer_files[layer].parent]
        if len(same_dir) == 1:
            selected[layer] = same_dir[0]
        else:
            raise SystemExit(
                f"ERROR: More than one main FDMNES *_in.txt file found for layer {layer}:\\n"
                + "\\n".join(f"  {p}" for p in paths)
            )
    return selected, unassigned


def collect_fdmnes_convolution_inputs():
    """Return at most one auxiliary FDMNES convolution input deck per layer."""
    files_by_layer = {}
    unassigned = []
    for path in sorted(p for p in xas_dir.rglob("*_in.txt") if p.is_file()):
        if not is_fdmnes_convolution_input(path):
            continue
        layer = identify_layer(path)
        if layer is None:
            unassigned.append(path)
            continue
        files_by_layer.setdefault(layer, []).append(path)

    selected = {}
    for layer, paths in files_by_layer.items():
        if len(paths) == 1:
            selected[layer] = paths[0]
            continue
        same_dir = [p for p in paths if layer in layer_files and p.parent == layer_files[layer].parent]
        if len(same_dir) == 1:
            selected[layer] = same_dir[0]
        else:
            raise SystemExit(
                f"ERROR: More than one FDMNES convolution input file found for layer {layer}:\\n"
                + "\\n".join(f"  {p}" for p in paths)
            )
    return selected, unassigned


def common_and_layer_specific(layer_settings):
    """Split parsed settings into values common to every parsed layer and differences."""
    if not layer_settings:
        return {}, {}
    layers = sorted(layer_settings)
    common_keys = set(layer_settings[layers[0]])
    for layer in layers[1:]:
        common_keys &= set(layer_settings[layer])
    common = {}
    for key in sorted(common_keys):
        first = layer_settings[layers[0]][key]
        if all(layer_settings[layer][key] == first for layer in layers[1:]):
            common[key] = first
    specific = {
        f"layer_{layer}": {
            key: value for key, value in settings.items() if common.get(key) != value
        }
        for layer, settings in sorted(layer_settings.items())
    }
    specific = {key: value for key, value in specific.items() if value}
    return common, specific


def read_timestamp_metadata():
    """Use only explicitly recorded acquisition timestamps; never invent them."""
    starts = []
    ends = []
    sources = []

    # Explicit environment values are valid provenance if supplied by the user/job.
    if os.environ.get("ISAAC_START_UTC"):
        starts.append(os.environ["ISAAC_START_UTC"])
        sources.append("ISAAC_START_UTC")
    if os.environ.get("ISAAC_END_UTC"):
        ends.append(os.environ["ISAAC_END_UTC"])
        sources.append("ISAAC_END_UTC")

    # Search root and layer directories for actual recorded run metadata.
    candidates = [xas_dir / "isaac_run_metadata.json"]
    candidates.extend(sorted(xas_dir.rglob("isaac_run_metadata.json")))
    seen = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        ts = data.get("timestamps") if isinstance(data, dict) else None
        if not isinstance(ts, dict):
            continue
        if ts.get("acquired_start_utc"):
            starts.append(str(ts["acquired_start_utc"]))
            sources.append(str(path.relative_to(xas_dir)))
        if ts.get("acquired_end_utc"):
            ends.append(str(ts["acquired_end_utc"]))
            sources.append(str(path.relative_to(xas_dir)))

    result = {}
    if starts:
        result["acquired_start_utc"] = min(starts)
    if ends:
        result["acquired_end_utc"] = max(ends)
    return result, sorted(set(sources))


# Select one usable canonical final spectrum for each of the four layers.
# Restart/convolution auxiliary outputs are provenance assets, not competing
# layer spectra. Different layers may legitimately contain different point counts.
def is_auxiliary_conv(path):
    name = path.name.lower()
    return "resume" in name or "convolution" in name

def spectrum_numeric_rows(path):
    n = 0
    try:
        for raw in path.read_text(errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith(("#", "!", "*", "%")):
                continue
            vals = []
            for token in line.replace(",", " ").split():
                try:
                    v = float(token.replace("D", "E").replace("d", "e"))
                except ValueError:
                    continue
                if math.isfinite(v):
                    vals.append(v)
            if len(vals) >= 2:
                n += 1
    except Exception:
        return 0
    return n

def spectrum_preference(path, layer):
    name = path.name.lower()
    canonical = re.fullmatch(rf".*layer[_-]?{layer}_conv\.txt", name) is not None
    return (0 if canonical else 1, len(name), name, str(path))

conv_files = sorted(
    p for p in xas_dir.rglob("*_conv.txt")
    if p.is_file() and not is_auxiliary_conv(p)
)
if not conv_files:
    raise SystemExit(f"ERROR: No usable final *_conv.txt files found below {xas_dir}")

layer_candidates = {}
unassigned = []
for path in conv_files:
    layer = identify_layer(path)
    if layer is None:
        unassigned.append(path)
        continue
    nrows = spectrum_numeric_rows(path)
    if nrows < 50:
        continue
    layer_candidates.setdefault(layer, []).append(path)

layer_files = {}
for layer, paths in sorted(layer_candidates.items()):
    paths.sort(key=lambda p: spectrum_preference(p, layer))
    layer_files[layer] = paths[0]
    if len(paths) > 1:
        print(
            f"WARNING: Multiple usable spectra found for layer {layer}; "
            f"using {paths[0].relative_to(xas_dir)}",
            file=sys.stderr,
        )
        for extra in paths[1:]:
            print(f"  ignored: {extra.relative_to(xas_dir)}", file=sys.stderr)

layers = sorted(layer_files)
if len(layers) != 4:
    msg = [f"ERROR: Expected four usable layer-resolved spectra; found {len(layers)} layer(s): {layers}"]
    msg.append("Recognized usable files:")
    for k in layers:
        msg.append(f"  layer {k}: {layer_files[k].relative_to(xas_dir)}")
    if unassigned:
        msg.append("Unassigned *_conv.txt files:")
        msg.extend(f"  {p.relative_to(xas_dir)}" for p in unassigned)
    msg.append("Layer numbers do not need to start at 1; e.g. layer_2 through layer_5 is valid.")
    raise SystemExit("\n".join(msg))

if unassigned:
    print("WARNING: Ignoring unassigned *_conv.txt files:", file=sys.stderr)
    for p in unassigned:
        print(f"  {p.relative_to(xas_dir)}", file=sys.stderr)

# Parse FDMNES setup from nested layer input decks when they exist.  The
# conversion still works if only CONTCAR + four *_conv.txt files are present.
layer_input_files, unassigned_input_files = collect_fdmnes_layer_inputs()
layer_input_settings = {
    layer: parse_fdmnes_input(path)
    for layer, path in sorted(layer_input_files.items())
}
common_fdmnes_settings, layer_specific_fdmnes_settings = common_and_layer_specific(layer_input_settings)

# FDMNES can write a second input deck used only for spectral convolution, e.g.
# Cu_layer4_convolution_in.txt.  Keep it distinct from the main calculation
# input while still capturing broadening/convolution parameters when present.
convolution_input_files, unassigned_convolution_files = collect_fdmnes_convolution_inputs()
convolution_input_settings = {}
for layer, path in sorted(convolution_input_files.items()):
    settings = parse_fdmnes_input(path)
    # A convolution-only deck is not a solver choice. parse_fdmnes_input adds a
    # default calculation_mode for any non-empty settings, so remove it here.
    settings.pop("calculation_mode", None)
    convolution_input_settings[layer] = settings
common_convolution_settings, layer_specific_convolution_settings = common_and_layer_specific(convolution_input_settings)

# Sanity-check the parsed input deck against this dataset definition.
for layer, settings in layer_input_settings.items():
    z = settings.get("z_absorber")
    if z not in (None, 29, 29.0, "29"):
        raise SystemExit(f"ERROR: layer {layer} input reports Z_Absorber={z}, expected Cu (29)")
    edge = str(settings.get("edge", "K")).upper()
    if edge and edge != "K":
        raise SystemExit(f"ERROR: layer {layer} input reports Edge={edge}, expected K")

record = json.loads(base_json.read_text())
structure = parse_contcar(contcar)
path_labels = parse_structure_path_labels(contcar)

# Cross-check adsorbate counts against the initial POSCAR when available, while
# using relaxed CONTCAR coordinates for the actual adsorption-site assignment.
structure_dir = contcar.parent.parent if contcar.parent.name == "01_structure" else contcar.parent
poscar_candidates = [
    contcar.parent / "POSCAR",
    structure_dir / "POSCAR",
    structure_dir / "01_structure" / "POSCAR",
]
poscar = next((p for p in poscar_candidates if p.is_file()), None)
poscar_structure = None
if poscar is not None:
    try:
        poscar_structure = parse_contcar(poscar)
    except Exception as exc:
        print(f"WARNING: Could not parse POSCAR cross-check {poscar}: {exc}", file=sys.stderr)

adsorbate_info = analyze_adsorbates(structure, path_labels, poscar_structure)

# Keep the record creation time, but do not fabricate acquisition times.
# If real timestamps were captured in job metadata (or explicitly supplied via
# ISAAC_START_UTC/ISAAC_END_UTC), preserve those; otherwise omit them.
recorded_acquisition_times, timestamp_sources = read_timestamp_metadata()
timestamps = record.setdefault("timestamps", {})
created_utc = timestamps.get("created_utc")
timestamps.clear()
if created_utc:
    timestamps["created_utc"] = created_utc
timestamps.update(recorded_acquisition_times)

# -------------------------------------------------------------------------
# Common structure metadata: derive only what CONTCAR directly supports.
# Do not preserve semantic structure_info.json fields from the base converter.
# -------------------------------------------------------------------------
sample = record.setdefault("sample", {})
for key in (
    "configuration_id",
    "parent_structure_id",
    "catalyst",
    "adsorbate",
    "ml_labels",
    "structure_descriptors",
    "structure_id",
):
    sample.pop(key, None)

sample["material"] = {
    "name": structure["sample_name"],
    "formula": structure["formula"],
    "provenance": "theoretical",
    "notes": (
        "Relaxed structure metadata extracted directly from CONTCAR. "
        "Adsorbate identities/counts are derived from relaxed CONTCAR composition/connectivity; path labels only disambiguate formula-isomers and never override incompatible structure chemistry. Counts are cross-checked "
        "against POSCAR when available. Adsorption site geometry is assigned from relaxed "
        "CONTCAR coordinates; spatial distribution labels are parsed separately from the "
        "CO2RR dataset directory naming convention."
    ),
    "identifiers": [{
        "scheme": "internal",
        "value": path_labels["relative_structure_path"],
    }],
}
sample["sample_form"] = "slab_model"
# ISAAC v1.05 provides open sample.composition and sample.geometry objects.
# Store only quantities that CONTCAR directly supports; the full coordinates stay
# in the CONTCAR asset below.
sample["composition"] = {
    "atom_counts": structure["composition"],
    "total_atoms": structure["n_atoms"],
}
sample["geometry"] = {
    "lattice_vectors_A": structure["lattice_vectors_A"],
    "cell_volume_A3": structure["cell_volume_A3"],
    "structure_source": "CONTCAR",
    "comment": structure["comment"],
}

# Queryable adsorbate labels. Keep count/chemistry, atomistic adsorption site,
# and multi-adsorbate spatial distribution as separate concepts.
if adsorbate_info["adsorbate_counts"]:
    sample["composition"]["adsorbate_counts"] = adsorbate_info["adsorbate_counts"]
    sample["composition"]["adsorbate_count_total"] = adsorbate_info["adsorbate_count_total"]
    sample["composition"]["adsorbate_identities"] = sorted(adsorbate_info["adsorbate_counts"])
    sample["composition"]["adsorbate_count_source"] = "CONTCAR_connectivity"
    sample["composition"]["poscar_count_crosscheck"] = adsorbate_info["poscar_count_crosscheck"]
    sample["composition"]["path_count_hint_status"] = adsorbate_info["path_count_hint_status"]
    sample["composition"]["path_identity_hint_status"] = adsorbate_info["path_identity_hint_status"]

if path_labels.get("distribution"):
    sample["geometry"]["adsorbate_distribution"] = path_labels["distribution"]
if path_labels.get("distribution_label"):
    sample["geometry"]["adsorbate_distribution_label"] = path_labels["distribution_label"]
if path_labels.get("region"):
    sample["geometry"]["adsorption_region"] = path_labels["region"]
if adsorbate_info["adsorption_sites"]:
    sample["geometry"]["adsorption_site_source"] = "relaxed_CONTCAR_geometry"
    sample["geometry"]["adsorption_sites"] = adsorbate_info["adsorption_sites"]
    sample["geometry"]["adsorption_site_counts"] = adsorbate_info["adsorption_site_counts"]
    if len(adsorbate_info["adsorption_site_counts"]) == 1:
        sample["geometry"]["adsorption_site"] = next(iter(adsorbate_info["adsorption_site_counts"]))
    else:
        sample["geometry"]["adsorption_site"] = "mixed"

# Explicit representation for this layer-resolved dataset.  In ISAAC v1.05,
# absorber/edge belong in the open system.configuration namespace, not directly
# under system.  Calculations have no system.instrument block.
system = record.setdefault("system", {})
system["domain"] = "computational"
system["technique"] = "XAS"
system.pop("absorber", None)
system.pop("edge", None)
system.pop("instrument", None)
facility = system.setdefault("facility", {})
facility["facility_name"] = "NERSC"
facility["organization"] = "Argonne National Laboratory"
facility["cluster"] = "Perlmutter"
configuration = system.setdefault("configuration", {})
configuration.pop("code_version", None)
configuration["absorber"] = "Cu"
configuration["edge"] = "K"
configuration["absorber_edge_source"] = "layer_resolved_dataset_definition"

if layer_input_settings:
    configuration["fdmnes_input_parse_status"] = (
        "complete_4_layers" if len(layer_input_settings) == 4 else f"partial_{len(layer_input_settings)}_of_4_layers"
    )
    fdmnes_settings_payload = {
        "common": common_fdmnes_settings,
        "layer_specific": layer_specific_fdmnes_settings,
        "input_files": {
            f"layer_{layer}": str(path.relative_to(xas_dir))
            for layer, path in sorted(layer_input_files.items())
        },
    }
    # system.configuration must remain flat; serialize structured code-specific
    # settings rather than placing nested dicts directly in the record.
    configuration["fdmnes_settings_json"] = json.dumps(
        fdmnes_settings_payload, sort_keys=True, separators=(",", ":")
    )
else:
    configuration["fdmnes_input_parse_status"] = "no_input_decks_found"

if convolution_input_settings:
    configuration["fdmnes_convolution_input_parse_status"] = (
        "complete_4_layers" if len(convolution_input_settings) == 4
        else f"partial_{len(convolution_input_settings)}_of_4_layers"
    )
    convolution_settings_payload = {
        "common": common_convolution_settings,
        "layer_specific": layer_specific_convolution_settings,
        "input_files": {
            f"layer_{layer}": str(path.relative_to(xas_dir))
            for layer, path in sorted(convolution_input_files.items())
        },
    }
    configuration["fdmnes_convolution_settings_json"] = json.dumps(
        convolution_settings_payload, sort_keys=True, separators=(",", ":")
    )
else:
    configuration["fdmnes_convolution_input_parse_status"] = "no_convolution_input_decks_found"

# ISAAC v1.05 requires computation.method.family + code for every calculation.
# The current vocabulary has no dedicated multiple-scattering-XAS family, so use
# the accepted semi_empirical bucket temporarily and preserve the precise FDMNES
# Green-function multiple-scattering method in code/notes.
method = record.setdefault("computation", {}).setdefault("method", {})
method["family"] = "semi_empirical"
method.pop("solver", None)
method["code"] = "FDMNES"
# Do not invent a code version.  Preserve one only if the base converter actually
# had a non-placeholder value.
code_version = str(method.get("code_version") or "").strip()
if code_version.lower() in {"", "fdmnes", "not_specified", "not_reported"}:
    method.pop("code_version", None)
notes = str(method.get("notes") or "").strip()
mode = common_fdmnes_settings.get("calculation_mode")
mode_note = f" Calculation mode: {mode}." if mode else ""
if layer_input_settings:
    setup_note = f"FDMNES settings parsed from {len(layer_input_settings)} main layer input deck(s)."
else:
    setup_note = "No main FDMNES *_in.txt input deck was available; spectrum and structure data only."
if convolution_input_settings:
    setup_note += f" Convolution settings parsed from {len(convolution_input_settings)} auxiliary input deck(s)."
schema_note = ("ISAAC currently has no dedicated multiple-scattering-XAS method family; "
               "family=semi_empirical is used only as the present schema-compatible bucket.")
extra_schema_note = "" if "family=semi_empirical" in notes else " " + schema_note
method["notes"] = (notes + " " + setup_note + mode_note + extra_schema_note).strip()

# No acquisition timestamps were recorded for these legacy calculations unless
# they were explicitly supplied.  ISAAC only requires created_utc, so absence is
# represented by omitting acquired_start/end rather than adding non-schema fields.
context = record.setdefault("context", {})
context.pop("acquisition_timestamp_source", None)
context.pop("acquisition_timestamp_status", None)
context["environment"] = "in_silico"
# Temperature was not recorded; do not silently assert 0 K.
context["temperature_K"] = None
context["temperature_basis"] = "not_reported"

measurement = record.setdefault("measurement", {})
processing = measurement.setdefault("processing", {})
processing["type"] = "simulated_spectrum_broadened"

# Add the single common relaxed structure as an asset.
assets = record.setdefault("assets", [])
assets = [a for a in assets if a.get("asset_id") != "relaxed_structure_contcar"]
try:
    contcar_uri = os.path.relpath(contcar, xas_dir)
except Exception:
    contcar_uri = str(contcar)
assets.append({
    "asset_id": "relaxed_structure_contcar",
    "content_role": "input_structure",
    "uri": Path(contcar_uri).as_posix(),
    "media_type": "chemical/x-poscar",
    "sha256": sha256_file(contcar),
    "notes": "Single relaxed CONTCAR shared by all four layer-resolved Cu K-edge spectra; used for adsorption-site assignment.",
})
if poscar is not None:
    assets = [a for a in assets if a.get("asset_id") != "initial_structure_poscar"]
    try:
        poscar_uri = os.path.relpath(poscar, xas_dir)
    except Exception:
        poscar_uri = str(poscar)
    assets.append({
        "asset_id": "initial_structure_poscar",
        "content_role": "input_structure",
        "uri": Path(poscar_uri).as_posix(),
        "media_type": "chemical/x-poscar",
        "sha256": sha256_file(poscar),
        "notes": "Initial POSCAR used only to cross-check adsorbate composition/count against the relaxed CONTCAR.",
    })
record["assets"] = assets

# Four layer spectra under the same record/structure.
series = []
descriptors = []

for layer in layers:
    path = layer_files[layer]
    energy, intensity = read_conv(path)
    series_id = f"Cu_K_layer_{layer}"

    series.append({
        "series_id": series_id,
        "independent_variables": [{
            "name": "energy",
            "unit": "eV",
            "values": energy,
        }],
        "channels": [{
            "name": "absorption_cross_section",
            "unit": "arb",
            "role": "simulated_observable",
            "values": intensity,
        }],
    })

    descriptors.extend([
        {
            "name": f"edge_position.layer_{layer}",
            "kind": "absolute",
            "source": "auto",
            "value": edge_position_first_derivative(energy, intensity),
            "unit": "eV",
            "definition": f"First-derivative maximum for {series_id}.",
            "uncertainty": {"basis": "none"},
        },
        {
            "name": f"white_line_intensity.layer_{layer}",
            "kind": "absolute",
            "source": "auto",
            "value": max(intensity),
            "unit": "arb",
            "definition": f"Maximum simulated intensity for {series_id}.",
            "uncertainty": {"basis": "none"},
        },
    ])

measurement["series"] = series

# Replace single-spectrum auto descriptors with layer-resolved versions while
# preserving the record's existing descriptor-output wrapper.
outputs = record.setdefault("descriptors", {}).setdefault("outputs", [])
if outputs:
    outputs[0]["label"] = "auto_features_layer_resolved_v1"
    outputs[0]["generated_by"] = {
        "agent": "CO2RR XAS Agent layer converter",
        "version": "2026-10-19.1",
    }
    outputs[0]["descriptors"] = descriptors
else:
    # Normally the base converter already supplies generated_utc/generated_by.
    # Keep this fallback schema-complete if that ever changes.
    outputs.append({
        "label": "auto_features_layer_resolved_v1",
        "generated_utc": timestamps.get("created_utc"),
        "generated_by": {
            "agent": "CO2RR XAS Agent layer converter",
            "version": "2026-10-19.1",
        },
        "descriptors": descriptors,
    })

# Record-level attribution required for computation evidence.
attribution = record.setdefault("attribution", {})
attribution.pop("uploaded_by", None)  # server stamps this at ingestion
attribution["produced_by"] = {
    "group": "Maria K. Y. Chan group",
    "organization": "Argonne National Laboratory",
    "people": ["Haili Jia", "Maria K. Y. Chan"],
}
# Include both data owners so the ownership warning is resolved.
contributors = [
    c for c in attribution.get("contributors", [])
    if isinstance(c, dict) and c.get("role") != "data_owner"
]
contributors.extend([
    {
        "name": "Haili Jia",
        "role": "data_owner",
        "affiliation": "Argonne National Laboratory",
    },
    {
        "name": "Maria K. Y. Chan",
        "role": "data_owner",
        "affiliation": "Argonne National Laboratory",
    },
])
attribution["contributors"] = contributors

# All organization fields generated by this converter use ANL.
# (affiliation is handled separately above.)
def force_anl_organization(obj):
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key == "organization":
                obj[key] = "Argonne National Laboratory"
            else:
                force_anl_organization(value)
    elif isinstance(obj, list):
        for value in obj:
            force_anl_organization(value)

force_anl_organization(record)

# Add concise tags without introducing non-schema per-series metadata.
tags = record.setdefault("tags", [])
for tag in ("layer_resolved", "CONTCAR_structure"):
    if tag not in tags:
        tags.append(tag)

out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(record, indent=2) + "\n")

print(f"Wrote {out_json}")
print(f"Structure: {contcar}")
print(f"Formula: {structure['formula']} ({structure['n_atoms']} atoms)")
if adsorbate_info["adsorbate_counts"]:
    print(f"Adsorbates: {adsorbate_info['adsorbate_counts']} (total={adsorbate_info['adsorbate_count_total']})")
    print(f"Adsorption sites: {adsorbate_info['adsorption_site_counts']}")
    print(f"Distribution: {path_labels.get('distribution') or 'not_labeled'}; label={path_labels.get('distribution_label') or 'none'}")
    print(f"Path identity cross-check: {adsorbate_info['path_identity_hint_status']}")
    print(f"POSCAR count cross-check: {adsorbate_info['poscar_count_crosscheck']}")
print("Series:")
for layer in layers:
    print(f"  Cu_K_layer_{layer} <- {layer_files[layer].relative_to(xas_dir)}")
if layer_input_settings:
    print(f"FDMNES main inputs parsed: {len(layer_input_settings)}/{len(layers)} layers")
    if common_fdmnes_settings:
        print("Common FDMNES settings:")
        for key, value in sorted(common_fdmnes_settings.items()):
            print(f"  {key}: {value}")
else:
    print("FDMNES main inputs parsed: none (conversion still valid from CONTCAR + spectra)")
if convolution_input_settings:
    print(f"FDMNES convolution inputs parsed: {len(convolution_input_settings)}/{len(layers)} layers")
else:
    print("FDMNES convolution inputs parsed: none")
if recorded_acquisition_times:
    print(f"Recorded acquisition timestamps: {recorded_acquisition_times}")
else:
    print("Acquisition timestamps: not recorded; omitted from record")
PY

# Basic local sanity check.
python - "$OUTPUT" <<'PY'
import json
import sys

p = sys.argv[1]
r = json.load(open(p))
s = r.get("measurement", {}).get("series", [])

assert len(s) == 4, f"Expected 4 series, found {len(s)}"
cfg = r.get("system", {}).get("configuration", {})
assert cfg.get("absorber") == "Cu"
assert cfg.get("edge") == "K"
assert "absorber" not in r.get("system", {})
assert "edge" not in r.get("system", {})
assert "instrument" not in r.get("system", {})
assert r.get("sample", {}).get("geometry", {}).get("structure_source") == "CONTCAR"
assert r.get("system", {}).get("facility", {}).get("organization") == "Argonne National Laboratory"
assert r.get("computation", {}).get("method", {}).get("code") == "FDMNES"
assert r.get("computation", {}).get("method", {}).get("family") == "semi_empirical"
assert "solver" not in r.get("computation", {}).get("method", {})
assert r.get("attribution", {}).get("produced_by", {}).get("organization") == "Argonne National Laboratory"
assert "ml_labels" not in r.get("sample", {}), "structure_info-derived ML labels should not be present"

ts = r.get("timestamps", {})
assert ts.get("created_utc"), "created_utc should be retained"
# acquired_start/end are optional here: absence means they were not recorded.

layer_ids = []
for item in s:
    series_id = item.get("series_id", "")
    import re
    m = re.fullmatch(r"Cu_K_layer_(\d+)", series_id)
    assert m, series_id
    layer_ids.append(int(m.group(1)))
    x = item["independent_variables"][0]["values"]
    y = item["channels"][0]["values"]
    assert len(x) == len(y) and len(x) > 2, series_id

assert len(set(layer_ids)) == 4, f"Expected 4 distinct layer IDs, found {layer_ids}"

print(f"Local check passed: one CONTCAR structure + 4 valid Cu K-edge series in {p}")
PY
