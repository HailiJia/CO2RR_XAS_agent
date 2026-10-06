"""Configuration grouping and a fixed outer holdout for spectral ML."""
from __future__ import annotations

import hashlib
import json
import numpy as np
from .xas_record_utils import get_path


def configuration_group(row, group_field=None):
    record = row.get("record", {})
    metadata = row.get("metadata", {})
    if group_field:
        value = get_path(record, group_field)
        value = metadata.get(group_field) if value is None else value
        if value is None or str(value).strip() in {"", "None", "not_specified"}:
            raise ValueError(f"Missing group field {group_field} for a labeled spectrum.")
        return str(value), group_field
    for field in ("sample.configuration_id", "sample.structure_descriptors.configuration_id", "sample.structure_id"):
        value = get_path(record, field) or metadata.get(field)
        if value and str(value) != "not_specified":
            return str(value), field
    for field in ("configuration_id", "structure_id"):
        if row.get(field):
            return str(row[field]), field
    # Experimental replicate channels can at least be grouped by acquisition.
    # Simulation records need explicit linkage across absorbers/codes/recipes.
    if record.get("record_domain") == "experiment" and record.get("record_id"):
        return str(record["record_id"]), "experimental_record_id"
    raise ValueError("Simulation ML needs a decorated configuration_id or an explicit group field. Add configuration linkage before evaluation; the clean parent is provenance, not the default group.")


def groups_for_rows(rows, group_field=None):
    values = [configuration_group(row, group_field) for row in rows]
    groups = [v[0] for v in values]
    sources = set(v[1] for v in values)
    # Identical relaxed geometries reached from different initial structures
    # must also stay together. Configuration IDs keep preprocessing variants
    # together even when their geometry hashes are absent or differ.
    if group_field is None:
        parent = {group: group for group in groups}
        def find(group):
            while parent[group] != group:
                parent[group] = parent[parent[group]]
                group = parent[group]
            return group
        hashes = {}
        for row, group in zip(rows, groups):
            geometry = get_path(row.get("record", {}), "sample.structure_descriptors.geometry_hash")
            geometry = geometry or row.get("metadata", {}).get("sample.structure_descriptors.geometry_hash")
            if geometry:
                if geometry in hashes:
                    a, b = find(group), find(hashes[geometry])
                    if a != b:
                        parent[max(a, b)] = min(a, b)
                        sources.add("geometry_hash_deduplication")
                hashes[geometry] = group
        groups = [find(group) for group in groups]
    return np.asarray(groups, dtype=str), sorted(sources)


def grouped_holdout(groups, row_ids=None, test_fraction=0.25, seed=7, manifest=None):
    """Select whole groups without consulting spectra, features or targets."""
    groups = np.asarray(groups, dtype=str)
    if any(value.strip() in {"", "None", "nan", "not_specified"} for value in groups):
        raise ValueError("Every spectrum needs a nonempty configuration group ID.")
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError("Grouped evaluation needs at least three decorated configuration groups (two for training and one for testing).")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie between zero and one.")
    if manifest is not None:
        if set(manifest["train_groups"]) | set(manifest["test_groups"]) != set(unique):
            raise ValueError("Split manifest does not match the dataset groups.")
        if set(manifest["train_groups"]) & set(manifest["test_groups"]):
            raise ValueError("Split manifest leaks groups between training and test.")
        test_groups = set(manifest["test_groups"])
        if not test_groups or len(unique) - len(test_groups) < 2:
            raise ValueError("Split manifest needs nonempty test and at least two training groups.")
    else:
        shuffled = np.random.default_rng(seed).permutation(unique)
        count = min(len(unique) - 2, max(1, int(np.ceil(len(unique) * test_fraction))))
        test_groups = set(shuffled[:count])
    test = np.flatnonzero(np.isin(groups, list(test_groups)))
    train = np.flatnonzero(~np.isin(groups, list(test_groups)))
    ids = list(row_ids) if row_ids is not None else [str(i) for i in range(len(groups))]
    if len(ids) != len(groups) or len(set(ids)) != len(ids):
        raise ValueError("Row IDs must be unique and aligned with the spectra.")
    result = {"strategy": "configuration_group_holdout", "seed": int(seed),
              "train_groups": sorted(set(groups[train])), "test_groups": sorted(test_groups),
              "train_indices": train.tolist(), "test_indices": test.tolist(),
              "train_row_ids": [ids[i] for i in train], "test_row_ids": [ids[i] for i in test]}
    result["split_id"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:24]
    return train, test, result
