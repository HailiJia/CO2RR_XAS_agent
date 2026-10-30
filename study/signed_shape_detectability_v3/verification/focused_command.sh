#!/bin/sh
set -eu
python -m pytest -q \
 tests/test_geometry_labels.py \
 tests/test_grouped_xas_ml.py \
 tests/test_xas_ml_peak_descriptors.py \
 tests/test_nersc_workflow.py \
 tests/test_web_nersc_workflow_integration.py \
 tests/test_nersc_prompt_regressions.py \
 tests/test_label_workflow_integration.py \
 tests/test_study_reconciliation.py \
 tests/test_differential_xas.py \
 tests/test_dataset_refresh.py \
 tests/test_reference_v3.py \
 tests/test_relative_identity.py \
 tests/test_near_geometry_guard.py \
 tests/test_detectability_units.py \
 tests/test_signed_shape_size.py
