# Copyright 2026 tznurmin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .data_prep import align_word_labels, build_label_maps, load_canonical_rows, prepare_tokenized_rows
from .calibration import (
    apply_epoch_overrides_to_config,
    load_per_epoch_rows,
    summarize_calibration_for_model,
    write_calibrated_config_and_manifest,
)
from .experiment_matrix import build_run_matrix
from .experiment_runner import run_experiment_matrix
from .exp1_matrix import (
    build_exp1_run_matrix,
    normalize_training_models,
    prepared_dir_for_job,
    run_dir_for_job,
)
from .exp1_runner import run_exp1_matrix
from .hf_backend import run_hf_backend
from .parity import build_parity_signature, compare_metric_payloads, contract_mismatches, run_hf_parity_harness
from .prepared_cache import prepare_job_datasets
from .run_reporting import build_exp1_report_from_summary

__all__ = [
    "apply_epoch_overrides_to_config",
    "align_word_labels",
    "build_run_matrix",
    "build_exp1_run_matrix",
    "build_label_maps",
    "load_canonical_rows",
    "normalize_training_models",
    "prepare_tokenized_rows",
    "prepare_job_datasets",
    "prepared_dir_for_job",
    "build_exp1_report_from_summary",
    "build_parity_signature",
    "compare_metric_payloads",
    "contract_mismatches",
    "run_hf_backend",
    "run_hf_parity_harness",
    "run_experiment_matrix",
    "run_exp1_matrix",
    "run_dir_for_job",
    "load_per_epoch_rows",
    "summarize_calibration_for_model",
    "write_calibrated_config_and_manifest",
]
