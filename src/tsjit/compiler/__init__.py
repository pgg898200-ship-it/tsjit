# Copyright 2026 tsjit contributors
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

"""Plan-aware public time-series foundation model compiler."""

from .api import CompileConfig, CompiledTSFM, compile, normalize_precision, resolve_family_precision
from .chronos_pass import (
    Chronos2Compiler,
    ChronosBoltCompiler,
    is_chronos2_model,
    is_chronos_bolt_model,
)
from .generic_pass import GenericTSFMCompiler, compile_generic
from .graph_runner import DEFAULT_BATCH_BUCKETS, BucketedCudaGraphRunner, InductorOrGraphRunner
from .guards import (
    DEFAULT_RATIONAL_PERIODS,
    CausalFloat64DCStripper,
    ParityAuditReport,
    StrictParityValidator,
    extract_rational_lag_features,
    scale_free_norm_floor,
)
from .toto2_pass import (
    Toto2ArchitectureInfo,
    Toto2CompiledEngine,
    Toto2Compiler,
    compile_toto2,
    inspect_toto2_architecture,
    is_toto2_model,
    verify_parity,
)

__all__ = [
    "BucketedCudaGraphRunner",
    "CausalFloat64DCStripper",
    "Chronos2Compiler",
    "ChronosBoltCompiler",
    "CompileConfig",
    "CompiledTSFM",
    "DEFAULT_BATCH_BUCKETS",
    "DEFAULT_RATIONAL_PERIODS",
    "GenericTSFMCompiler",
    "InductorOrGraphRunner",
    "ParityAuditReport",
    "StrictParityValidator",
    "Toto2ArchitectureInfo",
    "Toto2CompiledEngine",
    "Toto2Compiler",
    "compile",
    "compile_generic",
    "compile_toto2",
    "extract_rational_lag_features",
    "inspect_toto2_architecture",
    "is_chronos2_model",
    "is_chronos_bolt_model",
    "is_toto2_model",
    "normalize_precision",
    "resolve_family_precision",
    "scale_free_norm_floor",
    "verify_parity",
]
