"""One split and metric contract; immutable defaults loaded from this clone."""
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
CONTRACT_ID = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()

@dataclass(frozen=True)
class Fold:
    name: str
    origin: int
    horizon: int = 28
    weight_days: int = 28

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.origin, self.horizon, self.weight_days)):
            raise ValueError("Fold days must be positive integers")
        if self.weight_days > self.origin:
            raise ValueError("Weight window exceeds history")

    @property
    def history_days(self):
        return [f"d_{d}" for d in range(1, self.origin + 1)]

    @property
    def weight_columns(self):
        return self.history_days[-self.weight_days:]

    @property
    def forecast_days(self):
        return [f"d_{d}" for d in range(self.origin + 1, self.origin + self.horizon + 1)]


def fold(name):
    if name not in CONFIG["folds"]:
        raise ValueError("Unknown fold")
    spec = CONFIG["folds"][name]
    result = Fold(name, spec["origin"], CONFIG["horizon"])
    if spec["history"] != [1, result.origin] or spec["weights"] != [result.origin - 27, result.origin]:
        raise ValueError("Inconsistent fold contract")
    if CONFIG[name] != [result.origin + 1, result.origin + 28]:
        raise ValueError("Inconsistent forecast contract")
    return result


_TUNING_SCOPE = ContextVar("wrmsse_tuning_scope", default=False)
_TRAINING_SCOPE = ContextVar("wrmsse_training_scope", default=False)
_TEST_SCOPE = ContextVar("wrmsse_test_scope", default=False)

@contextmanager
def stage2_training_scope(evidence):
    import hashlib
    path=(ROOT/evidence).resolve()
    if not path.is_relative_to(ROOT/"preparation"):
        raise ValueError("Test evidence must be local")
    report=json.loads(path.read_text(encoding="utf-8"))
    policy=json.loads((ROOT/"configs/stage2.json").read_text(encoding="utf-8"))
    if not policy["training_enabled"] or report["status"]!="passed" or report["contract_id"]!=CONTRACT_ID:
        raise RuntimeError("Stage 2 training requires passing checks")
    for relative,expected in report["code_sha256"].items():
        source=(ROOT/relative).resolve()
        if not source.is_relative_to(ROOT) or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise RuntimeError("Code changed after checks")
    token=_TRAINING_SCOPE.set(True)
    try: yield
    finally: _TRAINING_SCOPE.reset(token)


@contextmanager
def stage3_tuning_scope(evidence):
    """Future opt-in policy kept separate from immutable data/metric contract."""
    policy=json.loads((ROOT/"configs/stage3.json").read_text(encoding="utf-8"))
    if not policy.get("allow_tuning",False) or policy.get("allow_test_evaluation",False):
        raise RuntimeError("Batch tuning is not authorized")
    path=(ROOT/evidence).resolve()
    if not path.is_relative_to(ROOT/"preparation"):raise ValueError("Local evidence required")
    report=json.loads(path.read_text(encoding="utf-8"))
    if report["status"]!="passed" or report["contract_id"]!=CONTRACT_ID:raise RuntimeError("Checks required")
    for relative,expected in report["code_sha256"].items():
        source=(ROOT/relative).resolve()
        if not source.is_relative_to(ROOT) or hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
            raise RuntimeError("Code changed after checks")
    training=_TRAINING_SCOPE.set(True);tuning=_TUNING_SCOPE.set(True)
    try:yield
    finally:
        _TUNING_SCOPE.reset(tuning);_TRAINING_SCOPE.reset(training)


def require_stage_permission(action):
    if action not in ("tuning", "training", "test_evaluation"):
        raise ValueError("Unknown action")
    if action == "tuning" and _TUNING_SCOPE.get():
        return
    if action == "training" and _TRAINING_SCOPE.get():
        return
    if action == "test_evaluation" and _TEST_SCOPE.get():
        return
    if CONFIG["stage"] == "preparation" or not CONFIG.get("allow_" + action, False):
        raise RuntimeError(f"{action} is disabled by the preparation contract")
