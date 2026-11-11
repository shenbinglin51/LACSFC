from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class WorkloadShiftConfig:
    monitor_enabled: bool = False
    window_size: int = 50
    pa_threshold: float = 1.20
    score_threshold: float = 1.20
    reopt_enabled: bool = False
    reopt_sigma_scale: float = 0.25
    reopt_max_evals: int = 50
    state_dir: str = ""


def default_shift_status(config: Optional[WorkloadShiftConfig] = None) -> Dict[str, Any]:
    cfg = config or WorkloadShiftConfig()
    return {
        "monitor_enabled": bool(cfg.monitor_enabled),
        "detected": False,
        "trigger_step": -1,
        "reopt_used": False,
        "reopt_sigma0": 0.0,
        "reopt_max_evals": 0,
        "state_path": "",
        "reason": "disabled" if not cfg.monitor_enabled else "not_triggered",
        "config": asdict(cfg),
    }


def evaluate_shift_from_summary(
    *,
    config: WorkloadShiftConfig,
    train_query_dist: str,
    test_query_dist: str,
    current_avg_amp: float,
    baseline_avg_amp: Optional[float] = None,
    current_three_factor_score: Optional[float] = None,
    baseline_three_factor_score: Optional[float] = None,
    sigma0: float = 0.0,
) -> Dict[str, Any]:
    status = default_shift_status(config)
    if not config.monitor_enabled:
        return status

    reasons = []
    train_dist = str(train_query_dist or "")
    test_dist = str(test_query_dist or "")
    if train_dist and test_dist and train_dist != test_dist:
        reasons.append(f"query_dist_changed:{train_dist}->{test_dist}")

    pa_ratio = None
    if baseline_avg_amp is not None and float(baseline_avg_amp) > 0.0:
        pa_ratio = float(current_avg_amp) / float(baseline_avg_amp)
        if pa_ratio > float(config.pa_threshold):
            reasons.append(f"avg_amp_ratio:{pa_ratio:.6f}>{float(config.pa_threshold):.6f}")

    score_ratio = None
    if (
        baseline_three_factor_score is not None
        and current_three_factor_score is not None
        and float(baseline_three_factor_score) > 0.0
    ):
        score_ratio = float(current_three_factor_score) / float(baseline_three_factor_score)
        if score_ratio > float(config.score_threshold):
            reasons.append(f"three_factor_ratio:{score_ratio:.6f}>{float(config.score_threshold):.6f}")

    detected = bool(reasons)
    status.update({
        "detected": detected,
        "trigger_step": 0 if detected else -1,
        "reopt_used": bool(detected and config.reopt_enabled),
        "reopt_sigma0": float(sigma0) * float(config.reopt_sigma_scale) if detected and config.reopt_enabled else 0.0,
        "reopt_max_evals": int(config.reopt_max_evals) if detected and config.reopt_enabled else 0,
        "state_path": str(config.state_dir or ""),
        "reason": ";".join(reasons) if reasons else "not_triggered",
        "avg_amp_ratio": pa_ratio,
        "three_factor_ratio": score_ratio,
    })
    return status
