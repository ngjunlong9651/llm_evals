"""Small, readable eval toolkit used by evals_learning.ipynb.

    graders        code-based checks (exact, regex, JSON schema-ish)         exact_match, regex_match, valid_json_with
    judge          a SIMULATED LLM judge with known error rates              SimulatedJudge
    stats          bootstrap confidence intervals, Cohen's kappa             bootstrap_ci, cohen_kappa
    judge fixes    correct a noisy judge using its measured TPR/TNR          corrected_pass_rate
    data           a synthetic support-bot dataset with labeled failures     make_dataset

Everything is deterministic (seeded) and needs only numpy, so it runs offline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------ code-based graders
def exact_match(output: str, expected: str) -> bool:
    return output.strip().lower() == expected.strip().lower()


def regex_match(output: str, pattern: str) -> bool:
    return re.search(pattern, output, flags=re.I) is not None


def valid_json_with(output: str, required_keys: list[str]) -> bool:
    """Deterministic format check: parses, is an object, has the required keys."""
    try:
        obj = json.loads(output)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and all(k in obj for k in required_keys)


# -------------------------------------------------------------------------- statistics
def bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a mean. Resample the data with replacement, recompute the mean, repeat."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cohen_kappa(a, b) -> float:
    """Agreement between two binary raters, corrected for the agreement expected by chance."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()                                              # observed agreement
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())        # agreement expected by chance
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def tpr_tnr(judge, human):
    """Judge quality measured against human labels. 'Positive' = the output is GOOD (passes)."""
    j, h = np.asarray(judge, bool), np.asarray(human, bool)
    tpr = (j & h).sum() / max(h.sum(), 1)            # of truly good outputs, how many does the judge pass?
    tnr = (~j & ~h).sum() / max((~h).sum(), 1)       # of truly bad outputs, how many does the judge fail?
    return float(tpr), float(tnr)


def corrected_pass_rate(observed_rate: float, tpr: float, tnr: float) -> float:
    """Rogan-Gladen correction: estimate the TRUE pass rate from a noisy judge's observed rate.

    observed = tpr * true + (1 - tnr) * (1 - true)   =>   true = (observed + tnr - 1) / (tpr + tnr - 1)
    """
    denom = tpr + tnr - 1
    return float(np.clip((observed_rate + tnr - 1) / denom, 0, 1)) if denom > 0 else float("nan")


# ------------------------------------------------------------------------ simulated judge
@dataclass
class SimulatedJudge:
    """Stands in for an LLM judge. Real judges are noisy and biased; we model that with explicit error rates.

    tpr, tnr: probability of correctly passing a good output / correctly failing a bad one.
    verbosity_bias: extra chance of passing an output just because it is long (a documented LLM-judge bias).
    """
    tpr: float = 0.90
    tnr: float = 0.75
    verbosity_bias: float = 0.0
    seed: int = 1

    def judge(self, outputs: list[dict]) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        verdicts = []
        for o in outputs:
            p_pass = self.tpr if o["good"] else (1 - self.tnr)
            if len(o["answer"].split()) > 40:                  # long answers look more convincing
                p_pass = min(1.0, p_pass + self.verbosity_bias)
            verdicts.append(rng.random() < p_pass)
        return np.array(verdicts)


# --------------------------------------------------------------------------- the dataset
FAILURE_MODES = {
    "hallucinated_policy": "states a policy that does not exist",
    "ignored_context": "contradicts the retrieved document",
    "wrong_tone": "rude or dismissive",
    "too_verbose": "buries the answer in filler",
    "refused_wrongly": "refuses a safe, in-scope request",
    "format_error": "returns malformed JSON when JSON was requested",
}


def make_dataset(n: int = 200, seed: int = 7, true_pass_rate: float = 0.70) -> list[dict]:
    """Synthetic support-bot outputs. `good` is the hidden ground truth; `failure` names the problem when bad.

    Failure modes are deliberately NOT uniform (a few dominate), as in real systems: that is what makes
    error analysis pay off.
    """
    rng = np.random.default_rng(seed)
    modes = list(FAILURE_MODES)
    weights = np.array([0.34, 0.24, 0.14, 0.12, 0.10, 0.06])        # a few failure modes dominate
    rows = []
    for i in range(n):
        good = bool(rng.random() < true_pass_rate)
        failure = None if good else str(rng.choice(modes, p=weights / weights.sum()))
        words = int(rng.integers(8, 30)) + (45 if failure == "too_verbose" else 0) + (int(rng.integers(0, 25)) if rng.random() < 0.2 else 0)
        rows.append({"id": i, "good": good, "failure": failure, "answer": " ".join(["word"] * words)})
    return rows
