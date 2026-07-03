"""Ground-truth vs. algorithm event matching and performance metrics.

IED detection is a point-event task, so there is no well-defined "true negative"
window; specificity is therefore not computed. Only TP / FP / FN and sensitivity
are reported.
"""

from dataclasses import dataclass, field

MATCH_TOLERANCE_S = 0.100


@dataclass
class ScoringResult:
    tp_pairs: list = field(default_factory=list)   # list of (gt_time_s, algo_time_s)
    fn_times: list = field(default_factory=list)    # unmatched ground-truth events
    fp_times: list = field(default_factory=list)    # unmatched algorithm events

    @property
    def tp(self) -> int:
        return len(self.tp_pairs)

    @property
    def fn(self) -> int:
        return len(self.fn_times)

    @property
    def fp(self) -> int:
        return len(self.fp_times)

    @property
    def sensitivity(self):
        denom = self.tp + self.fn
        return (self.tp / denom) if denom > 0 else None


def score_events(gt_times, algo_times, tolerance_s: float = MATCH_TOLERANCE_S) -> ScoringResult:
    """Greedy nearest-neighbor one-to-one matching between ground-truth (manual)
    and algorithm event timestamps within +/- tolerance_s."""
    gt_times = sorted(gt_times)
    algo_times = sorted(algo_times)

    candidates = []
    for gi, gt in enumerate(gt_times):
        for ai, at in enumerate(algo_times):
            diff = abs(gt - at)
            if diff <= tolerance_s:
                candidates.append((diff, gi, ai))
    candidates.sort(key=lambda c: c[0])

    matched_gt = set()
    matched_algo = set()
    tp_pairs = []
    for diff, gi, ai in candidates:
        if gi in matched_gt or ai in matched_algo:
            continue
        matched_gt.add(gi)
        matched_algo.add(ai)
        tp_pairs.append((gt_times[gi], algo_times[ai]))

    tp_pairs.sort(key=lambda p: p[0])
    fn_times = [gt_times[i] for i in range(len(gt_times)) if i not in matched_gt]
    fp_times = [algo_times[i] for i in range(len(algo_times)) if i not in matched_algo]

    return ScoringResult(tp_pairs=tp_pairs, fn_times=fn_times, fp_times=fp_times)
