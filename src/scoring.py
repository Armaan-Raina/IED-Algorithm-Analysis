"""Ground-truth vs. algorithm event matching and performance metrics.

Metrics:
- TP: Validated events matched to algorithm events
- FP: Algorithm events not matched to validated events
- FN: Validated events not matched to algorithm events
- TN: No longer used (was for auto-detected candidates; now uses CSV imports)
- FP_rejected: Rejected events (algorithm detected but user rejected them)

Sensitivity = TP / (TP + FN)
Specificity: No longer calculated (TN undefined with CSV-based candidates)
"""

from dataclasses import dataclass, field

MATCH_TOLERANCE_S = 0.100


@dataclass
class ScoringResult:
    tp_pairs: list = field(default_factory=list)   # list of (gt_time_s, algo_time_s)
    fn_times: list = field(default_factory=list)    # unmatched ground-truth events
    fp_times: list = field(default_factory=list)    # unmatched algorithm events
    tn: int = 0                                       # rejected events not matched to algo
    fp_rejected: int = 0                              # rejected events matched to algo (false positives)

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

    @property
    def specificity(self):
        # No longer meaningful with CSV-based candidates (TN is always 0)
        return None


def score_events(gt_times, algo_times, tolerance_s: float = MATCH_TOLERANCE_S, rejected_times=None) -> ScoringResult:
    """Greedy nearest-neighbor one-to-one matching between ground-truth (manual)
    and algorithm event timestamps within +/- tolerance_s.

    Args:
        gt_times: Validated event times (ground truth / true events)
        algo_times: Algorithm-detected event times
        tolerance_s: Matching tolerance in seconds
        rejected_times: Rejected event times (not true IEDs)
    """
    gt_times = sorted(gt_times)
    algo_times = sorted(algo_times)
    rejected_times = sorted(rejected_times) if rejected_times else []

    # Match validated events to algorithm events
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

    # Count false positives from rejected events
    # Since candidates now come from CSV (algo_events), all rejected events are FP
    # TN is no longer meaningful (set to 0)
    fp_rejected_count = len(rejected_times) if rejected_times else 0

    return ScoringResult(
        tp_pairs=tp_pairs,
        fn_times=fn_times,
        fp_times=fp_times,
        tn=0,  # No longer meaningful since candidates come from CSV
        fp_rejected=fp_rejected_count
    )
