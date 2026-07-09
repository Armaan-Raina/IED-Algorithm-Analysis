"""Ground-truth vs. algorithm event matching and performance metrics.

Metrics:
- TP: Validated events matched to algorithm events
- FP: Algorithm events not matched to validated events
- FN: Validated events not matched to algorithm events
- TN: Rejected events not matched to algorithm events (correct rejections)

Sensitivity = TP / (TP + FN)
Specificity = TN / (TN + FP_rejected) where FP_rejected = rejected events with algorithm matches
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
        denom = self.tn + self.fp_rejected
        return (self.tn / denom) if denom > 0 else None


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

    # Count true negatives (rejected events not matched to algorithm)
    # and false positives from rejected events (algorithm detected something rejected)
    tn_count = 0
    fp_rejected_count = 0

    for ri, rejected_time in enumerate(rejected_times):
        matched = False
        for ai, algo_time in enumerate(algo_times):
            if abs(rejected_time - algo_time) <= tolerance_s and ai in matched_algo:
                # This rejected event was matched to an algo event (FP from algo perspective)
                fp_rejected_count += 1
                matched = True
                break
            elif abs(rejected_time - algo_time) <= tolerance_s and ai not in matched_algo:
                # This rejected event has a nearby algo event but it's not matched yet
                # Mark as matched to avoid double-counting
                matched_algo.add(ai)
                fp_rejected_count += 1
                matched = True
                break

        if not matched:
            # Rejected event with no nearby algorithm event (true negative)
            tn_count += 1

    return ScoringResult(
        tp_pairs=tp_pairs,
        fn_times=fn_times,
        fp_times=fp_times,
        tn=tn_count,
        fp_rejected=fp_rejected_count
    )
