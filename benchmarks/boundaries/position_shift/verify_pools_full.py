"""Post-hoc: self-check 4 at 100 % coverage, on the pools a committed record was made from.

Written for Revision 8 of the P2-A preregistration. The third run's self-check 4 sampled
12 spectra per arm per seed where the registration said 5 % of every pool. This script
answers the question that deviation leaves open -- would the pools have passed the
registered check? -- by regenerating every training pool (they are deterministic in the
seed) and running check 4 on ALL of them, not a sample.

Regeneration is only worth anything if the regenerated pools are the ones the run trained
on, so each seed is first ANCHORED to statistics the run itself recorded over every
sample: the augmentation statistics of check 10 (every arm's shift distribution) and the
near-duplicate statistics of check 12 (arm A's pool against the delta = 0 test sets). A
seed whose anchors do not match exactly is reported and not counted.

What this cannot do: turn the run's own gate into the registered one. It is evidence
about the pools, gathered after the fact, and it is reported as that.

    python benchmarks/boundaries/position_shift/verify_pools_full.py            # working tree
    python benchmarks/boundaries/position_shift/verify_pools_full.py --commit 8474c94
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import position_shift_boundary as P  # noqa: E402

RECORD = "benchmarks/boundaries/position_shift/results/position_shift_boundary.json"


def load_record(commit: str | None) -> dict:
    if commit is None:
        return json.loads((P.REPO_ROOT / RECORD).read_text(encoding="utf-8"))
    return json.loads(subprocess.run(["git", "show", f"{commit}:{RECORD}"], cwd=P.REPO_ROOT,
                                     check=True, capture_output=True).stdout)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", default=None,
                        help="read the record from this commit instead of the working tree")
    args = parser.parse_args(argv)
    record = load_record(args.commit)
    per_seed = record["self_checks"]["4_8_10_12_per_seed"]

    anchored, compared = 0, {arm: 0 for arm in P.ARM_ORDER}
    for seed, stored in enumerate(per_seed):
        pools = {arm: P.build_training_pool(seed, arm, P.ARMS[arm]["train_per_level"],
                                            P.AUG_SHIFT_STREAM_BASE) for arm in P.ARM_ORDER}
        regenerated_augmentation = P.check_augmentation(pools)
        augmentation_matches = all(
            regenerated_augmentation[arm][key] == value
            for arm in P.ARM_ORDER
            for key, value in stored["10_augmentation"][arm].items()
            if key in regenerated_augmentation[arm])
        delta_zero = {}
        for level_index, level in enumerate(P.NOISE_LEVELS):
            clean, noisy, _ = P.draw_test(seed, level_index, level, 0.0, P.N_TEST_PER_LEVEL)
            delta_zero[level] = {"clean": clean, "noisy": noisy}
        regenerated = P.check_leakage(pools, delta_zero)["near_duplicate_analysis"]
        recorded = stored["12_leakage"]["near_duplicate_analysis"]
        leakage_matches = all(regenerated[k] == recorded[k]
                              for k in recorded if isinstance(recorded[k], float))
        if not (augmentation_matches and leakage_matches):
            print(f"seed {seed:2d}: anchors do NOT match the record -- not counted")
            continue
        anchored += 1
        report = P.check_pool_rigidity(pools, np.random.default_rng(0), n_samples=10**9)
        for arm in P.ARM_ORDER:
            compared[arm] += report[arm]["n_reconstructed_and_compared"]
        print(f"seed {seed:2d}: anchored; check 4 on 100 % of every pool: passed", flush=True)

    print(f"\n{anchored}/{len(per_seed)} seeds anchored to the record and passed at 100 %")
    print("spectra rebuilt and compared bit-for-bit, per arm:", compared)
    return 0 if anchored == len(per_seed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
