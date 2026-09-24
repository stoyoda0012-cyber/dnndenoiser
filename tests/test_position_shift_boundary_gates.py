"""Every P2-A self-check must be shown to REJECT the failure it is named for.

A gate that has only ever been run on correct input has told us nothing: passing is
what a tautology does too. Four of P2-A's thirteen gates were tautologies at some
point -- each compared an expression against itself -- and all four passed every run
until an independent audit built the wrong input by hand. This file makes that
construction permanent.

Each gate gets a pair: the correct input is ACCEPTED, and each named wrong input is
REJECTED -- for the reason named, pinned with `match=`, because a rejection raised by
some other check on the way would pass a bare `pytest.raises` just as well. The pair
matters: a gate that rejects everything would pass the second half.

What this file does not establish, stated so it is not read as more: rejecting one
constructed failure does not prove a gate catches every failure. Each test names the
single failure mode it covers. Gates or failure modes with no test here are listed in
`benchmarks/boundaries/position_shift/README.md` as untested.

The measurement script is imported, never modified: the published record is the
output of that script at its producing commit.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "benchmarks" / "boundaries" / "position_shift" / "position_shift_boundary.py"

pytestmark = pytest.mark.skipif(
    not SCRIPT.is_file(),
    reason="benchmarks/boundaries is a source-checkout tree, not part of the distribution",
)


@pytest.fixture(scope="module")
def P():
    spec = importlib.util.spec_from_file_location("_p2a_measurement_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Pool sizes large enough that a CORRECT augmented pool fills all ten deciles with
# overwhelming probability (10 * 0.9**300 ~ 2e-13), small enough to build in well under
# a second. Arms C and D are subsets of arm A's sequence, as in the real design.
POOL_SIZES = {
    "A_narrow_2304": (100, 100, 100),
    "B_augmented_2304": (100, 100, 100),
    "C_narrow_461": (50, 50, 50),
    "D_narrow_144": (20, 20, 20),
}


def _pools(P, seed_index=0):
    return {arm: P.build_training_pool(seed_index, arm, POOL_SIZES[arm], P.AUG_SHIFT_STREAM_BASE)
            for arm in P.ARM_ORDER}


def _wrong_augmented_pool(P, *, applied_shift: bool, jitter: float, seed_index=0,
                          shift_source=None):
    """An arm-B pool built WRONGLY, recording shifts as the real one does."""
    shift_rng = np.random.default_rng(P.AUG_SHIFT_STREAM_BASE + seed_index)
    parts = {k: [] for k in ("clean", "noisy", "shifts", "batch_seeds", "keys", "levels")}
    energy = None
    for level_index, level in enumerate(P.NOISE_LEVELS):
        for sample_index in range(POOL_SIZES["B_augmented_2304"][level_index]):
            delta = (float(shift_rng.uniform(-1.5, 1.5)) if shift_source is None
                     else shift_source(shift_rng))
            seed = P.train_sample_seed(seed_index, level_index, sample_index)
            generator = P.make_generator(delta if applied_shift else 0.0, level,
                                         position_jitter=jitter)
            clean, noisy, energy, _ = generator.generate_batch(1, seed=seed)
            parts["clean"].append(clean[0])
            parts["noisy"].append(noisy[0])
            parts["shifts"].append(delta)
            parts["batch_seeds"].append(seed)
            parts["keys"].append((level_index, sample_index))
            parts["levels"].append(level)
    return {"clean": np.asarray(parts["clean"], np.float32),
            "noisy": np.asarray(parts["noisy"], np.float32),
            "shifts": np.asarray(parts["shifts"]), "batch_seeds": parts["batch_seeds"],
            "keys": parts["keys"], "levels": parts["levels"], "energy": energy}


# --------------------------------------------------------------------------------------
# 1 -- parameter count
# --------------------------------------------------------------------------------------

def test_1_accepts_the_reference_count(P):
    reference, _ = P.read_reference_record()
    P.check_parameter_count(reference)


def test_1_rejects_a_count_that_does_not_match(P):
    fake = {"self_checks": {"parameter_counts": {"observed": {P.ARCH: 1}}}}
    with pytest.raises(P.SelfCheckFailure, match=r"parameter count"):
        P.check_parameter_count(fake)


# --------------------------------------------------------------------------------------
# 2, 3 -- grid invariance and the test sweep's peak-set construction
# --------------------------------------------------------------------------------------

def test_2_3_accept_the_pinned_rigid_sweep(P):
    P.check_grid_and_rigidity()


def test_2_rejects_a_grid_that_moves_with_the_shift(P, monkeypatch):
    """Failure mode: forgetting to pin the energy range, so PeakSet auto-ranges."""
    kwargs = dict(P.GENERATOR_CONFIG_KWARGS)
    kwargs.pop("energy_range")
    monkeypatch.setattr(P, "GENERATOR_CONFIG_KWARGS", kwargs)
    monkeypatch.setattr(P, "shifted_peak_set", lambda delta: P.PeakSet(
        id="unpinned", name="unpinned",
        peaks=[P.PeakParams(mu=mu + delta, fwhm=w, intensity=i) for mu, w, i in
               zip(P.NOMINAL_CENTRES, P.NOMINAL_FWHM, P.NOMINAL_INTENSITY)]))
    with pytest.raises(P.SelfCheckFailure, match=r"grid moved"):
        P.check_grid_and_rigidity()


def test_3_rejects_a_non_rigid_sweep(P, monkeypatch):
    """Failure mode: each peak shifted by a different amount."""
    monkeypatch.setattr(P, "shifted_peak_set", lambda delta: P.PeakSet(
        id="nonrigid", name="nonrigid", energy_range=P.ENERGY_RANGE,
        peaks=[P.PeakParams(mu=mu + delta * (1 + 0.1 * k), fwhm=w, intensity=i)
               for k, (mu, w, i) in enumerate(zip(P.NOMINAL_CENTRES, P.NOMINAL_FWHM,
                                                   P.NOMINAL_INTENSITY))]))
    with pytest.raises(P.SelfCheckFailure, match=r"non-rigid test sweep"):
        P.check_grid_and_rigidity()


# --------------------------------------------------------------------------------------
# 4 -- training-pool rigidity (the check that was a tautology until Revision 4)
# --------------------------------------------------------------------------------------

def test_4_accepts_correctly_built_pools(P):
    P.check_pool_rigidity(_pools(P), np.random.default_rng(1))


@pytest.mark.parametrize("label,applied_shift,jitter", [
    ("per-peak jitter 1.8 instead of a rigid shift (the audit's pool)", False, 1.8),
    ("shift recorded but never applied", False, 0.3),
    ("shift applied at the wrong jitter width", True, 1.8),
])
def test_4_rejects_a_wrongly_built_augmented_pool(P, label, applied_shift, jitter):
    pools = _pools(P)
    pools["B_augmented_2304"] = _wrong_augmented_pool(P, applied_shift=applied_shift,
                                                      jitter=jitter)
    with pytest.raises(P.SelfCheckFailure, match=r"is not what a rigid shift"):
        P.check_pool_rigidity(pools, np.random.default_rng(1))


def test_4_rejects_a_mutated_peak_registry(P, monkeypatch):
    """Failure mode: `p.mu += delta` on the shared PEAK_SETS object."""
    pools = _pools(P)
    live = P.get_peak_set(P.PEAK_SET_ID)
    monkeypatch.setattr(live.peaks[0], "mu", live.peaks[0].mu + 0.5)
    with pytest.raises(P.SelfCheckFailure, match=r"was mutated during the run"):
        P.check_pool_rigidity(pools, np.random.default_rng(1))


# --------------------------------------------------------------------------------------
# 5 -- truncation
# --------------------------------------------------------------------------------------

def test_5_accepts_the_registered_sweep(P):
    P.check_truncation()


def test_5_rejects_a_sweep_past_the_truncation_limit(P, monkeypatch):
    """Failure mode: extending the sweep to +/-4.5 eV, where the dominant peak's
    retained area falls outside the 1% tolerance (measured at design time)."""
    monkeypatch.setattr(P, "DELTAS", tuple(sorted(set(P.DELTAS) | {4.5, -4.5})))
    with pytest.raises(P.SelfCheckFailure, match=r"truncation at delta=-4\.5"):
        P.check_truncation()


# --------------------------------------------------------------------------------------
# 6 -- input-SNR invariance: NOT independently testable (stated, not hidden)
# --------------------------------------------------------------------------------------

def test_6_tolerance_equals_the_registered_value_at_the_registered_size(P):
    """Only the tolerance helper is testable. The check itself is inline in `run()`
    and is exercised by full runs only; that is listed as a limitation in the README."""
    for level in P.NOISE_LEVELS:
        assert P.input_snr_tolerance(level, P.N_TEST_PER_LEVEL) == P.TOL_INPUT_SNR_DB[level]
        assert P.input_snr_tolerance(level, P.N_TEST_PER_LEVEL // 4) == pytest.approx(
            2.0 * P.TOL_INPUT_SNR_DB[level])


# --------------------------------------------------------------------------------------
# 7 -- translation equivariance (the check that was a tautology until Revision 3)
# --------------------------------------------------------------------------------------

def test_7_accepts_the_registered_background(P):
    P.check_translation_equivariance()


def test_7_rejects_a_background_that_breaks_translation(P, monkeypatch):
    """Failure mode: a background ramp steep enough that a peak moving along it
    departs from a pure translate by more than 1% of peak height."""
    kwargs = dict(P.GENERATOR_CONFIG_KWARGS)
    kwargs["background_slope"] = 0.01
    monkeypatch.setattr(P, "GENERATOR_CONFIG_KWARGS", kwargs)
    with pytest.raises(P.SelfCheckFailure, match=r"translation equivariance"):
        P.check_translation_equivariance()


# --------------------------------------------------------------------------------------
# 8 -- pairing across arms; 8b -- test-family rigidity
# --------------------------------------------------------------------------------------

def test_8_accepts_identically_seeded_arms(P):
    P.check_pairing_integrity(_pools(P))


def test_8_rejects_desynchronised_arms(P):
    pools = _pools(P)
    pools["B_augmented_2304"]["batch_seeds"] = [s + 1 for s in pools["B_augmented_2304"]["batch_seeds"]]
    with pytest.raises(P.SelfCheckFailure, match=r"streams desynchronised"):
        P.check_pairing_integrity(pools)


def test_8b_accepts_the_paired_family(P):
    for delta in (0.0, 1.5, -4.0):
        clean, _noisy, _e = P.draw_test(0, 1, 1000.0, delta, 16)
        P.check_test_family_rigidity(clean, 0, 1, 1000.0, delta)


def test_8b_rejects_a_family_drawn_from_another_seed(P):
    """Failure mode: a fresh seed per shift, which breaks pairing across the sweep."""
    clean, _noisy, _e = P.draw_test(7, 1, 1000.0, 1.5, 16)   # seed 7's family ...
    with pytest.raises(P.SelfCheckFailure, match=r"not the paired family"):
        P.check_test_family_rigidity(clean, 0, 1, 1000.0, 1.5)   # ... claimed as seed 0's


def test_8b_rejects_a_family_at_the_wrong_jitter(P):
    generator = P.make_generator(1.5, 1000.0, position_jitter=1.8)
    clean, _noisy, _e, _ = generator.generate_batch(16, seed=P.test_batch_seed(0, 1))
    with pytest.raises(P.SelfCheckFailure, match=r"not the paired family"):
        P.check_test_family_rigidity(clean.astype(np.float32), 0, 1, 1000.0, 1.5)


# --------------------------------------------------------------------------------------
# 9 -- argmax well-posedness and reference identity (was `clean is clean`)
# --------------------------------------------------------------------------------------

def test_9_accepts_scoring_against_the_shift_matched_reference(P):
    clean, noisy, energy = P.draw_test(0, 1, 1000.0, 2.0, 32)
    P.snr_db(noisy, clean)
    P.check_argmax_well_posed_and_reference_identity(clean, energy, 2.0)


def test_9_rejects_scoring_against_the_unshifted_reference(P):
    """Failure mode: M1 given the delta = 0 truth while the delta = 2 family is inspected."""
    clean, noisy, energy = P.draw_test(0, 1, 1000.0, 2.0, 32)
    unshifted, _n, _e = P.draw_test(0, 1, 1000.0, 0.0, 32)
    P.snr_db(noisy, unshifted)
    with pytest.raises(P.SelfCheckFailure, match=r"different array as its reference"):
        P.check_argmax_well_posed_and_reference_identity(clean, energy, 2.0)


def test_9_rejects_running_before_any_metric(P, monkeypatch):
    monkeypatch.setattr(P, "_LAST_SNR_REFERENCE", None)
    clean, _noisy, energy = P.draw_test(0, 1, 1000.0, 0.0, 8)
    with pytest.raises(P.SelfCheckFailure, match=r"before any SNR"):
        P.check_argmax_well_posed_and_reference_identity(clean, energy, 0.0)


def test_9_rejects_a_reference_peaking_at_the_wrong_place(P):
    """Failure mode: the inspected family is labelled with a shift it does not have."""
    clean, noisy, energy = P.draw_test(0, 1, 1000.0, 2.0, 32)
    P.snr_db(noisy, clean)
    with pytest.raises(P.SelfCheckFailure, match=r"argmax ill-posed"):
        P.check_argmax_well_posed_and_reference_identity(clean, energy, 0.0)


# --------------------------------------------------------------------------------------
# 10 -- augmentation actually happened (passed on an all-zero arm B until Revision 4)
# --------------------------------------------------------------------------------------

def test_10_accepts_correct_augmentation(P):
    P.check_augmentation(_pools(P))


@pytest.mark.parametrize("label,draw,reason", [
    ("never augmented: every shift zero", lambda rng: 0.0, r"shift SD"),
    ("too narrow: U(-0.3, +0.3)", lambda rng: float(rng.uniform(-0.3, 0.3)), r"shift SD"),
    ("one-sided: U(0, +1.5)", lambda rng: float(rng.uniform(0.0, 1.5)), r"shift mean"),
])
def test_10_rejects_the_wrong_shift_distribution(P, label, draw, reason):
    pools = _pools(P)
    pools["B_augmented_2304"] = _wrong_augmented_pool(P, applied_shift=True, jitter=0.3,
                                                      shift_source=draw)
    with pytest.raises(P.SelfCheckFailure, match=reason):
        P.check_augmentation(pools)


def test_10_rejects_a_narrow_arm_that_drew_shifts(P):
    pools = _pools(P)
    pools["C_narrow_461"]["shifts"] = pools["C_narrow_461"]["shifts"] + 0.1
    with pytest.raises(P.SelfCheckFailure, match=r"narrow arm but drew non-zero"):
        P.check_augmentation(pools)


# --------------------------------------------------------------------------------------
# 11 -- noise-model identity
# --------------------------------------------------------------------------------------

def test_11_accepts_the_pinned_noise_model(P):
    P.check_noise_model()


@pytest.mark.parametrize("label,override", [
    ("Gaussian approximation forced at every level", {"use_gaussian_approx": True}),
    ("the current per-bin default floor", {"gaussian_approx_min_rate": 3.0}),
])
def test_11_rejects_a_different_noise_model(P, monkeypatch, label, override):
    original = P.noise_config

    def altered(level):
        config = original(level)
        for key, value in override.items():
            setattr(config, key, value)
        return config

    monkeypatch.setattr(P, "noise_config", altered)
    with pytest.raises(P.SelfCheckFailure, match=r"noise config at level"):
        P.check_noise_model()


# --------------------------------------------------------------------------------------
# 12 -- leakage
# --------------------------------------------------------------------------------------

def _delta_zero_test(P):
    out = {}
    for level_index, level in enumerate(P.NOISE_LEVELS):
        clean, noisy, _e = P.draw_test(0, level_index, level, 0.0, 16)
        out[level] = {"clean": clean, "noisy": noisy}
    return out


def test_12_accepts_disjoint_train_and_test(P):
    P.check_leakage(_pools(P), _delta_zero_test(P))


@pytest.mark.parametrize("arm", ["A_narrow_2304", "B_augmented_2304", "D_narrow_144"])
def test_12_rejects_a_training_spectrum_in_the_test_set(P, arm):
    """Covers every arm, because the first implementation hashed arm A only."""
    pools = _pools(P)
    test = _delta_zero_test(P)
    test[1000.0]["clean"] = test[1000.0]["clean"].copy()
    test[1000.0]["noisy"] = test[1000.0]["noisy"].copy()
    test[1000.0]["clean"][0] = pools[arm]["clean"][0]
    test[1000.0]["noisy"][0] = pools[arm]["noisy"][0]
    with pytest.raises(P.SelfCheckFailure, match=r"byte-identical to a training"):
        P.check_leakage(pools, test)


# --------------------------------------------------------------------------------------
# provenance -- read from git at run time; a full run from a dirty tree is refused
# --------------------------------------------------------------------------------------

def _fake_git(status: str, touching=("c" * 40, "b" * 40, "a" * 40)):
    def fake(*args):
        if args[0] == "status":
            return status
        if args[0] == "rev-parse":
            return "d" * 40
        if args[0] == "log":
            return "\n".join(touching)
        raise AssertionError(f"unexpected git call {args}")
    return fake


def test_provenance_accepts_a_clean_tree_and_reads_the_registration_from_git(P, monkeypatch):
    monkeypatch.setattr(P, "_git", _fake_git(""))
    record = P.provenance_record(quick=False)
    assert record["working_tree_clean"] is True
    assert record["code_commit"] == "d" * 40
    # git log lists newest first: the first registration is the LAST line.
    assert record["registration"]["first_commit"] == "a" * 40
    assert record["registration"]["last_commit_before_run"] == "c" * 40


@pytest.mark.parametrize("label,status", [
    ("a modified tracked file", " M benchmarks/boundaries/position_shift/position_shift_boundary.py"),
    ("an untracked file", "?? uv.lock"),
])
def test_provenance_refuses_a_full_run_from_a_dirty_tree(P, monkeypatch, label, status):
    monkeypatch.setattr(P, "_git", _fake_git(status))
    with pytest.raises(P.SelfCheckFailure, match=r"refusing a full run"):
        P.provenance_record(quick=False)


def test_provenance_lets_a_quick_run_proceed_and_says_the_tree_was_dirty(P, monkeypatch):
    monkeypatch.setattr(P, "_git", _fake_git("?? scratch.txt"))
    record = P.provenance_record(quick=True)
    assert record["working_tree_clean"] is False
    assert record["working_tree_changes_if_dirty"] == ["?? scratch.txt"]


def test_provenance_carries_no_developer_specific_path(P):
    """Against the real repository. The interpreter's location in particular is an
    absolute path and must not appear; only whether it is a virtual environment."""
    spec = importlib.util.spec_from_file_location(
        "_tracked_paths", REPO_ROOT / "tests" / "test_tracked_paths.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    text = json.dumps(P.provenance_record(quick=True))
    assert not [t for t in guard.forbidden_tokens() if t in text], text
