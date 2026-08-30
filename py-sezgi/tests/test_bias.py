"""M3-1 Task 8: Python bindings for the bias-scanning module (sezgi.bias).

Mirrors crates/bias's public structs 1:1 by field name -- see that crate's
`structural`/`central`/`report` modules for the full method provenance.
Budgets/dims are kept tiny throughout (CI speed), except `runs=30` for the
structural scan, which is the BIAS-toolbox method's own verified minimum
(sezgi_bias::structural::DEFAULT_RUNS) -- not something this test suite can
shrink without changing what is actually being tested.
"""
import sezgi
import pytest


def rs_spec(pop_size=5, budget=50):
    return sezgi.presets.random_search(pop_size=pop_size, budget=budget)


# ---- sezgi.bias.structural -------------------------------------------------

def test_structural_dict_shape():
    spec = rs_spec()
    r = sezgi.bias.structural(spec, dim=2, budget=50, runs=30, seed=20260830)

    assert set(r.keys()) == {
        "per_dim_ks", "per_dim_ad", "holm_rejections_ks", "holm_rejections_ad",
        "verdict", "detail", "final_positions",
    }
    assert len(r["per_dim_ks"]) == 2
    assert len(r["per_dim_ad"]) == 2
    for row in r["per_dim_ks"]:
        assert set(row.keys()) == {"d", "p_value", "n"}
        assert isinstance(row["d"], float)
        assert isinstance(row["p_value"], float)
        assert isinstance(row["n"], int)
    for row in r["per_dim_ad"]:
        assert set(row.keys()) == {"a2", "p_value", "n"}
        assert isinstance(row["a2"], float)
        assert isinstance(row["p_value"], float)
        assert isinstance(row["n"], int)
    assert isinstance(r["holm_rejections_ks"], int)
    assert isinstance(r["holm_rejections_ad"], int)
    assert r["verdict"] in ("no_evidence", "evidence")
    if r["verdict"] == "no_evidence":
        assert r["detail"] is None
    else:
        assert isinstance(r["detail"], str)
    assert len(r["final_positions"]) == 30
    assert all(len(row) == 2 for row in r["final_positions"])


def test_structural_deterministic_and_anchored_verdict():
    """random_search (pop=5, budget=50) on f0(dim=2), runs=30, seed=20260830:
    has no directional operator (uniform resampling + elitist replacement),
    so it should show no structural bias on f0 -- ANCHORED (per this
    project's convention): this is the actual measured verdict at this
    seed/config, captured from this test's own first successful run, not a
    re-derivation."""
    spec = rs_spec()
    r1 = sezgi.bias.structural(spec, dim=2, budget=50, runs=30, seed=20260830)
    r2 = sezgi.bias.structural(spec, dim=2, budget=50, runs=30, seed=20260830)
    assert r1 == r2, "same config must yield a bit-identical result dict"

    # Measured at this exact config (captured from this test's own first
    # successful run, before this assertion was added):
    #   per_dim_ks p-values = [0.8931464115659824, 0.7346814853531314]
    #   per_dim_ad p-values = [0.8926490729470692, 0.7520431448530165]
    # None below ALPHA (0.01), let alone after Holm correction --
    # holm_rejections_ks = 0, holm_rejections_ad = 0, verdict = no_evidence.
    assert r1["holm_rejections_ks"] == 0
    assert r1["holm_rejections_ad"] == 0
    assert r1["verdict"] == "no_evidence"
    assert r1["detail"] is None
    # Bit-exact spot checks (per this task's bit-equality mandate: scalar
    # stats pass through as exact f64, no rounding/formatting) -- exact `==`
    # on the anchored values above, not an approximate comparison.
    assert r1["per_dim_ks"][0]["p_value"] == 0.8931464115659824
    assert r1["per_dim_ad"][0]["a2"] == r2["per_dim_ad"][0]["a2"]
    assert r1["final_positions"][0][0] == r2["final_positions"][0][0]


def test_structural_invalid_dim_raises():
    spec = rs_spec()
    with pytest.raises(ValueError):
        sezgi.bias.structural(spec, dim=0, budget=50, runs=30, seed=1)


def test_structural_too_few_runs_raises():
    spec = rs_spec()
    with pytest.raises(ValueError):
        sezgi.bias.structural(spec, dim=2, budget=50, runs=3, seed=1)


def test_structural_invalid_spec_raises():
    spec = rs_spec()
    spec["stages"][0]["generator"]["kind"] = "yok/boyle"
    with pytest.raises(ValueError, match="yok/boyle"):
        sezgi.bias.structural(spec, dim=2, budget=50, runs=30, seed=1)


def test_structural_invalid_spec_json_string_raises():
    with pytest.raises(ValueError):
        sezgi.bias.structural("{not json", dim=2, budget=50, runs=30, seed=1)


# ---- sezgi.bias.central -----------------------------------------------------

def test_central_dict_shape():
    spec = rs_spec()
    r = sezgi.bias.central(spec, dim=2, budget=50, fids=[1], instances_shifted=[1],
                            runs_per=5, seed=20260830)

    assert set(r.keys()) == {
        "gap_centered", "gap_shifted", "wilcoxon", "effect", "verdict", "detail",
    }
    assert len(r["gap_centered"]) == 5
    assert len(r["gap_shifted"]) == 5
    assert all(isinstance(v, float) for v in r["gap_centered"])
    assert all(isinstance(v, float) for v in r["gap_shifted"])
    assert set(r["wilcoxon"].keys()) == {"w_statistic", "z", "p_value", "n_effective", "method"}
    assert r["wilcoxon"]["method"] in ("exact", "normal_approx")
    assert isinstance(r["effect"], float)
    assert r["verdict"] in ("no_evidence", "evidence")
    if r["verdict"] == "no_evidence":
        assert r["detail"] is None
    else:
        assert isinstance(r["detail"], str)


def test_central_deterministic():
    spec = rs_spec()
    r1 = sezgi.bias.central(spec, dim=2, budget=50, fids=[1], instances_shifted=[1],
                             runs_per=5, seed=20260830)
    r2 = sezgi.bias.central(spec, dim=2, budget=50, fids=[1], instances_shifted=[1],
                             runs_per=5, seed=20260830)
    assert r1 == r2, "same config must yield a bit-identical result dict"


def test_central_default_fids_and_instances():
    """fids/instances_shifted default to crates/bias's own [1, 4, 13] / [1, 2]
    (DEFAULT_CENTRAL_FIDS/DEFAULT_CENTRAL_INSTANCES) when omitted."""
    spec = rs_spec()
    r = sezgi.bias.central(spec, dim=2, budget=50, runs_per=1, seed=1)
    # 3 fids * 2 instances * 1 run_per = 6 pairs.
    assert len(r["gap_centered"]) == 6
    assert len(r["gap_shifted"]) == 6


def test_central_rejects_non_translation_invariant_fid_5():
    spec = rs_spec()
    with pytest.raises(ValueError):
        sezgi.bias.central(spec, dim=5, budget=50, fids=[5], instances_shifted=[1, 2],
                            runs_per=3, seed=1)


@pytest.mark.parametrize("fid", [5, 6, 20, 24])
def test_central_rejects_all_non_translation_invariant_fids(fid):
    spec = rs_spec()
    with pytest.raises(ValueError):
        sezgi.bias.central(spec, dim=5, budget=50, fids=[fid], instances_shifted=[1, 2],
                            runs_per=3, seed=1)


def test_central_too_few_pairs_raises():
    spec = rs_spec()
    with pytest.raises(ValueError):
        sezgi.bias.central(spec, dim=2, budget=50, fids=[1], instances_shifted=[1],
                            runs_per=1, seed=1)


# ---- sezgi.bias.report -------------------------------------------------------

def test_report_dict_shape_and_latex_nan_free():
    spec = rs_spec()
    r = sezgi.bias.report(spec, dim=2, budget=50, seed=20260830,
                           structural_runs=30, central_fids=[1],
                           central_instances=[1], central_runs_per=5)

    assert set(r.keys()) == {"structural", "central", "signature", "latex_summary", "plot_data"}
    assert set(r["structural"].keys()) == {
        "per_dim_ks", "per_dim_ad", "holm_rejections_ks", "holm_rejections_ad",
        "verdict", "detail", "final_positions",
    }
    assert set(r["central"].keys()) == {
        "gap_centered", "gap_shifted", "wilcoxon", "effect", "verdict", "detail",
    }
    # T6 (Rajwar-Deep signature test) is deferred in sezgi-bias itself:
    # always None -- the method could not be pinned from accessible sources.
    assert r["signature"] is None
    assert "NaN" not in r["latex_summary"]
    assert "\\toprule" in r["latex_summary"]
    assert "not run: the Rajwar-Deep method could not be pinned" in r["latex_summary"]
    assert set(r["plot_data"].keys()) == {"final_positions", "gap_centered", "gap_shifted"}
    assert r["plot_data"]["final_positions"] == r["structural"]["final_positions"]
    assert r["plot_data"]["gap_centered"] == r["central"]["gap_centered"]
    assert r["plot_data"]["gap_shifted"] == r["central"]["gap_shifted"]


def test_report_deterministic():
    spec = rs_spec()
    r1 = sezgi.bias.report(spec, dim=2, budget=50, seed=20260830,
                            structural_runs=30, central_fids=[1],
                            central_instances=[1], central_runs_per=5)
    r2 = sezgi.bias.report(spec, dim=2, budget=50, seed=20260830,
                            structural_runs=30, central_fids=[1],
                            central_instances=[1], central_runs_per=5)
    assert r1 == r2, "same config must yield a bit-identical report dict"


def test_report_defaults_use_documented_verified_method_defaults():
    """Omitting every optional knob falls back to BiasReportConfig::new's
    own documented defaults: structural_runs=30, central_fids=[1, 4, 13],
    central_instances=[1, 2], central_runs_per=20 -- so the default report
    has 30 final_positions rows and 3*2*20=120 gap pairs."""
    spec = rs_spec()
    r = sezgi.bias.report(spec, dim=2, budget=50, seed=1)
    assert len(r["structural"]["final_positions"]) == 30
    assert len(r["central"]["gap_centered"]) == 120
    assert len(r["central"]["gap_shifted"]) == 120


def test_report_invalid_spec_raises():
    spec = rs_spec()
    spec["stages"][0]["generator"]["kind"] = "yok/boyle"
    with pytest.raises(ValueError, match="yok/boyle"):
        sezgi.bias.report(spec, dim=2, budget=50, seed=1,
                           structural_runs=30, central_fids=[1],
                           central_instances=[1], central_runs_per=5)


def test_report_invalid_central_fid_raises():
    spec = rs_spec()
    with pytest.raises(ValueError):
        sezgi.bias.report(spec, dim=2, budget=50, seed=1,
                           structural_runs=30, central_fids=[5],
                           central_instances=[1], central_runs_per=5)
