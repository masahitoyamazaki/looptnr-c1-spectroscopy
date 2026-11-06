"""Fit saved representative U1 chi=40 medium-tube spectra as in note7 §4.1.

Run from the repository root:
    MPLCONFIGDIR=/tmp/sixvertex_mpl python3 Fushio/code/six_vertex/scripts/paper_u1_current_20261101.py
No RG, eigensolver, or tensor write is performed. The appendix retains its
separate legacy data and fixed-angle windows; figures 5/6 use free gamma/omega.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import least_squares
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[4]
SIX = ROOT / "Fushio/code/six_vertex"
DATA = SIX / "results/u1_chi40_representative_medium_tube_20261031"
OUT = SIX / "results/paper_u1_current_20261101"
TABLE = ROOT / "research_note/calculation_table/support/representative_calculations.csv"
NOTE = ROOT / "Fushio/my_notes/note7_six_vertex_readout_and_current.md"
NOTE_CODE = SIX / "scripts/note7_readout_20261027/analyze.py"
NOTE_RATES = SIX / "results/note7_readout_20261027/current_rates.csv"
LEGACY_FITS = SIX / "results/perturbation_chi40_cu1_u1_20261021/analysis/odd_current_fits.csv"
TARGETS = (.75, 1., 1.25, 1.5, 1.75, 2.5)
PANELS = (.75, 1.5, 2.5)
INPUT_HASHES: dict[Path, str] = {}


def tracked_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert path not in INPUT_HASHES or INPUT_HASHES[path] == digest
    INPUT_HASHES[path] = digest
    return data


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(tracked_bytes(path).decode().splitlines()))


def write_csv(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_points() -> tuple[list[dict], list[dict]]:
    scope = [row for row in read_csv(TABLE)
             if row["group"] == "representative" and row["symmetry"] == "U1"
             and int(row["chi"]) == 40]
    assert sorted(float(row["K_exact"]) for row in scope) == list(TARGETS)
    for row in scope:
        assert float(row["a"]) == float(row["b"]) == 1
        assert "[√2,2√2,0]" in row["tube"]
        assert str((DATA / "observables.csv").relative_to(ROOT)) in row["cft_evidence"]
    by_k = {float(row["K_exact"]): row for row in scope}
    points = []
    for row in read_csv(DATA / "observables.csv"):
        k, n = float(row["K_target"]), int(row["rg_step"])
        assert k in by_k and 0 <= n <= 50
        path = DATA / row["result"]
        if not path.is_file():
            points.append(dict(K=k, rg_step=n, weak=False, rejection_reasons="missing_result",
                               result=row["result"]))
            continue
        d = tomllib.loads(tracked_bytes(path).decode())
        o = d["observables"]
        assert o["K_target"] == k and o["rg_step"] == n
        assert o["symmetry"] == "U1Irrep" and o["chi"] == 40
        assert np.allclose(o["shape"], [np.sqrt(2), 2*np.sqrt(2), 0], rtol=0, atol=1e-14)
        assert math.isclose(o["vertex_c"], float(by_k[k]["c"]), rel_tol=0, abs_tol=1e-12)
        assert math.isclose(np.pi/(2*np.arccos(1-o["vertex_c"]**2/2)), k, abs_tol=1e-10)
        assert o["status"] == row["status"]
        assert o["checkpoint_sha256"] == row["checkpoint_sha256"]
        pair = [next((s for s in d.get("spectra", [])
                      if s["charge"] == q and s["rank"] == 1), {}) for q in (1, -1)]
        reasons = []
        if not o.get("attempt_complete", False):
            reasons.append("attempt_incomplete")
        if o["status"] != "ok":
            reasons.append("status_" + o["status"])
        for q, s in zip((1, -1), pair):
            if not s:
                reasons.append(f"missing_charge_{q}_rank1")
                continue
            if not np.isfinite(s.get("x", np.nan)) or s["x"] <= 0:
                reasons.append(f"nonpositive_or_nonfinite_charge_{q}")
            if not np.isfinite(s.get("residual_relative", np.nan)) or s["residual_relative"] > 1e-8:
                reasons.append(f"residual_charge_{q}")
            if s.get("converged", 0) < 1:
                reasons.append(f"unconverged_charge_{q}")
        xp, xm = [s.get("x", np.nan) for s in pair]
        for value, name in ((xp, "x1"), (xm, "xm1")):
            if np.isfinite(value):
                assert np.isclose(value, o[name], rtol=1e-12, atol=1e-15)
                assert np.isclose(value, float(row[name]), rtol=1e-12, atol=1e-15)
        exact = 1/(4*k)
        error = max(abs(xp/exact-1), abs(xm/exact-1))
        if not np.isfinite(error) or error >= .1:
            reasons.append("charge1_relative_error_ge_0.10_or_nonfinite")
        central = o.get("central_charge", np.nan)
        if not np.isfinite(central) or abs(central-1) >= .2:
            reasons.append("central_charge_error_ge_0.2_or_nonfinite")
        points.append(dict(K=k, vertex_c=o["vertex_c"], rg_step=n, status=o["status"],
                           x_plus=xp, x_minus=xm, exact1=exact, E1=(xp+xm)/2,
                           O1=(xp-xm)/2, dE1=(xp+xm)/2-exact,
                           central_charge=central, max_charge1_relative_error=error,
                           residual_plus=pair[0].get("residual_relative", np.nan),
                           residual_minus=pair[1].get("residual_relative", np.nan),
                           converged_plus=pair[0].get("converged", 0),
                           converged_minus=pair[1].get("converged", 0),
                           weak=not reasons, rejection_reasons=";".join(reasons),
                           result=row["result"], checkpoint_source=o["source"],
                           checkpoint_sha256=o["checkpoint_sha256"],
                           worker_sha256=o["worker_sha256"], manifest_sha256=o["manifest_sha256"]))
    assert len({(r["K"], r["rg_step"]) for r in points}) == len(points)
    return scope, points


def select_window(points: list[dict]) -> tuple[list[dict], dict]:
    index = {row["rg_step"]: row for row in points}
    good = []
    for n in range(15, 51):
        row = index.get(n)
        if row is None:
            return good, dict(step=n, reason="missing_step")
        if not row["weak"]:
            return good, dict(step=n, reason=row["rejection_reasons"])
        good.append(row)
    return good, dict(step=None, reason="end_of_available_range")


def oddfit(rows: list[dict]) -> tuple[dict, np.ndarray]:
    """The variable-projection search in note7 analyze.py::oddfit, with coefficients saved."""
    n = np.array([r["rg_step"] for r in rows])
    t = n-n[-1]
    y = np.array([r["O1"] for r in rows])
    scale = np.linalg.norm(y)
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("zero or nonfinite odd signal")

    def solve(z):
        g, w = z
        design = np.exp(g*t)[:, None] * np.array([np.cos(w*t), np.sin(w*t)]).T
        coeff = np.linalg.lstsq(design, y, rcond=None)[0]
        return design @ coeff, coeff

    starts = sorted((np.linalg.norm(solve((g, w))[0]-y), g, w)
                    for g in np.linspace(-.7, 1, 9)
                    for w in np.linspace(.1, 3.04, 35))[:5]
    opts = [least_squares(lambda z: (solve(z)[0]-y)/scale, [g, w],
                         bounds=([-1.5, .01], [1.5, np.pi-.01]),
                         xtol=1e-11, ftol=1e-11, gtol=1e-11) for _, g, w in starts]
    best = min(opts, key=lambda o: np.linalg.norm(o.fun))
    assert best.success, best.message
    g, w = best.x
    pred, coeff = solve(best.x)
    boundary = bool(abs(g) > 1.499 or min(w-.01, np.pi-.01-w) < .001)
    return dict(start=int(n[0]), end=int(n[-1]), n_star=int(n[-1]), npoints=len(n),
                log_gamma=float(g), gamma=float(np.exp(g)), omega=float(w),
                omega_deg=float(np.degrees(w)), u=float(coeff[0]), v=float(coeff[1]),
                rmse=float(np.sqrt(np.mean((pred-y)**2))),
                relative_residual=float(np.linalg.norm(pred-y)/scale), boundary=boundary), pred


def analyze(points: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    summaries, windows, predictions = [], [], []
    for k in TARGETS:
        subset = [r for r in points if r["K"] == k]
        good, stop = select_window(subset)
        for r in subset:
            r["selection_role"] = ("before_RG15" if r["rg_step"] < 15 else
                                    "eligible" if r in good else
                                    "first_rejected" if r["rg_step"] == stop["step"] else "after_stop")
            r["in_primary"] = False
            r["in_figure5"] = k in PANELS and r in good
        summary = dict(K=k, eligible_points=len(good), stop_step=stop["step"], stop_reason=stop["reason"])
        if len(good) < 10 or np.linalg.norm([r["O1"] for r in good[-10:]]) == 0:
            summaries.append(dict(**summary, fit_status="insufficient_points_or_zero_signal"))
            continue
        main = good[-10:]
        fit, pred = oddfit(main)
        sensitivity = []
        for length in (8, 10, 12):
            for shift in (0, 1, 2):
                hi = len(good)-shift
                if hi < length:
                    continue
                window, _ = oddfit(good[hi-length:hi])
                windows.append(dict(K=k, length=length, endpoint_shift=shift, **window))
                sensitivity.append(window)
        summary.update(fit_status="boundary" if fit["boundary"] else "ok", **fit,
                       gamma_min=min(w["gamma"] for w in sensitivity),
                       gamma_max=max(w["gamma"] for w in sensitivity),
                       omega_min=min(w["omega_deg"] for w in sensitivity),
                       omega_max=max(w["omega_deg"] for w in sensitivity),
                       sensitivity_windows=len(sensitivity),
                       boundary_windows=sum(w["boundary"] for w in sensitivity),
                       relative_to_sqrt2=fit["gamma"]/np.sqrt(2)-1)
        assert summary["gamma_min"] <= fit["gamma"] <= summary["gamma_max"]
        for row, prediction in zip(main, pred):
            row["in_primary"] = True
            predictions.append(dict(K=k, rg_step=row["rg_step"], O1=row["O1"],
                                    O1_fit=float(prediction), residual=float(row["O1"]-prediction)))
        summaries.append(summary)
    return summaries, windows, predictions


def validate(summaries: list[dict], windows: list[dict], predictions: list[dict]) -> dict:
    expected = { .75: (18, 27, 1.4089184664058951), 1.: (21, 30, 1.4134944289007878),
                 1.25: (24, 33, 1.4139603522182227), 1.5: (21, 30, 1.414117161480069),
                 1.75: (22, 31, 1.4144024663902222), 2.5: (22, 31, 1.413970631942139)}
    assert len(summaries) == 6 and all(s["fit_status"] == "ok" for s in summaries)
    assert all(not w["boundary"] for w in windows)
    for s in summaries:
        lo, hi, gamma = expected[s["K"]]
        assert (s["start"], s["end"], s["npoints"]) == (lo, hi, 10)
        assert abs(s["gamma"]-gamma) < 1e-7
        rr = [r for r in predictions if r["K"] == s["K"]]
        n = np.array([r["rg_step"] for r in rr])
        t = n-s["n_star"]
        recon = s["gamma"]**t*(s["u"]*np.cos(s["omega"]*t)+s["v"]*np.sin(s["omega"]*t))
        assert np.allclose(recon, [r["O1_fit"] for r in rr], rtol=1e-12, atol=1e-15)
        assert np.isclose(np.sqrt(np.mean([r["residual"]**2 for r in rr])), s["rmse"], rtol=1e-12)
    old = next(r for r in read_csv(NOTE_RATES) if float(r["vertex_c"]) == 1.)
    new = next(s for s in summaries if s["K"] == 1.5)
    assert abs(new["gamma"]-float(old["r"])) < 1e-7
    assert abs(new["omega_deg"]-float(old["omega_deg"])) < 1e-7
    synthetic = []
    for gamma, omega in ((1.35, 2.1), (.85, .9)):
        n = np.arange(15, 27)
        t = n-n[-1]
        y = gamma**t*(.03*np.cos(omega*t)-.02*np.sin(omega*t))
        result, _ = oddfit([dict(rg_step=int(a), O1=float(b)) for a, b in zip(n, y)])
        assert abs(result["gamma"]-gamma) < 1e-7 and abs(result["omega"]-omega) < 1e-7
        synthetic.append(dict(expected_gamma=gamma, expected_omega=omega, **result))
    fake = [dict(rg_step=n, weak=n != 18, rejection_reasons="test_rejection") for n in range(15, 31)]
    good, stop = select_window(fake)
    assert [r["rg_step"] for r in good] == [15, 16, 17] and stop["step"] == 18
    good, stop = select_window([r for r in fake if r["rg_step"] != 17])
    assert len(good) == 2 and stop == dict(step=17, reason="missing_step")
    insufficient = [dict(K=k, rg_step=n, weak=True, rejection_reasons="", O1=.01)
                    for k in TARGETS for n in range(15, 24)]
    failed, no_windows, no_predictions = analyze(insufficient)
    assert all(s["fit_status"] == "insufficient_points_or_zero_signal" for s in failed)
    assert not no_windows and not no_predictions
    return dict(passed=True, primary_windows_and_pilot_values=True, note7_K1p5_reproduced=True,
                signed_predictions_reconstructed=True, no_boundary_solutions=True,
                stop_at_first_failure_or_missing=True, insufficient_windows_recorded_without_fit=True,
                synthetic_growth_and_decay=synthetic)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=240)
    plt.close(fig)


def plot_main(points, summaries, predictions) -> list[dict]:
    fig, axes = plt.subplots(1, 3, figsize=(7.12, 2.65), layout="constrained", sharey=True)
    plot_records = []
    for i, (ax, k) in enumerate(zip(axes, PANELS)):
        s = next(s for s in summaries if s["K"] == k)
        data = sorted((r for r in points if r["K"] == k and r["in_figure5"]), key=lambda r: r["rg_step"])
        pred = [r for r in predictions if r["K"] == k]
        n = [r["rg_step"] for r in data]
        nf = [r["rg_step"] for r in pred]
        assert n == list(range(15, s["end"]+1))
        assert nf == list(range(s["start"], s["end"]+1))
        ax.axvspan(s["start"], s["end"], color="#b54d32", alpha=.07, lw=0)
        ax.semilogy(n, [abs(r["dE1"]) for r in data], "s-", color="#376f9f", ms=2.8, lw=.7,
                    label=r"$|E_1-1/(4K)|$")
        ax.semilogy(n, [abs(r["O1"]) for r in data], "o", color="#b54d32", ms=3.1, label=r"$|O_1|$")
        ax.semilogy(nf, [abs(r["O1_fit"]) for r in pred], "-", color="#b54d32", lw=.8, label=r"$O_1$ fit")
        ax.set(xlabel=r"RG step $n$", ylim=(5e-7, .07), xlim=(14.3, 31.7))
        ax.set_xticks([15, 20, 25, 30])
        ax.set_title(rf"({chr(97+i)}) $K={k:g}$", fontsize=8.5, loc="left")
        ax.text(.04, .95, f"fit: n = {s['start']}–{s['end']}", transform=ax.transAxes,
                va="top", fontsize=7.3)
        ax.grid(alpha=.16, linewidth=.5)
        if i == 0:
            ax.set_ylabel("level deviation")
            ax.legend(fontsize=6.7, loc="lower right", framealpha=.9)
        plot_records.append(dict(K=k, display_steps=n, fit_steps=nf,
                                 result_files=[r["result"] for r in data]))
    save(fig, "u1_representative_flows")

    k = np.array([s["K"] for s in summaries])
    gamma = np.array([s["gamma"] for s in summaries])
    low = np.array([s["gamma_min"] for s in summaries])
    high = np.array([s["gamma_max"] for s in summaries])
    fig, ax = plt.subplots(figsize=(3.42, 2.35), layout="constrained")
    ax.errorbar(k, gamma, yerr=[gamma-low, high-gamma], fmt="o", color="#b54d32",
                markersize=3.7, elinewidth=.8, capsize=2)
    ax.axhline(np.sqrt(2), color="#333333", linewidth=.9, linestyle="--")
    pad = .08*(high.max()-low.min())
    ax.set(xlim=(.6, 2.65), xlabel=r"$K$", ylabel=r"growth factor $\gamma$",
           ylim=(low.min()-pad, high.max()+pad))
    ax.set_yticks([1.405, 1.410, np.sqrt(2)], ["1.405", "1.410", r"$\sqrt{2}$"])
    ax.grid(alpha=.16, linewidth=.5)
    save(fig, "u1_current_growth")
    return plot_records


def plot_legacy_appendix() -> list[dict]:
    rows = sorted(read_csv(LEGACY_FITS), key=lambda r: float(r["K_exact"]))
    k = [float(r["K_exact"]) for r in rows]
    tube = [float(r["tube_ratio"]) for r in rows]
    residual = [100*float(r["charge_linearity_residual"]) for r in rows]
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.9), layout="constrained", sharex=True)
    for ax in axes:
        ax.set(xlim=(.43, 3.22))
        ax.grid(alpha=.16, linewidth=.5)
    axes[-1].set_xlabel(r"$K$")
    axes[0].plot(k, tube, "o", color="#b54d32", markersize=3.7)
    axes[0].axhline(2, color="#333333", linewidth=.9, linestyle="--")
    axes[0].set(ylabel=r"tube ratio $\rho$", ylim=(1.986, 2.006))
    axes[0].set_title("(a) tube ratio", fontsize=8.5, loc="left")
    axes[1].plot(k, residual, "o", color="#b54d32", markersize=3.7)
    axes[1].set(yscale="log", ylabel=r"charge residual $\epsilon_{\rm ch}$ (\%)", ylim=(.01, 50))
    axes[1].set_title(r"(b) $O_q\propto q$ check", fontsize=8.5, loc="left")
    save(fig, "u1_current_appendix_checks")
    return [dict(K=float(r["K_exact"]), vertex_c=float(r["vertex_c"]),
                 start=int(r["start"]), end=int(r["end"]), omega_fixed_deg=135.,
                 tube_ratio=float(r["tube_ratio"]), charge_linearity_residual=float(r["charge_linearity_residual"])) for r in rows]


def main() -> None:
    tracked_bytes(NOTE)
    tracked_bytes(NOTE_CODE)
    tracked_bytes(Path(__file__))
    scope, points = load_points()
    summaries, windows, predictions = analyze(points)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("current_fits", summaries), ("fit_windows", windows),
                       ("selection_audit", points), ("fit_predictions", predictions)):
        write_csv(name, rows)
    try:
        validation = validate(summaries, windows, predictions)
    except Exception as exc:
        (OUT / "validation.json").write_text(json.dumps(dict(passed=False, error=repr(exc)), indent=2)+"\n")
        raise
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8.5,
                         "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.linewidth": .7, "pdf.fonttype": 42})
    figure5 = plot_main(points, summaries, predictions)
    legacy = plot_legacy_appendix()
    for path, digest in INPUT_HASHES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Input changed: {path}"
    validation["input_hashes_unchanged"] = True
    (OUT / "validation.json").write_text(json.dumps(validation, indent=2)+"\n")
    (OUT / "figure_provenance.json").write_text(json.dumps(dict(
        schema_version=2, method="note7 section 4.1: free gamma and omega, signed O1",
        medium_tube=[np.sqrt(2), 2*np.sqrt(2), 0], legacy_tube_name="wide", chi=40,
        inputs=[dict(path=str(p.relative_to(ROOT)), sha256=h) for p, h in sorted(INPUT_HASHES.items())],
        inventory_scope=scope, conditions=summaries, figure5=figure5, figure6_K=list(TARGETS),
        selection=dict(start=15, stop="first failure or missing step", main_points=10,
                       charge1_relative_error_lt=.1, central_charge_error_lt=.2,
                       residual_relative_le=1e-8, positive_dimensions=True, converged_ge=1,
                       status="ok", sensitivity_lengths=[8, 10, 12], endpoint_shifts=[0, 1, 2]),
        fit=dict(log_gamma_bounds=[-1.5, 1.5], omega_bounds=[.01, np.pi-.01],
                 grid_log_gamma=[-.7, 1., 9], grid_omega=[.1, 3.04, 35], best_starts=5,
                 xtol=1e-11, ftol=1e-11, gtol=1e-11),
        legacy_appendix_conditions=legacy, new_fits=True, new_RG_or_CFT=False,
        checkpoint_hash_verification="embedded hashes matched to measurement inventory; tensor bodies not rehashed",
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         matplotlib=matplotlib.__version__)), ensure_ascii=False, indent=2)+"\n")
    for s in summaries:
        print(f"K={s['K']:g}: n={s['start']}–{s['end']}, gamma={s['gamma']:.9f}, "
              f"omega={s['omega_deg']:.6f} deg, relative residual={s['relative_residual']:.3g}")
    print(f"Validated {len(summaries)} primary fits, {len(windows)} sensitivity windows; "
          f"{len(INPUT_HASHES)} inputs unchanged.")


if __name__ == "__main__":
    main()
