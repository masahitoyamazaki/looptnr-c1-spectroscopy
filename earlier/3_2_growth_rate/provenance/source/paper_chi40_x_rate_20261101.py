"""Fit saved CU1 chi=40 small-tube charge-one CFT levels for paper Fig. 2.

Run from the repository root: MPLCONFIGDIR=/tmp/sixvertex_mpl python3 Fushio/code/six_vertex/scripts/paper_chi40_x_rate_20261101.py
No RG or CFT measurements are launched.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / "Fushio/code/six_vertex/results/growth_rate_chi40_cu1_dense_small_20261023/tables/charged_dimensions.csv"
OUT = ROOT / "Fushio/code/six_vertex/results/paper_chi40_x_rate_20261101"
RATE_STARTS = (1.1, 1.5, 2.0, 2.6, 3.2)


def selected(rows, observable, start, cutoff):
    """Otsuki window: from start until the first weak-shift cutoff crossing."""
    ns, xs = [], []
    for row in rows:
        n = int(row["rg_step"])
        if n < start:
            continue
        if row["valid"] != "True" or not np.isfinite(float(row["x1"])):
            continue
        k, delta = float(row["K_exact"]), float(row["x1"])
        x = 1 - 4*k*delta if observable == "shift" else 1/(4*k*delta)-1
        if abs(x) > cutoff:
            break
        ns.append(n)
        xs.append(x)
    return np.asarray(ns), np.asarray(xs)


def neldermead_1d(f, x0, step=.1, iters=4000, tol=1e-14):
    """One-parameter transcription of Otsuki's Nelder--Mead routine."""
    points = [x0, x0 + step]
    values = [f(x) for x in points]
    for _ in range(iters):
        if values[1] < values[0]:
            points.reverse(); values.reverse()
        if values[1]-values[0] <= tol*(abs(values[0])+1e-30):
            break
        center = points[0]
        reflected = 2*center-points[1]
        fr = f(reflected)
        if fr < values[0]:
            expanded = 3*center-2*points[1]
            fe = f(expanded)
            points[1], values[1] = (expanded,fe) if fe < fr else (reflected,fr)
        elif fr < values[0]:
            points[1], values[1] = reflected,fr
        else:
            contracted = .5*(center+points[1])
            fc = f(contracted)
            if fc < values[1]:
                points[1], values[1] = contracted,fc
            else:
                points[1] = .5*(points[0]+points[1]); values[1] = f(points[1])
    idx = int(np.argmin(values))
    return points[idx], values[idx]


def fit(ns, ys, k, umklapp, lattice=False):
    """Otsuki variable projection, early/main terms and RSS profile."""
    s = 2**(2-4*k)
    use_positive_lattice = lattice and abs(s-.5) > .01
    p = 4 + int(umklapp) + int(lattice) + int(use_positive_lattice)
    if len(ns)-p < 3:
        return None
    t = (ns-(ns[0]+ns[-1])/2)/10

    def solve(theta):
        exponent = theta*(ns-ns[-1])
        if np.max(exponent) > 700:
            return float("inf"), None
        cols = [np.ones(len(ns)), t, np.exp(exponent)]
        if umklapp:
            cols.append(s**(ns-ns[0]))
        if lattice:
            cols.append((-.5)**(ns-ns[0]))
        if use_positive_lattice:
            cols.append(.5**(ns-ns[0]))
        design = np.column_stack(cols)
        beta = np.linalg.lstsq(design, ys, rcond=None)[0]
        residual = ys-design@beta
        return float(residual@residual), beta

    def rss(theta):
        return solve(theta)[0]

    theta, best = None, float("inf")
    for rate0 in RATE_STARTS:
        candidate, value = neldermead_1d(rss, np.log(rate0))
        candidate, value = neldermead_1d(rss, candidate, step=.01)
        if value < best:
            theta, best = candidate, value
    if theta is None:
        return None
    residual_sum, beta = solve(theta)
    threshold = residual_sum*(1+1/(len(ns)-p))

    def edge(direction):
        inside, step = theta, .005
        while True:
            outside = inside+direction*step
            if abs(outside-theta) > 3:
                return float("nan")
            if rss(outside) > threshold:
                break
            inside, step = outside,step*1.6
        for _ in range(30):
            midpoint = (inside+outside)/2
            if rss(midpoint) > threshold:
                outside = midpoint
            else:
                inside = midpoint
        return (inside+outside)/2

    lo, hi = edge(-1),edge(1)
    return dict(r=float(np.exp(theta)), profile_lo=float(np.exp(lo)),
                profile_hi=float(np.exp(hi)), rss=residual_sum,
                rms=float(np.sqrt(residual_sum/(len(ns)-p))), n=len(ns),
                first=int(ns[0]), last=int(ns[-1]), amplitude=float(beta[2]),
                beta=[float(value) for value in beta], parameters=p)


def main():
    groups = defaultdict(list)
    with INPUT.open(newline="") as stream:
        for row in csv.DictReader(stream):
            groups[row["vertex_c"]].append(row)
    output = []
    for c, rows in groups.items():
        rows.sort(key=lambda row:int(row["rg_step"]))
        k = float(rows[0]["K_exact"])
        early_n, early_x = selected(rows,"shift",6,.10)
        early = fit(early_n,early_x,k,umklapp=True,lattice=True) if k < 1.9-1e-6 else None
        s = 2**(2-4*k)
        umklapp_ratio = (abs(early["beta"][3]*s**(12-early["first"]))/early["rms"]
                          if early else None)
        umklapp = bool(umklapp_ratio is not None and umklapp_ratio > 1)
        n, x = selected(rows,"shift",12,.10)
        central = fit(n,x,k,umklapp) if k < 1.9-1e-6 else None
        rec = dict(c=float(c),K=k,theory=2**(2-k),geometry="[sqrt(2),sqrt(2),0]",
                   status="estimated" if central else "rate_not_identified",
                   umklapp_term=umklapp, umklapp_signal_to_early_rms=umklapp_ratio,
                   available_points=len(n))
        if central:
            variants=[]
            inverse_n,inverse_x=selected(rows,"inverse",12,.10)
            inverse=fit(inverse_n,inverse_x,k,umklapp)
            for observable in ("shift","inverse"):
                for start in (10,12,14):
                    for cutoff in (.05,.10):
                        vn,vx=selected(rows,observable,start,cutoff)
                        val=fit(vn,vx,k,umklapp)
                        if val: variants.append(val["r"])
            lattice_main=fit(n,x,k,umklapp,lattice=True)
            rec.update(central)
            rec["early_r"]=early["r"]
            rec["inverse_r"]=inverse["r"] if inverse else None
            rec["lattice_main_r"]=lattice_main["r"] if lattice_main else None
            rec["lattice_at_12_to_early_rms"]=(
                sum(abs(early["beta"][i])*.5**(12-early["first"])
                    for i in range(4,len(early["beta"]))) / early["rms"])
            rec["variant_count"]=len(variants)
            rec["sensitivity_lo"]=min(variants)
            rec["sensitivity_hi"]=max(variants)
            rec["error_lo"]=min(rec["profile_lo"],rec["sensitivity_lo"])
            rec["error_hi"]=max(rec["profile_hi"],rec["sensitivity_hi"])
        output.append(rec)
    output.sort(key=lambda item:item["K"])
    assert len(output) == 16
    assert sum(item["status"] == "estimated" for item in output) == 10
    assert all(item["error_lo"] <= item["r"] <= item["error_hi"]
               for item in output if item["status"] == "estimated")
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"rates.csv").open("w",newline="") as stream:
        fields=list(dict.fromkeys(key for rec in output for key in rec))
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(output)
    (OUT/"summary.json").write_text(json.dumps(output,indent=2)+"\n")

    plt.rcParams.update({"font.size":9,"font.family":"serif","mathtext.fontset":"dejavuserif",
                         "axes.linewidth":.7,"pdf.fonttype":42})
    fig,ax=plt.subplots(figsize=(3.45,2.9),layout="constrained")
    ks=np.linspace(.55,2.85,300)
    ax.plot(ks,2**(2-ks),color="#222222",lw=1.2,label=r"$2^{2-K}$")
    fitted=[rec for rec in output if rec["status"]=="estimated"]
    ax.errorbar([v["K"] for v in fitted],[v["r"] for v in fitted],
                yerr=[[v["r"]-v["error_lo"] for v in fitted],
                      [v["error_hi"]-v["r"] for v in fitted]],
                fmt="o",markersize=4.3,capsize=2,elinewidth=.9,color="#215da3",
                markeredgecolor="white",markeredgewidth=.4,label=r"CU1, $\chi=40$",zorder=3)
    control=[rec for rec in output if rec["status"]!="estimated"]
    ax.scatter([v["K"] for v in control],[1.0]*len(control),marker="v",s=24,
               color="#888888",label=r"unresolved (at $r=1$)",zorder=3)
    ax.axhline(1,color="#aaaaaa",lw=.7,ls=":")
    ax.set(xlim=(.52,2.85),ylim=(.65,3.55),xlabel=r"$K$",ylabel=r"rate per RG step $r$")
    ax.set_yticks([.75,1,1.5,2,2.5,3,3.5])
    ax.grid(alpha=.15,lw=.5)
    ax.legend(loc="upper right",fontsize=7,frameon=False)
    fig.savefig(OUT/"figure3_chi40_x_rate.pdf")
    fig.savefig(OUT/"figure3_chi40_x_rate.png",dpi=220)
    plt.close(fig)
    for rec in output:
        print(f"K={rec['K']:.3f} {rec['status']} " +
              (f"r={rec['r']:.5f} [{rec['error_lo']:.5f},{rec['error_hi']:.5f}] N={rec['n']} variants={rec['variant_count']}" if rec['status']=='estimated' else ""))


if __name__=="__main__":
    main()
