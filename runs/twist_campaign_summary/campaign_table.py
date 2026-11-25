"""LaTeX rows of the campaign table of Paper 1b (App. C) from campaign_summary.csv and refit_b9_table.csv (mu*).
Usage: python campaign_table.py <refit_b9_table.csv> > campaign_rows.tex"""
import csv, math, sys
mus = {(float(r["K"]), int(r["chi"])): float(r["mu_star"]) for r in csv.DictReader(open(sys.argv[1]))}
def sci(x, d=2):
    if x is None or x != x: return "---"
    e = int(math.floor(math.log10(abs(x)))); m = x / 10**e
    return f"${m:+.{d}f}\\times10^{{{e}}}$".replace("+", "") if x > 0 else f"${m:.{d}f}\\times10^{{{e}}}$"
for r in csv.DictReader(open("campaign_summary.csv")):
    K, chi = float(r["K"]), int(r["chi"]); ms = mus.get((K, chi))
    mp = float(r["mu_pred"]); st = r["status"]
    ne = f"${r['ne_zero']}\\to{r['ne_pred']}$" if r["ne_pred"] else f"${r['ne_zero']}$"
    fc = float(r["fcal"]); ds = float(r["delay_src"])
    fcs = "---" if fc != fc else (f"${100*fc:.1f}$" if abs(fc) < 0.3 else "sign change")
    dss = "---" if (ds != ds or abs(fc) >= 0.3) else f"${ds:.0f}$"
    note = {"held": " (held)", "no calibration": " (no $\\mupred$)"}.get(st, "")
    print(f"      {K:.2f} & {chi} & {r['n_cal']} & {sci(mp) if mp == mp else '---'} & {sci(ms, 1) if ms else '---'} & {ne}{note} & "
          f"{('$'+r['delay']+'$') if r['delay'] else '---'} & {fcs} & {dss} \\\\")
