"""V_{0,+-1} signature (main_en.tex Sec. V.C). (1) K.C. transverse-field runs (C only, K=1.5, chi=16; Drive archive
Six_vertex/_archive/Cho_backup_20261104/symmetry_validation_20261024, runs/cal_C_transverse_field_*): shift of M and S
relative to h_t=0, growth rate and ratio dS/dM.  Usage: python3 v01_signature.py RUNS_DIR_OF_EXTRACTED_ARCHIVE
Yamamoto/finite_bonddim/src/conly_cpt_fit.py is imported for the coefficients (not modified)."""
import re, sys, os, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "external", "yamamoto_conly_scan", "finite_bonddim", "src"))
import types; sys.modules['usb_stage3_ingest']=types.SimpleNamespace(memory=lambda: {})
import conly_cpt_fit as Y
R=sys.argv[1]
def parse(path):
    meas=[]; cur=None; sp=None
    for line in open(path, encoding="utf-8"):
        s=line.strip()
        if s=="[[measurements]]": cur={"spectra":[]}; meas.append(cur); sp=None; continue
        if s=="[[measurements.spectra]]": sp={}; cur["spectra"].append(sp); continue
        m=re.match(r'(\w+)\s*=\s*(.+)$', s)
        if not m or cur is None: continue
        k,v=m.groups(); v=v.strip().strip('"')
        try: v=float(v)
        except ValueError: pass
        (sp if sp is not None else cur)[k]=v
    return meas
K=1.5; xs=1/(4*K)
def series(run):
    out={}
    for m in parse(os.path.join(R,run,"result.toml")):
        if "step" not in m: continue
        ev=sorted(s["x"] for s in m["spectra"] if s.get("sector")=="Irrep[ℤ₂](0)")[1:]
        od=sorted(s["x"] for s in m["spectra"] if s.get("sector")=="Irrep[ℤ₂](1)")
        if not ev or not od: continue
        e=min(ev,key=lambda x:abs(x-xs)); o=min(od,key=lambda x:abs(x-xs))
        out[int(m["step"])]=dict(n=int(m["step"]),K=K,xstar=xs,central_charge=m.get("central_charge",1),x_Ceven=e,x_Codd=o,M=(e+o)/2,S=(e-o)/2,valid=True)
    return out
base=series("baseline_C_K1p5_chi16")
c=Y.coefficients(K)
print(f"r1 = 2^(2-1/4K) = {c['r_1']:.3f}, r2 = {c['r_2']:.3f}, b_S/b_M = {c['beta_split']/c['beta_mean']:.3f}")
runs={"cal_C_transverse_field_0":-2e-6,"cal_C_transverse_field_1":-1e-6,"cal_C_transverse_field_2":1e-6,"cal_C_transverse_field_3":2e-6}
print(" h_t     | dM(n)=M-M0 and dS(n)=S-S0 at n=10,12,14,16,18,20 | dS/dM | growth of dM per step (n=10-18)")
for r,h in runs.items():
    s=series(r+"_K1p5_chi16"); ns=[n for n in (10,12,14,16,18,20) if n in s and n in base]
    dM=[s[n]["M"]-base[n]["M"] for n in ns]; dS=[s[n]["S"]-base[n]["S"] for n in ns]
    nn=np.array([n for n in range(10,19) if n in s and n in base]); y=np.array([abs(s[n]["M"]-base[n]["M"]) for n in nn])
    g=math.exp(np.polyfit(nn,np.log(y+1e-30),1)[0]) if len(nn)>3 else float('nan')
    print(f"{h:+.0e} | " + " ".join(f"{a:+.1e}/{b:+.1e}" for a,b in zip(dM,dS)) + f" | {np.median([b/a for a,b in zip(dM,dS) if a]):+.2f} | {g:.3f}")
