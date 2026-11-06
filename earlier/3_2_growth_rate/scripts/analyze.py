from pathlib import Path
import csv, json, math, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
from decimal import Decimal
import re
import fit_method as method
from bundle import check_csv,check_json,finish
from plot import plot
RESULTS=ROOT/'data/processed'
SOURCE=ROOT/'data/cft'
TABLE=ROOT/'data/representative_calculations.csv'
PATTERN = re.compile(r'fushio χ = 40\s+c([\d.]+) \(K = ([\d.]+)\)\s+step\s+(\d+)\s+c = ([\d.e+-]+)\s+K_m = ([\d.e+-]+)\s+\(([\d.e+-]+)\)')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def rounding_interval(token):
    value = Decimal(token)
    half = Decimal(5) * Decimal(10) ** (value.as_tuple().exponent - 1)
    return float(value - half), float(value + half)


def load():
    groups, sources = {}, {}
    for path in sorted(SOURCE.glob('sixvertex_rate_cft_fushio*log')):
        sources[str(path.relative_to(ROOT))] = sha(path)
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            m = PATTERN.match(line)
            if not m:
                continue
            cv, printed_k, step, cc, km, rel = m.groups()
            cv, step = float(cv), int(step)
            k = math.pi / (4 * math.asin(cv / 2))
            klo, khi = rounding_interval(km)
            rlo, rhi = rounding_interval(rel)
            # Both printed values come from the same unrounded K_m. Intersect
            # their rounding intervals, retaining uncertainty rather than inventing precision.
            klo, khi = max(klo, k * (1 + rlo)), min(khi, k * (1 + rhi))
            assert klo <= khi and klo > 0, (path, step, klo, khi)
            kmid = (klo + khi) / 2
            row = dict(vertex_c=cv, K=k, n=step, central_charge=float(cc),
                       km_mid=kmid, km_lo=klo, km_hi=khi,
                       X=1-k/kmid, X_lo=1-k/klo, X_hi=1-k/khi,
                       inverse=kmid/k-1, x1=1/(4*kmid),
                       source=str(path.relative_to(ROOT)), line=lineno,
                       printed_K=printed_k, printed_Km=km, printed_relative=rel)
            key = (cv, step)
            assert key not in groups, key
            groups[key] = row
    result = {}
    for row in groups.values():
        result.setdefault(row['vertex_c'], []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: row['n'])
        assert all(b['n']==a['n']+1 for a,b in zip(rows,rows[1:])), 'missing saved steps'
    representatives = sorted({float(row['K_exact']) for row in csv.DictReader(TABLE.open())
                              if row['symmetry']=='O2' and row['chi']=='40'})
    assert representatives == [.75,1.,1.25,1.5,1.75,2.5]
    # Retain five calculation-table points; replace K=1.25 by the umklapp example.
    representatives = [.60, .75, 1., 1.5, 1.75, 2.5]
    return result, sources, representatives


def window(rows, start=12, cutoff=.1, observable='X'):
    selected = []
    for row in rows:
        if row['n'] < start:
            continue
        if abs(row[observable]) > cutoff:
            break
        selected.append(row)
    return (np.array([row['n'] for row in selected]),
            np.array([row[observable] for row in selected]), selected)


def components(ns, fit, k, umklapp, lattice):
    ns = np.asarray(ns)
    t = (ns-(fit['first']+fit['last'])/2)/10
    b = fit['beta']; j = 3
    out = {'background': b[0]+b[1]*t,
           'dimer': b[2]*fit['r']**(ns-fit['last'])}
    if umklapp:
        out['umklapp'] = b[j]*2**((2-4*k)*(ns-fit['first'])); j += 1
    if lattice:
        out['alternating'] = b[j]*(-.5)**(ns-fit['first']); j += 1
        if abs(2**(2-4*k)-.5)>.01:
            out['positive_lattice'] = b[j]*.5**(ns-fit['first'])
    out['total'] = sum(out.values())
    return out


def control_fit(ns, xs, k):
    # Exactly K=2: the dimer term is constant and cannot be separated from a.
    t = (ns-(ns[0]+ns[-1])/2)/10
    cols = [np.ones(len(ns)), t, (-.5)**(ns-ns[0]), .5**(ns-ns[0])]
    names = ['constant','linear','alternating','positive_lattice']
    rate = 2**(2-k)
    if abs(k-2)>1e-8:
        cols.append(rate**(ns-ns[0])); names.append('fixed_dimer')
    design = np.column_stack(cols)
    beta = np.linalg.lstsq(design,xs,rcond=None)[0]
    prediction = design@beta
    return dict(first=int(ns[0]),last=int(ns[-1]),n=len(ns),names=names,
                beta=beta.tolist(),fixed_theory_rate=rate,
                rss=float(np.sum((prediction-xs)**2)),
                rms=float(np.sqrt(np.sum((prediction-xs)**2)/(len(ns)-len(beta)))),
                background=(design[:,:2]@beta[:2]).tolist(),
                total=prediction.tolist(),
                fixed_dimer=(cols[-1]*beta[-1]).tolist() if names[-1]=='fixed_dimer' else [0.]*len(ns))


def check_synthetic():
    n=np.arange(6,35); k=1.25
    y=.001+2e-5*(n-20)+.03*1.7**(n-n[-1])+.02*2**((2-4*k)*(n-n[0]))+.003*(-.5)**(n-n[0])+.001*.5**(n-n[0])
    result=method.fit(n,y,k,True,True)
    assert abs(result['r']-1.7)<1e-6, result
    reconstructed=components(n,result,k,True,True)['total']
    assert np.max(abs(reconstructed-y))<1e-9
    c=.001+2e-5*(n-20)+.003*(-.5)**(n-n[0])+.001*.5**(n-n[0])+.02*2**(-.5*(n-n[0]))
    control=control_fit(n,c,2.5)
    assert np.max(abs(np.array(control['total'])-c))<1e-12
    return dict(recovered_growth=result['r'],expected_growth=1.7,control_max_error=float(np.max(abs(np.array(control['total'])-c))))


def analyze(groups):
    records=[]; fits=[]; predictions=[]
    rng=np.random.default_rng(20261101)
    for cv,rows in sorted(groups.items(), reverse=True):
        k=rows[0]['K']; en,ex,erows=window(rows,6)
        plot_ns,plot_x,plot_rows=window(rows,start=1,cutoff=.3)
        rec=dict(c=cv,K=k,theory=2**(2-k),available_steps=len(rows),first_saved=rows[0]['n'],last_saved=rows[-1]['n'])
        if k>=1.9-1e-8:
            cfit=control_fit(en,ex,k)
            rec.update(status='rate_not_identified',control=cfit)
            t=(plot_ns-(cfit['first']+cfit['last'])/2)/10
            beta=dict(zip(cfit['names'],cfit['beta']))
            bg=beta['constant']+beta['linear']*t
            dim=beta.get('fixed_dimer',0.)*cfit['fixed_theory_rate']**(plot_ns-cfit['first'])
            total=bg+dim+beta['alternating']*(-.5)**(plot_ns-cfit['first'])+beta['positive_lattice']*.5**(plot_ns-cfit['first'])
            for i,row in enumerate(plot_rows):
                predictions.append(dict(K=k,n=row['n'],X=row['X'],X_lo=row['X_lo'],X_hi=row['X_hi'],background=float(bg[i]),
                                        subtracted=float(plot_x[i]-bg[i]),fit_subtracted=float(total[i]-bg[i]),
                                        dimer=float(dim[i]),umklapp=0.,
                                        in_fit_window=cfit['first']<=row['n']<=cfit['last'],kind='fixed_rate_control'))
            records.append(rec); print(f'K={k:.6f} control; fixed rate is not a measurement',flush=True);continue
        early=method.fit(en,ex,k,True,True)
        assert early is not None
        s=2**(2-4*k)
        signal=abs(early['beta'][3]*s**(12-early['first']))/early['rms']
        use_b=bool(signal>1)
        ns,xs,selected=window(rows)
        central=method.fit(ns,xs,k,use_b)
        assert central is not None
        parts=components(plot_ns,early,k,True,True)
        in_fit=(plot_ns>=early['first']) & (plot_ns<=early['last'])
        np.testing.assert_allclose(np.sum((parts['total'][in_fit]-plot_x[in_fit])**2),early['rss'],rtol=1e-8,atol=1e-24)
        for i,row in enumerate(plot_rows):
            predictions.append(dict(K=k,n=row['n'],X=row['X'],X_lo=row['X_lo'],X_hi=row['X_hi'],background=float(parts['background'][i]),
                                    subtracted=float(plot_x[i]-parts['background'][i]),fit_subtracted=float(parts['total'][i]-parts['background'][i]),
                                    dimer=float(parts['dimer'][i]),umklapp=float(parts['umklapp'][i]),
                                    in_fit_window=bool(in_fit[i]),kind='free_rate_early'))
        variants=[]
        for obs in ['X','inverse']:
            for start in [10,12,14]:
                for cutoff in [.05,.1]:
                    vn,vx,_=window(rows,start,cutoff,obs)
                    v=method.fit(vn,vx,k,use_b)
                    if v:
                        variants.append(v['r']);fits.append(dict(K=k,observable=obs,start=start,cutoff=cutoff,kind='window',**v))
        lattice=method.fit(ns,xs,k,use_b,True)
        fits.append(dict(K=k,observable='X',start=12,cutoff=.1,kind='lattice_restored',**lattice))
        # Rounding perturbations are a deterministic sensitivity check, NOT draws from a noise law.
        lo=np.array([row['X_lo'] for row in selected]);hi=np.array([row['X_hi'] for row in selected])
        rounding=[]
        for index,values in enumerate([lo,hi]+[rng.uniform(lo,hi) for _ in range(16)]):
            rf=method.fit(ns,values,k,use_b)
            rounding.append(rf['r'])
            fits.append(dict(K=k,observable='X',start=12,cutoff=.1,kind=f'rounding_{index}',**rf))
        rounding_boundaries=[row['n'] for row in rows if row['n']>=6 and
                             (row['X_lo']<=-.1<=row['X_hi'] or row['X_lo']<=.1<=row['X_hi'])]
        assert not rounding_boundaries, rounding_boundaries
        rec.update(status='estimated',early=early,main=central,umklapp_term=use_b,umklapp_signal_to_rms=signal,
                   lattice_main_r=lattice['r'],window_min=min(variants),window_max=max(variants),
                   rounding_min=min(rounding),rounding_max=max(rounding),
                   error_lo=min(central['profile_lo'],*variants,*rounding,lattice['r']),
                   error_hi=max(central['profile_hi'],*variants,*rounding,lattice['r']))
        records.append(rec)
        print(f'K={k:.6f} r={central["r"]:.6f}, B={use_b}, rounding shift={max(abs(v-central["r"]) for v in rounding):.3g}',flush=True)
    return records,fits,predictions


def main():
    synthetic=check_synthetic()
    groups,sources,reps=load()
    assert len(groups)==22
    records,fits,predictions=analyze(groups)
    assert sum(r['status']=='estimated' for r in records)==15
    write_csv(RESULTS/'input_points.csv',[r for rows in groups.values() for r in rows])
    write_csv(RESULTS/'predictions.csv',predictions)
    write_csv(RESULTS/'sensitivity_fits.csv',fits)
    (RESULTS/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    check_csv('input_points.csv',ignore=('source',))
    check_csv('predictions.csv');check_csv('sensitivity_fits.csv');check_json('summary.json')
    plot(records,predictions,reps)
    finish(dict(conditions=22,estimated_rates=15,unresolved_rates=7,synthetic=synthetic,rounded_log_input=True))
