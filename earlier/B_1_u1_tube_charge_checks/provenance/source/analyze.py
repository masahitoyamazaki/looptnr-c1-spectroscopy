"""Finite-chi diagnostics from saved spectra. Does not run or modify RG tensors."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import tomli
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[2] / "results/perturbation_chi40_cu1_u1_20261021"
SOURCE = ROOT.parent / 'flow_chi40_cu1_u1_20261020/results'
PREVIOUS = ROOT.parent / 'tube_chi40_cu1_u1_20261021/results'
OUT = ROOT / 'analysis'
SYMS = ['CU1Irrep', 'U1Irrep']
CS = [.5, 1., 1.4, 1.7, 1.9, 1.98]
LB = math.log(2)/2

def exact_k(c):
    return math.pi/(2*math.acos(1-c*c/2))

def dumpcsv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stable_even(row):
    return row['amplitude_end']>0 and row['normalized_rmse']<.02 and row['sensitivity_max']-row['sensitivity_min']<.5

def load():
    paths = list(SOURCE.glob('*Irrep/c_*/measurements/step_*.toml'))
    paths += list(PREVIOUS.glob('*Irrep/c_*/step_*_wide.toml'))
    paths += list((ROOT/'results').glob('*Irrep/c_*/step_*_wide.toml'))
    index, provenance = {}, []
    for p in sorted(paths):
        d = tomli.loads(p.read_text()); o = d['observables'].copy()
        tube = 'small' if p.parent.name == 'measurements' else 'wide'
        o.update(tube=tube, record=os.path.relpath(p, ROOT))
        if o['rg_step'] < 15:
            continue
        k = exact_k(o['vertex_c']); o['K_exact'] = k
        for q in range(1, 4):
            plus = o.get(f'x{q}', np.nan)
            minus = o.get(f'xm{q}', np.nan) if o['symmetry'] == 'U1Irrep' else plus
            o[f'E{q}'] = (plus+minus)/2
            o[f'O{q}'] = (plus-minus)/2
            o[f'exact{q}'] = q*q/(4*k)
            o[f'dE{q}'] = o[f'E{q}']-o[f'exact{q}']
        odd = np.array([o[f'O{q}'] for q in range(1,4)])
        charges = np.arange(1,4)
        o['B'] = float(odd @ charges / 14)
        o['odd_nonlinearity'] = (float(np.linalg.norm(odd-o['B']*charges)/np.linalg.norm(odd)) if np.linalg.norm(odd)>0 else 0.) if np.all(np.isfinite(odd)) else np.nan
        o['R2'] = o['E2']/4-o['E1']; o['R3'] = o['E3']/9-o['E1']
        o['G2'] = o['E2']/(4*o['E1'])-1 if o['E1'] else np.nan
        o['G3'] = o['E3']/(9*o['E1'])-1 if o['E1'] else np.nan
        aspect = .5 if tube=='wide' else 1.
        o['fixed_metric_E1'] = o['E1']*o.get('tau_tm_imag',np.nan)/aspect
        leading = [v for v in d.get('spectra',[]) if v['rank']==1]
        relres = [v.get('residual_relative', np.nan) for v in leading]
        o['max_leading_relative_residual'] = max(relres, default=np.nan)
        o['all_requested_converged'] = all(v.get('converged',0)>=v.get('requested',6) for v in d.get('sectors',[]))
        o['warnings'] = str(d.get('warnings',''))
        key = (o['symmetry'], o['vertex_c'], o['rg_step'], tube)
        if key in index:
            raise RuntimeError(f'Duplicate measurement {key}')
        index[key] = o
        provenance.append({'record':o['record'], 'sha256':sha(p), 'checkpoint_sha256':o.get('checkpoint_sha256','')})
    return index, provenance

def series(index, sym, c, tube, lo=15, hi=70):
    return [index[sym,c,n,tube] for n in range(lo,hi+1) if (sym,c,n,tube) in index]

def expfit(n, y, background=0, rate=None):
    """Separable nonlinear fit; amplitudes/offsets linear, rate profiled."""
    n=np.asarray(n,dtype=float); y=np.asarray(y,dtype=float); t=n-n[-1]
    def fit(r):
        cols=[np.ones(len(n))]
        if background: cols.append(t)
        cols.append(np.exp(r*t))
        X=np.array(cols).T
        coeff=np.linalg.lstsq(X,y,rcond=None)[0]
        pred=X@coeff
        return float(np.sum((pred-y)**2)), coeff, pred
    if rate is None:
        # A grid followed by local minimization avoids an assumed unimodal profile.
        grid=np.linspace(.03,1.5,150); losses=[fit(r)[0] for r in grid]
        j=int(np.argmin(losses))
        opt=minimize_scalar(lambda r:fit(r)[0],bounds=(grid[max(j-1,0)],grid[min(j+1,len(grid)-1)]),method='bounded',options={'xatol':1e-10})
        rate=float(opt.x)
    loss,coeff,pred=fit(rate)
    return {'rate':rate,'power':rate/LB,'offset':float(coeff[0]),'drift':float(coeff[1]) if background else 0.,
            'amplitude_end':float(coeff[-1]),'rmse':math.sqrt(loss/len(n)),
            'normalized_rmse':math.sqrt(loss/len(n))/(np.ptp(y) or np.nan),
            'start':int(n[0]),'end':int(n[-1]),'npoints':len(n),'background':background}, pred

def even_fits(index):
    rows=[]; sensitivity=[]
    for sym in SYMS:
        for c in CS[1:]:
            data=series(index,sym,c,'wide')
            # First connected weak-response interval. Both average gap and odd split bounded.
            good=[]
            for o in data:
                valid=(o['status']=='ok' and abs(o['dE1'])/o['exact1']<.10 and
                       abs(o['central_charge']-1)<.2 and abs(o['O1'])/o['exact1']<.4)
                if not valid: break
                good.append(o)
            if len(good)<8:continue
            end=good[-1]['rg_step']
            for length in [8,10,12]:
                for shorten in [0,1,2]:
                    hi=end-shorten; lo=max(15,hi-length+1)
                    wide=series(index,sym,c,'wide',lo,hi)
                    if len(wide)!=hi-lo+1 or len(wide)<8:continue
                    for bg in [0,1]:
                        for q in [1,2,3]:
                            f,_=expfit([o['rg_step'] for o in wide],[o[f'E{q}'] for o in wide],bg)
                            sensitivity.append(dict(symmetry=sym,vertex_c=c,charge=q,**f))
            lo=max(15,end-9)
            wide=series(index,sym,c,'wide',lo,end)
            if len(wide)!=end-lo+1:continue
            for q in [1,2,3]:
                f,_=expfit([o['rg_step'] for o in wide],[o[f'E{q}'] for o in wide],1)
                small=series(index,sym,c,'small',lo,end)
                fs,_=expfit([o['rg_step'] for o in small],[o[f'E{q}'] for o in small],1,rate=f['rate'])
                ratio=f['amplitude_end']/fs['amplitude_end']
                subset=[v['power'] for v in sensitivity if v['symmetry']==sym and v['vertex_c']==c and v['charge']==q]
                expected=2*(2-exact_k(c))
                row=dict(symmetry=sym,vertex_c=c,charge=q,K_exact=exact_k(c),
                                 expected_power=expected,**f,
                                 sensitivity_min=min(subset),sensitivity_max=max(subset),
                                 amplitude_small=fs['amplitude_end'],tube_ratio=ratio,
                                 tube_power=math.log(ratio,2) if ratio>0 else np.nan,
                                 inferred_dimension_second_order=2-f['power']/2)
                row['stable_window_diagnostic']=stable_even(row)
                rows.append(row)
    return rows,sensitivity

def oddfit(data):
    n=np.array([o['rg_step'] for o in data],float); y=np.array([o['O1'] for o in data]); t=n-n[-1]
    def fit(rate):
        X=np.exp(rate*t)[:,None]*np.array([np.cos(3*np.pi*n/4),np.sin(3*np.pi*n/4)]).T
        coeff=np.linalg.lstsq(X,y,rcond=None)[0]; pred=X@coeff
        return np.sum((y-pred)**2),coeff,pred
    opt=minimize_scalar(lambda r:fit(r)[0],bounds=(.1,.7),method='bounded',options={'xatol':1e-10})
    loss,coeff,pred=fit(opt.x)
    return dict(rate=float(opt.x),power=float(opt.x/LB),amplitude_end=float(np.linalg.norm(coeff)),
                cos_coeff=float(coeff[0]),sin_coeff=float(coeff[1]),rmse=float(np.sqrt(loss/len(n))),
                normalized_rmse=float(np.sqrt(loss/np.sum(y*y))),start=int(n[0]),end=int(n[-1]),npoints=len(n)),pred

def odd_fits(index):
    rows=[]; sens=[]
    for c in CS:
        data=series(index,'U1Irrep',c,'wide'); good=[]
        for o in data:
            if (o['status']!='ok' or abs(o['O1'])/o['exact1']>.25 or abs(o['dE1'])/o['exact1']>.1 or abs(o['central_charge']-1)>.2):break
            good.append(o)
        if len(good)<10:continue
        end=good[-1]['rg_step'];lo=max(18,end-11)
        d=series(index,'U1Irrep',c,'wide',lo,end)
        if len(d)!=end-lo+1:continue
        f,_=oddfit(d)
        small=series(index,'U1Irrep',c,'small',lo,end)
        # Direct same-step ratio projection, robust at oscillation nodes.
        bs=np.array([o['O1'] for o in small]); bw=np.array([o['O1'] for o in d]); ratio=float(bs@bw/(bs@bs))
        charge_res=float(np.sqrt(sum(sum((o[f'O{q}']-q*o['B'])**2 for q in range(1,4)) for o in d)/sum(sum(o[f'O{q}']**2 for q in range(1,4)) for o in d)))
        for short in [0,1,2]:
            for length in [8,10,12]:
                dd=series(index,'U1Irrep',c,'wide',max(18,end-short-length+1),end-short)
                if len(dd)<8:continue
                ff,_=oddfit(dd);sens.append(dict(vertex_c=c,**ff))
        pp=[v['power'] for v in sens if v['vertex_c']==c]
        rows.append(dict(vertex_c=c,K_exact=exact_k(c),**f,sensitivity_min=min(pp),sensitivity_max=max(pp),
                         tube_ratio=ratio,tube_ratio_relative_residual=float(np.linalg.norm(bw-ratio*bs)/np.linalg.norm(bw)),
                         charge_linearity_residual=charge_res))
    return rows,sens

def ratio_fits(index, evenrows):
    rows=[]
    for main in evenrows:
        if main['charge']!=1:continue
        sym,c=main['symmetry'],main['vertex_c']
        for quantity in ['G2','G3','fixed_metric_E1']:
            data=series(index,sym,c,'wide',main['start'],main['end'])
            f,_=expfit([o['rg_step'] for o in data],[o[quantity] for o in data],1)
            rows.append(dict(symmetry=sym,vertex_c=c,quantity=quantity,**f))
    return rows

def summary(index, er, orr):
    plan=json.loads((ROOT/'measurement_plan.json').read_text())
    keys=[(v['symmetry'],v['vertex_c'],v['rg_step'],v['tube']) for v in plan['all_dense_points']]
    missing=[k for k in keys if k not in index]
    dense=[index[k] for k in keys if k in index]
    counts={}
    for o in dense:counts[o['status']]=counts.get(o['status'],0)+1
    quality=[]
    for sym in SYMS:
        for c in CS:
            d=[o for o in dense if o['symmetry']==sym and o['vertex_c']==c]
            q=dict(symmetry=sym,vertex_c=c,K_exact=exact_k(c),points=len(d),end=max(o['rg_step'] for o in d))
            for threshold in [.01,.05,.1]:
                # Earliest 3 consecutive measured steps; a persistent practical error threshold.
                for component in ['even','raw']:
                    hit=[]
                    for o in d:
                        err=abs(o['dE1'])/o['exact1'] if component=='even' else max(abs(o['dE1']+o['O1']),abs(o['dE1']-o['O1']))/o['exact1']
                        if np.isfinite(err) and err>threshold:hit.append(o['rg_step'])
                    onset=next((n for n in sorted(hit) if n+1 in hit and n+2 in hit),None)
                    q[f'{component}_{int(100*threshold)}pct_onset']=onset
            quality.append(q)
    dumpcsv('onsets.csv',quality)
    fitkeys=set()
    for r in er:
        if r['charge']==1:
            fitkeys.update((r['symmetry'],r['vertex_c'],n,'wide') for n in range(r['start'],r['end']+1))
    for r in orr:fitkeys.update(('U1Irrep',r['vertex_c'],n,'wide') for n in range(r['start'],r['end']+1))
    resid=[index[k]['max_leading_relative_residual'] for k in fitkeys]
    result={'planned_dense_points':len(keys),'completed_dense_points':len(dense),'missing':missing,'dense_status_counts':counts,
            'new_measurement_files':len(list((ROOT/'results').glob('*Irrep/c_*/step_*_wide.toml'))),
            'new_cft_seconds_sum':sum(o.get('seconds',0) for o in dense if o['record'].startswith('results/')),
            'fit_max_leading_relative_residual':max(resid,default=None),'fit_measurement_points':len(fitkeys),
            'new_RG_steps':0,'new_tensor_files':0}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def makeplots(index,er,orr):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/chi40_perturbation_mpl')
    os.environ.setdefault('XDG_CACHE_HOME','/tmp/chi40_perturbation_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=['#2865a8','#db7b14','#188a72']; symcolors=['#2865a8','#d85d39']
    ef={(r['symmetry'],r['vertex_c']):r for r in er if r['charge']==1}
    of={r['vertex_c']:r for r in orr}
    def save(fig,name):
        fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    # Every newly measured point is shown; x-axis ends at continuous coverage.
    plan=json.loads((ROOT/'measurement_plan.json').read_text())['all_dense_points']
    ends={(s,c):max(o['rg_step'] for o in plan if o['symmetry']==s and o['vertex_c']==c) for s in SYMS for c in CS}
    fig,axs=plt.subplots(2,6,figsize=(18,7),constrained_layout=True)
    for i,s in enumerate(SYMS):
        for j,c in enumerate(CS):
            ax=axs[i,j];data=series(index,s,c,'wide',15,ends[s,c]);n=[o['rg_step'] for o in data]
            for q,col in zip(range(1,4),colors):
                ax.plot(n,[o[f'E{q}']/o[f'exact{q}'] for o in data],'.-',color=col,ms=3,lw=1,label=f'charge-even q={q}')
            if s=='U1Irrep':
                for sign,ls in [(1,'--'),(-1,':')]:
                    ax.plot(n,[(o['E1']+sign*o['O1'])/o['exact1'] for o in data],ls,color='.45',lw=.8,label=f'q={sign:+}')
            ax.axhline(1,color='k',ls=':',lw=1);ax.set_ylim(.5,1.5);ax.set_xlim(14,ends[s,c]+.5)
            ax.set_title(f'{s[:-5]}, $c_v={c:g}$\n$K={exact_k(c):.3f}$');ax.grid(alpha=.2);ax.set_xlabel('RG step N')
            if j==0:ax.set_ylabel(r'$E_q / x_q^{\mathrm{exact}}$')
            if (s,c) in ef:
                r=ef[s,c];ax.axvspan(r['start'],r['end'],color='#a6b6c6',alpha=.18)
    axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8)
    fig.suptitle(r'$\chi=40$, wide tube: charged dimensions (outside 0.5--1.5 omitted); shade = even-fit window')
    save(fig,'charged_spectrum_flow')
    fig,axs=plt.subplots(2,5,figsize=(16,6.5),constrained_layout=True)
    for i,s in enumerate(SYMS):
        for j,c in enumerate(CS[1:]):
            ax=axs[i,j];r=ef.get((s,c));ax.set_title(f'{s[:-5]}, $c_v={c:g}$');ax.set_xlabel('RG step N');ax.grid(alpha=.2)
            if not r:continue
            data=series(index,s,c,'wide',r['start'],r['end']);n=np.array([o['rg_step'] for o in data]);t=n-r['end']
            y=np.array([o['E1'] for o in data])-r['offset']-r['drift']*t
            if r['amplitude_end']<=0 or r['normalized_rmse']>.02:
                ax.plot(n,[o['dE1'] for o in data],'o-',ms=3);ax.axhline(0,color='.5',lw=1)
                ax.text(.05,.85,'No stable positive-growth fit',transform=ax.transAxes,fontsize=8)
                continue
            ax.semilogy(n,y,'o',ms=4,color=symcolors[i],label='Measured minus fitted background')
            ax.semilogy(n,r['amplitude_end']*np.exp(r['rate']*t),'-',color=symcolors[i],label=f"Free fit p={r['power']:.2f}")
            ax.semilogy(n,r['amplitude_end']*np.exp(r['expected_power']*LB*t),'k--',label=f"2(2-K)={r['expected_power']:.2f}")
            ax.legend(fontsize=7)
            if not stable_even(r):
                ax.text(.04,.85,'Background sensitive: p not constrained',transform=ax.transAxes,fontsize=7,color='#a64729')
            if j==0:ax.set_ylabel('Growing part of even charge-1 gap')
    fig.suptitle(r'Weak-flow fit: $E_1=a+b(N-N_{end})+A\,2^{p(N-N_{end})/2}$; dashed theory fixes only slope')
    save(fig,'even_growth_fit')
    fig,axs=plt.subplots(1,3,figsize=(14,4.3),constrained_layout=True)
    kk=np.linspace(.5,1.65,200)
    axs[0].plot(kk,2*(2-kk),'k--',label=r'$p=2(2-K)$ (second order)')
    axs[1].plot(kk,2*(2-kk),'k--',label=r'$\log_2(A_{wide}/A_{small})=2(2-K)$')
    axs[2].plot(kk,kk,'k--',label=r'$x_\Phi=K$')
    for i,s in enumerate(SYMS):
        selected=[r for r in er if r['symmetry']==s and r['charge']==1 and stable_even(r)]
        for r in selected:
            x=r['K_exact'];p=r['power'];err=np.array([[max(0,p-r['sensitivity_min'])],[max(0,r['sensitivity_max']-p)]])
            axs[0].errorbar([x],[p],yerr=err,fmt='o',color=symcolors[i],capsize=3)
            axs[1].plot(x,r['tube_power'],'o',color=symcolors[i]);axs[2].errorbar([x],[2-p/2],yerr=err[::-1]/2,fmt='o',color=symcolors[i],capsize=3)
    for ax in axs:
        ax.set_xlabel(r'$K_{exact}$');ax.grid(alpha=.2)
        handles,labels=ax.get_legend_handles_labels();ax.legend(handles+[Line2D([],[],marker='o',ls='',color=x) for x in symcolors],labels+['CU1','U1'],fontsize=8)
    axs[0].set_ylabel('Growth power p in length');axs[1].set_ylabel('Independent tube-amplitude power');axs[2].set_ylabel(r'Inferred $x_\Phi=2-p/2$')
    fig.suptitle('Stable even fits (c_v <= 1.9): second-order interpretation; bars = window/background sensitivity, not statistical confidence')
    save(fig,'relevance_vs_K')
    fig,axs=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for ax,c in zip(axs.flat,CS):
        if c not in of:continue
        r=of[c];data=series(index,'U1Irrep',c,'wide',r['start'],r['end']);n=np.array([o['rg_step'] for o in data]);_,pred=oddfit(data)
        ax.plot(n,[o['O1'] for o in data],'o',ms=4,label=r'Measured $O_1$');ax.plot(n,pred,'-',color='#d85d39',label=f"Rotating current fit: y={r['power']:.4f}")
        ax.axhline(0,color='.6',lw=.7);ax.set_title(f'$c_v={c:g}, K={exact_k(c):.3f}$');ax.set_xlabel('RG step N');ax.set_ylabel(r'$(x_{+1}-x_{-1})/2$');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(r'U1 odd splitting: $O_1\simeq e^{y\ln(\sqrt{2})(N-N_{end})}[u\cos(3\pi N/4)+v\sin(3\pi N/4)]$')
    save(fig,'u1_rotating_current')
    fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for r in orr:
        x=r['K_exact'];y=r['power'];err=[[max(0,y-r['sensitivity_min'])],[max(0,r['sensitivity_max']-y)]]
        axs[0].errorbar([x],[y],yerr=err,fmt='o',capsize=3,color='#d85d39')
        axs[1].plot(x,r['tube_ratio'],'o',color='#d85d39')
        axs[2].semilogy(x,100*r['charge_linearity_residual'],'o',color='#d85d39')
    axs[0].axhline(1,color='k',ls='--');axs[1].axhline(2,color='k',ls='--')
    for ax in axs:ax.set_xlabel(r'$K_{exact}$');ax.grid(alpha=.2)
    axs[0].set_ylabel('Odd growth exponent y (current: 1)');axs[1].set_ylabel(r'$O_{1,wide}/O_{1,small}$ (current: 2)');axs[2].set_ylabel(r'Charge-linearity residual (%) for $O_q=qB$')
    fig.suptitle('Independent U1 current diagnostics: exponent, tube circumference, and q dependence')
    save(fig,'u1_current_checks')
    fig,axs=plt.subplots(2,6,figsize=(18,6),constrained_layout=True)
    for i,s in enumerate(SYMS):
        for j,c in enumerate(CS):
            ax=axs[i,j];data=series(index,s,c,'wide',15,ends[s,c]);n=[o['rg_step'] for o in data]
            for q,col in [(2,colors[1]),(3,colors[2])]:ax.plot(n,[100*o[f'G{q}'] for o in data],'.-',ms=3,color=col,label=f'q={q}')
            ax.axhline(0,color='k',ls=':',lw=1);ax.set_ylim(-20,5);ax.set_xlabel('RG step N');ax.set_title(f'{s[:-5]}, $c_v={c:g}$');ax.grid(alpha=.2)
            if j==0:ax.set_ylabel(r'$100[E_q/(q^2 E_1)-1]$')
    axs[0,0].legend(fontsize=8);fig.suptitle('Deviation from a Gaussian K drift: dimension ratios cancel the estimated metric; outside view omitted')
    save(fig,'non_gaussian_ratios')
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for j,c in enumerate([.5,1.]):
        data=series(index,'U1Irrep',c,'wide',15,40);n=[o['rg_step'] for o in data]
        axs[0,j].plot(n,[100*o['dE1']/o['exact1'] for o in data],'.-',label='Reported CFT dimension')
        axs[0,j].plot(n,[100*(o['fixed_metric_E1']/o['exact1']-1) for o in data],'.-',label='Gap with fixed Im(tau)=1/2')
        axs[0,j].set_ylim(-10,10);axs[0,j].set_ylabel('Even charge-1 error (%)');axs[0,j].set_title(f'U1 $c_v={c:g}$');axs[0,j].legend(fontsize=8)
        axs[1,j].plot(n,[o['tau_tm_imag'] for o in data],'.-',color='#8151a3');axs[1,j].axhline(.5,color='k',ls='--');axs[1,j].set_ylabel('Estimated Im(tau_TM)');axs[1,j].set_ylim(.4,.6)
        for i in range(2):axs[i,j].set_xlabel('RG step N');axs[i,j].grid(alpha=.2)
    fig.suptitle('U1 metric diagnostic: fixed-metric gaps isolate normalization drift; they are not independently validated scaling dimensions')
    save(fig,'u1_metric_drift')
    # Existing chi=28 data: matched small tube, no new computations or exponent fits.
    fig,axs=plt.subplots(2,6,figsize=(18,6),constrained_layout=True)
    chirows=[]
    for i,(s,folder) in enumerate(zip(SYMS,['flow_chi28_cu1_20261020','flow_chi28_u1_20261020'])):
        with (ROOT.parent/folder/'observables.csv').open() as f:old=list(csv.DictReader(f))
        for j,c in enumerate(CS):
            ax=axs[i,j]
            for chi in [28,40]:
                data=series(index,s,c,'small') if chi==40 else sorted([{k:float(v) if k in ['vertex_c','rg_step','x1','xm1'] and v else v for k,v in r.items()} for r in old if float(r['vertex_c'])==c and int(r['rg_step'])>=15],key=lambda r:r['rg_step'])
                nn=[];yy=[]
                for o in data:
                    try:
                        vals=[float(o['x1'])];vals+= [float(o['xm1'])] if s=='U1Irrep' else []
                        err=max(abs(v/(1/(4*exact_k(c)))-1) for v in vals)
                    except (ValueError,KeyError):err=np.nan
                    nn.append(o['rg_step']);yy.append(100*err)
                    chirows.append(dict(symmetry=s,vertex_c=c,chi=chi,rg_step=o['rg_step'],error_percent=100*err))
                ax.semilogy(nn,yy,'-' if chi==40 else '--',label=f'chi={chi}',lw=1.3)
            ax.set_ylim(1e-3,1e3);ax.set_xlim(14,70);ax.axhline(5,color='.5',ls=':',lw=1);ax.set_xlabel('RG step N');ax.set_title(f'{s[:-5]}, $c_v={c:g}$');ax.grid(alpha=.2)
            if j==0:ax.set_ylabel('Worst +/-1 dimension error (%)')
    axs[0,0].legend(fontsize=8);fig.suptitle('Existing chi=28/40 trajectories at the same small tube; chi improvement is not uniform')
    save(fig,'chi28_chi40_control');dumpcsv('chi28_chi40_small_errors.csv',chirows)

def main():
    OUT.mkdir(exist_ok=True)
    index,provenance=load()
    dumpcsv('observables.csv',list(index.values()))
    er,es=even_fits(index);orr,oss=odd_fits(index)
    dumpcsv('even_growth_fits.csv',er);dumpcsv('even_fit_sensitivity.csv',es)
    dumpcsv('odd_current_fits.csv',orr);dumpcsv('odd_fit_sensitivity.csv',oss)
    dumpcsv('ratio_and_fixed_metric_fits.csv',ratio_fits(index,er))
    summary(index,er,orr)
    makeplots(index,er,orr)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Wide points:',sum(k[-1]=='wide' for k in index))
    for row in er:
        if row['charge']==1:print('EVEN',row)
    for row in orr:print('ODD',row)
    return index,er,es,orr,oss

if __name__=='__main__':
    main()
