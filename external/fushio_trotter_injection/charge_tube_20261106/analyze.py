"""Note13: same-checkpoint charged readout, larger physical spatial tube."""
import csv, hashlib, json, os, sys, tomllib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
ROOT=H.parents[5]
sys.path.insert(0,str(ROOT/'Fushio/docs/six_vertex/o2_u1_coupling_readout_20261101'))
from coefficients import alpha
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/note13_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
STEPS=list(range(0,17,2)); KEYS=['m1','m2','m3','X']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_rows():
    rows=[]
    for p in sorted((H/'data').glob('*/*.toml')):
        d=tomllib.loads(p.read_text());K=d['K_vertex'];s=d['sectors'];xs=[t['x'] for t in s[1:]]
        aa=[alpha(K,m) for m in (1,2,3)];ax=15*aa[0]-6*aa[1]+aa[2]
        rows.append(dict(tube='141',h=d['h'],step=d['step'],K=K,x=xs,
            residual=max(t['residual'] for sector in s for t in sector['trials']),phase=max(abs(t['phase']) for t in s),
            agreement=max(t['start_agreement'] for t in s),status=d['status'],
            G={**{f'm{m}':(xs[m-1]-m*m/(4*K))/aa[m-1] for m in (1,2,3)},'X':(15*xs[0]-6*xs[1]+xs[2])/ax},
            coefficients=dict(alpha=aa,AX=ax),source=str(p.relative_to(ROOT)),sha256=digest(p),checkpoint_sha256=d['checkpoint_sha256']))
    old=H.parent/'perturbed_chi32_20261105/native_121/data'
    for p in sorted(old.glob('stack_K1.5*.toml')):
        d=tomllib.loads(p.read_text())
        for v in d['measurements']:
            if v['step'] not in STEPS:continue
            s=[next((s['levels'][0] for s in v['sectors'] if s['sector'].endswith(f'({m}, 2)')),None) for m in (1,2,3)]
            K=v['K_vertex'];xs=[t['x'] if t else None for t in s];aa=[alpha(K,m) for m in (1,2,3)];ax=15*aa[0]-6*aa[1]+aa[2]
            rows.append(dict(tube='121',h=d['h'],step=v['step'],K=K,x=xs,residual=max(t['residual'] for t in s if t),phase=max(abs(t['phase']) for t in s if t),agreement=0.,status='measured',
                G={**{f'm{m}':(xs[m-1]-m*m/(4*K))/aa[m-1] if xs[m-1] is not None else None for m in (1,2,3)},'X':(15*xs[0]-6*xs[1]+xs[2])/ax if all(x is not None for x in xs) else None},source=str(p.relative_to(ROOT)),sha256=digest(p),checkpoint_sha256=v['checkpoint_sha256']))
    baseline={(r['tube'],r['step']):r for r in rows if r['h']==0}
    for r in rows:
        r['numeric_pass']=r['status']=='measured' and r['residual']<1e-8 and r['agreement']<1e-8
        r['weak_pass']=r['numeric_pass'] and abs(4*r['K']*r['x'][0]-1)<.1 and r['phase']<.05
        r['relative_level_errors']=[4*r['K']*x/(m*m)-1 if x is not None else None for m,x in enumerate(r['x'],1)]
        r['x1_minus_cft']=r['x'][0]-1/(4*r['K'])
        b=baseline.get((r['tube'],r['step']))
        if r['h'] and b:
            r['dG']={k:r['G'][k]-b['G'][k] if r['G'][k] is not None and b['G'][k] is not None else None for k in KEYS}
            r['tube_scale']=2**(4-2*r['K']) if r['tube']=='141' else 1.
            r['dG_at_small_L']={k:v/r['tube_scale'] if v is not None else None for k,v in r['dG'].items()}
            r['pair_pass']=r['weak_pass'] and b['numeric_pass'] and abs(4*b['K']*b['x'][0]-1)<.1 and b['phase']<.05
            a=list(r['dG'].values());r['spread']=(max(a)-min(a))/np.mean(a) if all(v is not None and v>0 for v in a) else None
    return rows

def plot_lowest_charged(rows):
    plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.2,'figure.dpi':160})
    F=H/'figures';F.mkdir(exist_ok=True)
    wide=[r for r in rows if r['tube']=='141'];K=wide[0]['K']
    fig,ax=plt.subplots(figsize=(8,4.2),layout='constrained')
    for i,h in enumerate([0.,.01,.02]):
        rr=sorted([r for r in wide if r['h']==h],key=lambda r:r['step'])
        delta=[r['x1_minus_cft'] for r in rr]
        assert all(v>0 for v in delta),'Signed deviation is not positive; reconsider log plotting explicitly'
        ax.plot([r['step'] for r in rr],delta,'-o',color=f'C{i}',ms=4,label=f'h={h:g}')
    ax.set_yscale('log')
    ax.grid(which='minor',axis='y',alpha=.1)
    ax.set_xticks(STEPS);ax.set_xlabel('RG step');ax.set_ylabel(r'$\Delta x_1=x_{m=1}-1/(4K_0)$ (log scale)')
    ax.set_title(r'Tube [1,4,1], $\chi=32$, $K_0=1.4995965$')
    ax.legend(loc='upper left')
    for ext in ('png','svg'):fig.savefig(F/f'lowest_charged_141.{ext}')
    plt.close(fig)

def plot(rows):
    plot_lowest_charged(rows)
    plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.2,'figure.dpi':160})
    F=H/'figures';F.mkdir(exist_ok=True)
    def selected(t,h):return sorted([r for r in rows if r['tube']==t and r['h']==h and 'dG' in r],key=lambda r:r['step'])
    def curves(ax,rr,field):
        for i,k in enumerate(KEYS):
            ax.plot([r['step'] for r in rr],[r[field][k] for r in rr],'-',color=f'C{i}',lw=1.5,label=k)
            for r in rr:
                if r[field][k] is None:continue
                ax.plot(r['step'],r[field][k],'o',color=f'C{i}',ms=4,mfc=f'C{i}' if r['pair_pass'] else 'white')
        ax.axhline(0,color='k',lw=.7);ax.set_xticks(STEPS);ax.set_xlabel('RG step')
        ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,h in zip(axs,[.01,.02]):
        curves(ax,selected('141',h),'dG');ax.set_title(f'[1,4,1], h={h:g}');ax.set_ylabel(r'$\delta\widehat{g_d^2}$');ax.legend(ncol=2)
    fig.suptitle(r'$\chi=32$, $K_0=1.4995965$; matched $h=0$ subtraction')
    for ext in ('png','svg'):fig.savefig(F/f'couplings_141.{ext}')
    plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained',sharey='col')
    for j,h in enumerate([.01,.02]):
        for i,t in enumerate(['121','141']):
            curves(axs[i,j],selected(t,h),'dG_at_small_L')
            axs[i,j].set_title(f'h={h:g}; '+('[1,2,1]' if t=='121' else r'[1,4,1] / $2^{4-2K_0}$'))
            axs[i,j].set_ylabel(r'$\delta\widehat{g_d^2}$ at small-tube scale');axs[i,j].legend(ncol=2)
    fig.suptitle(r'$\chi=32$, $K_0\simeq1.5$: same-tensor tube comparison')
    for ext in ('png','svg'):fig.savefig(F/f'tube_comparison.{ext}')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for t,style in [('121','--'),('141','-')]:
        rr=sorted([r for r in rows if r['tube']==t and r['h']==0],key=lambda r:r['step'])
        for i in range(3):axs[0].plot([r['step'] for r in rr],[100*r['relative_level_errors'][i] if r['relative_level_errors'][i] is not None else np.nan for r in rr],style+'o',ms=3,color=f'C{i}',label=f'm{i+1} [{t[0]},{t[1]},{t[2]}]')
        for i,h in enumerate([.01,.02]):
            rr=selected(t,h);axs[1].plot([r['step'] for r in rr],[100*r['spread'] if r['spread'] is not None else np.nan for r in rr],style,color=f'C{i}',label=f'h={h:g} [{t[0]},{t[1]},{t[2]}]')
            for r in rr:
                if r['spread'] is not None:axs[1].plot(r['step'],100*r['spread'],'o',ms=3,color=f'C{i}',mfc=f'C{i}' if r['pair_pass'] else 'white')
    axs[0].set_title('Unperturbed charged levels');axs[0].set_ylabel('Relative deviation from CFT (%)');axs[0].axhline(0,color='k',lw=.7)
    axs[1].set_title('Spread among m1, m2, m3, X');axs[1].set_ylabel('(max - min) / mean (%)');axs[1].set_yscale('log')
    for ax in axs:ax.set_xticks(STEPS);ax.set_xlabel('RG step');ax.legend(fontsize=9,ncol=2)
    axs[1].legend(fontsize=9,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.2))
    fig.suptitle(r'$\chi=32$, $K_0\simeq1.5$: background and readout consistency')
    for ext in ('png','svg'):fig.savefig(F/f'background_spread.{ext}')
    plt.close(fig)

def main():
    rows=read_rows();(H/'analysis.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
    with (H/'observables.csv').open('w') as f:
        names=['tube','h','step','x1','x2','x3','x1_minus_cft',*KEYS,'spread','pair_pass','residual','phase'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for r in rows:w.writerow(dict(tube=r['tube'],h=r['h'],step=r['step'],**{f'x{i+1}':x for i,x in enumerate(r['x'])},x1_minus_cft=r['x1_minus_cft'],**r.get('dG',{}),spread=r.get('spread'),pair_pass=r.get('pair_pass'),residual=r['residual'],phase=r['phase']))
    plot(rows)
    print('wide points',sum(r['tube']=='141' for r in rows))
    for r in rows:
        if r['tube']=='141' and r['h']:print(r['h'],r['step'],r.get('dG'),r.get('spread'),r.get('pair_pass'))
if __name__=='__main__':main()
