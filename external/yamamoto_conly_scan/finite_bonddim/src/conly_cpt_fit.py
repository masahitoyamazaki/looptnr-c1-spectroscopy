"""AI-generated approximate simultaneous CPT fits; existing CSV only, no RG.

Closed forms transcribed from Otsuki's perturbation-results PDF, Eqs.16/17.
No Otsuki code is imported. Couplings follow homogeneous linearized RG within
each short window; ongoing truncation sources are not independently fitted.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import csv,json,hashlib,math,argparse,gc
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import gamma,rgamma
from usb_stage3_ingest import memory

ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'work/mplconfig')
INPUT=ROOT/'data/yamamoto_conly_scan_20261026_v1/analysis/inputs.csv'
OUT=ROOT/'data/conly_cpt_20261029'
FIG=ROOT/'figures/conly_cpt_20261029'

def guard():
    m=memory()
    if m['available_bytes']<180*1024**2 or m['working_bytes']>256*1024**2:raise MemoryError(m)
    return m

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

def gam(x):return gamma(x)*rgamma(1-x)

def unpaired(k):
    def integ(m):
        return math.pi*gamma(1-k)*rgamma(k)*(gamma((k+m)/2)*rgamma(1+(m-k)/2))**2
    return -math.pi*(integ(1)-integ(0))

def paired(k,sign):
    return -math.pi**2*(gam(3*k/2)*gam(1-k)*gam(-k/2)+sign*gam(-k/2)**2*gam(1+k)/2-gam(k/2)**2*gam(1-k))

def coefficients(K):
    k=float(K);q=1/(4*k)
    if K<2:
        # The m=1 primary has a removable singularity at the free-fermion point.
        a=4*math.pi**2*math.log(2) if abs(K-1)<1e-10 else unpaired(k)
    else:a=0. # Marginal/irrelevant d omitted, not analytically continued across K=2.
    be,bo=paired(q,1),paired(q,-1)
    return dict(K=K,alpha_d=float(a),beta_even=float(be),beta_odd=float(bo),
                beta_mean=float((be+bo)/2),beta_split=float((be-bo)/2),
                r_d=2**(2-K),r_1=2**(2-1/(4*K)),r_2=2**(1-1/(2*K)),d_included=K<2)

def design(n,c,sfloor=True):
    t=np.asarray(n,float)-max(n);z=np.zeros(len(t));one=np.ones(len(t))
    names=['M0','Mdrift'];cols=[np.r_[one,z],np.r_[t,z]]
    if sfloor:names+=['S0'];cols+=[np.r_[z,one]]
    if c['d_included']:
        names+=['gd2'];cols+=[np.r_[c['alpha_d']*c['r_d']**t,z]]
    names+=['g1sq','g2']
    cols+=[np.r_[c['beta_mean']*c['r_1']**t,c['beta_split']*c['r_1']**t],
           np.r_[z,math.sqrt(2)*math.pi*c['r_2']**t]]
    return names,np.array(cols).T

def solve(part,c,sfloor=True,drop=None):
    n=[r['n'] for r in part];xs=1/(4*c['K'])
    y=np.r_[[r['M']-xs for r in part],[r['S'] for r in part]]
    names,A=design(n,c,sfloor)
    if drop in names:
        j=names.index(drop);names.pop(j);A=np.delete(A,j,axis=1)
    norms=np.linalg.norm(A,axis=0);As=A/norms
    lower=np.array([0. if x in ('gd2','g1sq') else -np.inf for x in names])
    fit=lsq_linear(As,y,bounds=(lower,np.full(len(names),np.inf)),tol=1e-13,max_iter=1000)
    v=fit.x/norms;pred=A@v;N=len(part);dM=pred[:N];S=pred[N:]
    E=xs+dM+S;O=xs+dM-S
    err=np.r_[E-np.array([r['x_Ceven'] for r in part]),O-np.array([r['x_Codd'] for r in part])]
    # Baseline uses exactly the same background columns and observations.
    bg=[j for j,s in enumerate(names) if s in ('M0','Mdrift','S0')]
    by=A[:,bg]@np.linalg.lstsq(A[:,bg],y,rcond=None)[0]
    result=dict(n_min=min(n),n_max=max(n),npoints=N,S_floor=sfloor,success=bool(fit.success),
        condition=float(np.linalg.cond(As)),rank=int(np.linalg.matrix_rank(As)),parameters=len(names),
        branch_rmse_rel=float(np.sqrt(np.mean(err**2))/xs),
        M_rmse_rel=float(np.sqrt(np.mean((dM-y[:N])**2))/xs),
        S_rmse_rel=float(np.sqrt(np.mean((S-y[N:])**2))/xs),
        improvement=float(1-np.sum((pred-y)**2)/max(np.sum((by-y)**2),1e-30)),
        **{s:float(x) for s,x in zip(names,v)})
    # Signed unconstrained solution diagnoses incompatibility with real squared couplings.
    u=np.linalg.lstsq(As,y,rcond=None)[0]/norms
    result.update({f'unconstrained_{s}':float(u[j]) for j,s in enumerate(names) if s in ('gd2','g1sq')})
    return result

def inputs():
    groups=defaultdict(list)
    for r in read(INPUT):
        for k in ('K','xstar','central_charge','M','S','x_Ceven','x_Codd'):r[k]=float(r[k])
        r['n']=int(r['n']);r['chi']=int(r['chi']);r['valid']=r['valid']=='True'
        groups[r['source'],r['run_id']].append(r)
    return groups

def weak(rr,start):
    by={r['n']:r for r in rr};part=[]
    for n in range(start,max(by)+1):
        r=by.get(n)
        if r is None or not r['valid'] or abs(r['central_charge']-1)>=.2:break
        if max(abs(r[k]/r['xstar']-1) for k in ('x_Ceven','x_Codd'))>=.1:break
        part.append(r)
    return part

def checks():
    a=coefficients(1)['alpha_d'];target=float(4*math.pi**2*math.log(2))
    assert abs(a/target-1)<1e-12
    c=coefficients(1.3)
    dual=sorted([c['beta_even'],c['beta_odd']])
    assert abs(dual[0]-171.5)<.1 and abs(dual[1]-375.9)<.1
    assert abs(paired(.75,1)-83.53)<.01 and abs(paired(.75,-1)-48.78)<.01
    # A nonzero synthetic coupling of each type must be recovered simultaneously.
    c=coefficients(.75);n=list(range(12,22));names,A=design(n,c)
    v=np.array([.001,2e-5,1e-5,1e-5,2e-6,-.0001]);y=A@v;N=len(n);xs=1/(4*.75)
    rr=[dict(n=j,M=xs+y[i],S=y[N+i],x_Ceven=xs+y[i]+y[N+i],x_Codd=xs+y[i]-y[N+i]) for i,j in enumerate(n)]
    f=solve(rr,c)
    assert max(abs(f[s]-x) for s,x in zip(names,v))<1e-10
    return dict(alpha_at_K1=a,expected=target,dual_K1p3=coefficients(1.3),synthetic_recovery=True)

def analyze():
    OUT.mkdir(parents=True,exist_ok=True);guard();test=checks();groups=inputs()
    fits=[];summary=[];coefs={};ablations=[]
    for (source,run),rr in groups.items():
        guard();K=rr[0]['K'];c=coefs.setdefault(K,coefficients(K))
        for protocol,start in [('pilot8',8),('strict15',15)]:
            part=weak(rr,start);base=dict(source=source,run_id=run,K=K,chi=rr[0]['chi'],protocol=protocol)
            if len(part)<8:
                summary.append(dict(**base,status='insufficient_window',weak_points=len(part)));continue
            end=part[-1]['n'];seen=set();local=[]
            for length in (8,10,12):
                for shift in (0,1,2):
                    pp=[r for r in part if end-shift-length+1<=r['n']<=end-shift]
                    if len(pp)<8:continue
                    for floor in (True,False):
                        key=(pp[0]['n'],pp[-1]['n'],floor)
                        if key in seen:continue
                        seen.add(key);f=dict(**base,**solve(pp,c,floor));local.append(f);fits.append(f)
            main=next(f for f in local if f['n_max']==end and f['npoints']==min(10,len(part)) and f['S_floor'])
            s=dict(**main,status='fit',weak_points=len(part),variants=len(local))
            for k in ('gd2','g1sq','g2','branch_rmse_rel'):
                if k in main:
                    rate={'gd2':c['r_d'],'g1sq':c['r_1'],'g2':c['r_2'],'branch_rmse_rel':1.}[k]
                    values=[f[k]*rate**(end-f['n_max']) for f in local]
                    s[k+'_min']=min(values);s[k+'_max']=max(values)
            mainpart=[r for r in part if main['n_min']<=r['n']<=end]
            for drop in ('gd2','g1sq','g2'):
                if drop=='gd2' and not c['d_included']:continue
                ablations.append(dict(**base,omitted=drop,**solve(mainpart,c,True,drop)))
            summary.append(s)
    write('coefficients.csv',list(coefs.values()));write('all_fits.csv',fits);write('summary.csv',summary);write('ablations.csv',ablations)
    predictions=[];max_residual_difference=0.
    for f in fits:
        part=[r for r in groups[f['source'],f['run_id']] if f['n_min']<=r['n']<=f['n_max']]
        names,A=design([r['n'] for r in part],coefs[f['K']],f['S_floor'])
        y=A@np.array([f[k] for k in names]);N=len(part);xs=1/(4*f['K']);errors=[]
        for i,r in enumerate(part):
            E=xs+y[i]+y[N+i];O=xs+y[i]-y[N+i]
            errors.extend([E-r['x_Ceven'],O-r['x_Codd']])
            predictions.append(dict(source=f['source'],run_id=f['run_id'],protocol=f['protocol'],
                n_min=f['n_min'],n_max=f['n_max'],S_floor=f['S_floor'],n=r['n'],
                M_fit=xs+y[i],S_fit=y[N+i],x_E_fit=E,x_O_fit=O,x_E_data=r['x_Ceven'],x_O_data=r['x_Codd']))
        difference=abs(float(np.sqrt(np.mean(np.array(errors)**2))/xs)-f['branch_rmse_rel'])
        max_residual_difference=max(max_residual_difference,difference)
        assert f['success'] and difference<1e-12
    write('predictions.csv',predictions)
    test['all_saved_fit_reconstructions']=len(fits)
    test['max_residual_reconstruction_difference']=max_residual_difference
    pdf=ROOT.parents[1]/'Otsuki/sixvertex/perturbation_results/sixvertex_perturbation_results.pdf'
    meta=dict(date='2026-10-29 JST',input=str(INPUT.relative_to(ROOT)),input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=test,memory=guard(),
        coefficient_document_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        notes=['No new RG or USB access.','Rates fixed by scaling dimensions, not measured.',
               'Squared couplings nonnegative; g2 signed.','d omitted at K>=2.',
               'Finite-window effective couplings; truncation source and induced running not independently modeled.'])
    (OUT/'checks.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('FITS',len(fits),'SUMMARIES',len(summary),'MEMORY',meta['memory'])

def plot(kind_filter=None,page_filter=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True)
    sums=[s for s in read(OUT/'summary.csv') if s['source']=='Yamamoto' and s['protocol']=='pilot8']
    groups=inputs();sums.sort(key=lambda r:float(r['K']))
    for kind,page in [(k,p) for k in ('M','S','branches') for p in (1,2)]:
        if kind_filter and kind!=kind_filter:continue
        if page_filter and page!=page_filter:continue
        guard();batch=sums[(page-1)*9:page*9]
        fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
        for ax,s in zip(axes.flat,batch):
            rr=groups[s['source'],s['run_id']];K=float(s['K']);xs=1/(4*K);c=coefficients(K)
            pp=[r for r in rr if 6<=r['n']<=32];n=np.array([r['n'] for r in pp])
            if kind=='branches':
                for field,color in [('x_Ceven','tab:orange'),('x_Codd','tab:blue')]:
                    ax.plot(n,[(r[field]-xs)/xs for r in pp],'.-',ms=3,lw=.7,color=color,label=field)
            else:ax.plot(n,[(r[kind]-(xs if kind=='M' else 0))/xs for r in pp],'k.-',ms=3,lw=.7,label='data')
            if s['status']=='fit':
                nn=np.arange(int(s['n_min']),int(s['n_max'])+1);names,A=design(nn,c)
                v=np.array([float(s[k]) for k in names]);N=len(nn);y=A@v
                ax.axvspan(nn[0],nn[-1],alpha=.15,color='gold')
                if kind=='branches':
                    ax.plot(nn,(y[:N]+y[N:])/xs,'r-',lw=1.6,label='fit even')
                    ax.plot(nn,(y[:N]-y[N:])/xs,'b--',lw=1.6,label='fit odd')
                else:
                    sl=slice(0,N) if kind=='M' else slice(N,2*N)
                    ax.plot(nn,y[sl]/xs,'r-',lw=1.6,label='joint fit')
                    for j,name in enumerate(names):
                        z=A[:,j]*v[j]
                        if np.any(z[sl]) and name not in ('M0','Mdrift','S0'):
                            label={'gd2':'d (2nd)','g1sq':'W1 (2nd)','g2':'W2 (1st)'}[name]
                            ax.plot(nn,z[sl]/xs,'--',lw=.8,label=label)
                    bg=sum((A[:,j]*v[j] for j,name in enumerate(names) if name in ('M0','Mdrift','S0')),np.zeros(2*N))
                    ax.plot(nn,bg[sl]/xs,':',color='gray',label='background')
                note=f"{s['n_min']}-{s['n_max']}; err={100*float(s['branch_rmse_rel']):.2g}%"
            else:note='fewer than 8 valid points'
            ax.set_title(f'K={K:g}; {note}',fontsize=8)
            ax.set_yscale('symlog',linthresh=1e-4);ax.set_ylim(-.3,.3);ax.grid(alpha=.2)
            ax.legend(fontsize=5,loc='best');ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(batch):]:ax.set_visible(False)
        fig.suptitle(f'C-only approximate CPT: {kind}; chi=16, pilot8; shaded fit window\nNormalized shifts; symlog axis; omitted operators not fitted',fontsize=12)
        fig.savefig(FIG/f'{kind}_page{page}.png',dpi=100)
        plt.close(fig)
        del fig,axes;gc.collect()
        print('PLOT',kind,page,memory(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plot',action='store_true')
    p.add_argument('--kind',choices=('M','S','branches'));p.add_argument('--page',type=int,choices=(1,2));args=p.parse_args()
    plot(args.kind,args.page) if args.plot else analyze()
