"""Read atomic saved outputs, retain rejected points, update partial figures/status.

This script never launches RG or modifies predictions. The neutral-multiset RMS
is diagnostic until individual level identities have been reviewed.
"""
import csv, json, math, os, re, tomllib, tempfile
from pathlib import Path
from datetime import datetime,timezone
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'twist16-mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
ROOT=H.parents[5]
def finite_json(x):
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:finite_json(v) for k,v in x.items()}
    if isinstance(x,list):return [finite_json(v) for v in x]
    return x
def dump(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(finite_json(d),indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def csvwrite(p,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def savefig(fig,name):
    for ext in ['png','pdf']:fig.savefig(H/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
def alive(pid):
    try:os.kill(int(pid),0);return True
    except (ProcessLookupError,ValueError,TypeError):return False
def theory(K,charge,parity,nmax=5):
    # Oscillator partitions, with parity (-1)^(number of current modes).
    parts=[(0,0)]
    for mode in range(1,nmax+1):
        parts=[(n+mode*k,p+k) for n,p in parts for k in range((nmax-n)//mode+1)]
    ans=[]
    for e in (range(-3,4) if charge else range(4)):
        for n,p in parts:
            for nb,pb in parts:
                x=K*e*e+charge*charge/(4*K)+n+nb
                if x>8:continue
                if charge or e>0 or (1 if (p+pb)%2==0 else -1)==parity:ans.append(x)
    return sorted(ans)
def secondary(m,K):
    errors=[];accepted=[];charged=[];charged_raw=[]
    def sector_key(s):
        match=re.search(r'\((\d+),\s*(\d+)\)$',s)
        return tuple(map(int,match.groups())) if match else None
    for a,b in [(0,0),(0,1),(1,2),(2,2),(3,2)]:
        vals=sorted([r for r in m['spectra'] if sector_key(r['sector'])==(a,b)],key=lambda r:r['rank'])
        ref=theory(K,a,1 if b==0 else -1)
        for i,r in enumerate(vals):
            if i>=len(ref):continue
            if (a==0 and b==0 and i==0) or (a==1 and i==0):continue
            if ref[i]>2.5:continue
            err=r['x']-ref[i];errors.append(err)
            numerical=r['residual']<1e-8 and abs(r['phase'])<1e-5 and math.isfinite(r['x'])
            if numerical:accepted.append(err)
            if a in [2,3] and i==0:
                charged_raw.append(err)
                if numerical:charged.append(err)
    rms=lambda x:float(np.sqrt(np.mean(np.square(x)))) if x else None
    return dict(diagnostic_multiset_rms=rms(accepted),diagnostic_expected_count=len(errors),diagnostic_accepted_count=len(accepted),
                charged_heldout_rms=rms(charged) if len(charged)==2 else None,charged_heldout_count=len(charged),
                charged_raw_rms=rms(charged_raw),charged_expected_count=len(charged_raw),
                diagnostic_gate=m['numerical_gate'] and len(accepted)>=6 and len(accepted)==len(errors) and rms(accepted)<.01 and abs(m['central_charge']-1)<.005)
def main():
    protocol=json.loads((H/'protocol.json').read_text());campaign=json.loads((H/'campaign.json').read_text())
    datasets={};summary=[];rows=[];twists=[]
    for p in sorted((H/'data').glob('*.toml')):
        d=tomllib.loads(p.read_text());c=d['config'];datasets[c['id']]=d
        state=d['status'];process=H/'control'/f"{c['id']}.process.json"
        if state=='running' and process.exists() and not alive(json.loads(process.read_text())['pid']):
            state='interrupted_requires_review'
        summary.append(dict(id=c['id'],group=c['group'],kind=c['kind'],status=state,saved=d['last_saved_step'],
                            measured=d['last_measured_step'],n_e=d.get('n_e'),reused=c.get('reuse',False),source=str(p.relative_to(ROOT))))
        for m in d['measurements']:
            r=dict(id=c['id'],group=c['group'],K=c['K'],chi=c['chi'],h=c['h'],step=m['step'],X=m['paper_X'],
                   c=m['central_charge'],numerical_gate=m['numerical_gate'],checkpoint_sha256=m['checkpoint_sha256'])
            r.update(secondary(m,c['K']));rows.append(r)
        for t in d['twist_measurements']:
            twists.append(dict(id=c['id'],group=c['group'],h=c['h'],step=t['step'],x_even=t['x_even'],x_odd=t['x_odd'],
                splitting=t['splitting_x'],g=t['g_signed'],numerical_gate=t['numerical_gate'],phase=t['max_phase'],
                residual=t['leading_relative_residual'],checkpoint_sha256=t['checkpoint_sha256']))
    cal=[];fits=[]
    for g in protocol['groups']:
        ds={datasets[i]['config']['label']:datasets[i] for i in g['calibration_ids'] if i in datasets}
        maps={label:{t['step']:t for t in d['twist_measurements']} for label,d in ds.items()}
        measurements={label:{m['step']:m for m in d['measurements']} for label,d in ds.items()}
        if 'zero' not in maps:continue
        for j in range(3):
            a,b=f'plus{j}',f'minus{j}'
            if a not in maps or b not in maps:continue
            eps=g['epsilon']/2**j
            for n in sorted(set(maps['zero'])&set(maps[a])&set(maps[b])):
                z,pp,mm=[maps[label][n] for label in ['zero',a,b]]
                slope=(pp['splitting_x']-mm['splitting_x'])/(2*eps)
                even=(pp['splitting_x']+mm['splitting_x'])/2-z['splitting_x']
                gate=all(t['numerical_gate'] for t in [z,pp,mm]) and all(measurements[label][n]['numerical_gate'] and abs(measurements[label][n]['paper_X'])<=.1 for label in ['zero',a,b]) and max(abs(t['splitting_x']) for t in [z,pp,mm])<=.2
                cal.append(dict(group=g['id'],K=g['K'],chi=g['chi'],step=n,epsilon=eps,slope=slope,even_response=even,
                    mu_pred=-z['splitting_x']/slope if abs(slope)>1e-30 else None,
                    gamma_theory=2**((2-g['K'])/2),rescaled_slope=slope/2**((2-g['K'])*n/2),gate=gate))
            selected=[r for r in cal if r['group']==g['id'] and r['epsilon']==eps and r['step']>=8 and r['gate'] and abs(r['slope'])>1e-30]
            if len(selected)>=3:
                x=np.array([r['step'] for r in selected]);y=np.log(np.abs([r['slope'] for r in selected]));v=np.polyfit(x,y,1)
                fits.append(dict(group=g['id'],epsilon=eps,n_min=int(x.min()),n_max=int(x.max()),points=len(x),gamma=float(np.exp(v[0])),
                    gamma_theory=2**((2-g['K'])/2),log_rms=float(np.sqrt(np.mean((y-np.polyval(v,x))**2))),
                    interpretation='exploratory window; late endpoint and epsilon sensitivity retained'))
    predictions=[]
    for p in sorted((H/'predictions').glob('*.json')):
        if not p.stem.endswith('_campaign'):predictions.append(json.loads(p.read_text()))
    csvwrite(H/'observables.csv',rows);csvwrite(H/'twist.csv',twists);csvwrite(H/'calibration.csv',cal)
    dump(H/'analysis.json',dict(series=summary,calibration=cal,growth_fits=fits,predictions=predictions,
        caveat='Partial data. Neutral RMS is a sorted theoretical-multiset diagnostic, pending identity review. All rejected points retained.',physical_convergence=False))
    completed=sum(s['status']!='running' for s in summary)
    controller_path=H/'control/controller.json'
    controller=json.loads(controller_path.read_text()) if controller_path.exists() else {}
    controller['alive']=alive(controller.get('pid'))
    error_path=H/'control/controller_error.json'
    launch_failures=[json.loads(p.read_text()) for p in (H/'control').glob('*.completion.json')
                    if json.loads(p.read_text()).get('status')=='startup_failed']
    status=dict(updated_at_utc=datetime.now(timezone.utc).isoformat(),planned_groups=len(protocol['groups']),
        calibration_jobs=len(campaign['runs']),conditional_prediction_jobs=sum(g['predict'] for g in protocol['groups']),
        conditional_sensitivity_jobs=len(protocol['sensitivity_groups'])*len(protocol['sensitivity_factors']),
        started_jobs=len(summary)+len(launch_failures),terminal_jobs=completed+len(launch_failures),
        failed_jobs=sum(s['status'] in ['failed','interrupted_requires_review'] for s in summary)+len(launch_failures),
        controller=controller,controller_error=json.loads(error_path.read_text()) if error_path.exists() else None,
        launch_failures=launch_failures,
        measurements=len(rows),twist_measurements=len(twists),new_checkpoints=sum(len(d['checkpoints']) for d in datasets.values() if not d['config'].get('reuse')),
        running=[s for s in summary if s['status']=='running'],prediction_status={p['group']:p['status'] for p in predictions},
        scope='Launch/saved progress only; terminal status is not physical convergence. Drive archival pending for this active campaign.')
    dump(H/'status.json',status)
    text=['# docs16 計算の保存状況','',f"更新（UTC）: {status['updated_at_utc']}",'',
          f"較正は{status['calibration_jobs']}系列、条件付き補償は最大{status['conditional_prediction_jobs']}系列、近傍の対照は最大{status['conditional_sensitivity_jobs']}系列です。",
          f"現在 {len(summary)}系列に出力があり、終了状態は{completed}系列です。通常測定{len(rows)}点、twist測定{len(twists)}点を保存しています。物理的収束は別に評価します。",'',
          f"controller: {controller.get('status','未起動')} / PID生存確認: {controller['alive']} / 要確認の失敗・中断: {status['failed_jobs']}",'',
          '| 系列 | 状態 | 保存RG | 測定RG | n_e |','|---|---|---:|---:|---:|']
    text += [f"| {s['id']} | {s['status']} | {s['saved']} | {s['measured']} | {s['n_e'] if s['n_e'] is not None else '未判定'} |" for s in summary]
    text += ['','較正条件を通過しない組は補償を保留します。失敗点・未完了点は削除しません。',
             'Driveへの転送とリモート照合は、このcampaignの計算・解析がまとまった段階で行います。']
    (H/'progress.md').write_text('\n'.join(text)+'\n')
    if not twists:return
    # Plot available cases, explicitly retaining partial coverage in the title.
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    g0=next((g for g in protocol['groups'] if any(r['group']==g['id'] for r in cal)),protocol['groups'][0]);gid=g0['id']
    for label,color in [('zero','black'),('plus0','C0'),('minus0','C1'),('plus2','C2'),('minus2','C3')]:
        rr=[t for t in twists if t['id']==gid+'_'+label]
        if not rr:continue
        for name,style in [('x_even','-'),('x_odd','--')]:axs[0,0].plot([r['step'] for r in rr],[r[name] for r in rr],style+'o',ms=2,color=color,label=label+' '+name[2:])
    axs[0,0].axhline(g0['K']/4,color='gray',ls=':',label='K/4')
    axs[0,0].set(title=gid,ylabel='Twisted scaling dimension',xlabel='RG step')
    for eps in sorted({r['epsilon'] for r in cal if r['group']==gid},reverse=True):
        rr=[r for r in cal if r['group']==gid and r['epsilon']==eps]
        for ax,key in [(axs[0,1],'rescaled_slope'),(axs[1,0],'mu_pred')]:
            ax.plot([r['step'] for r in rr],[r[key] for r in rr],'-o',ms=3,label=f'epsilon={eps:g}')
            bad=[r for r in rr if not r['gate']];ax.plot([r['step'] for r in bad],[r[key] for r in bad],'rx',ms=6)
    axs[0,1].set(xlabel='RG step',ylabel='Slope / theoretical amplification')
    axs[1,0].set(xlabel='RG step',ylabel='Predicted compensation input')
    for g in protocol['groups']:
        ff=[f for f in fits if f['group']==g['id']]
        if ff:axs[1,1].plot([g['K']]*len(ff),[f['gamma']/f['gamma_theory'] for f in ff],'o',ms=3,label=g['id'])
    axs[1,1].axhline(1,color='black',ls=':');axs[1,1].set(xlabel='K',ylabel='Measured / theoretical growth factor')
    for ax in axs.flat:
        ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[1]:ax.legend(fontsize=6)
    fig.suptitle('Signed-twist campaign: saved data only (partial)');savefig(fig,'signed_calibration_partial')
    fig,axs=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for s in summary:
        if s['group']!=gid or not (s['id'].endswith('_zero') or s['kind']=='prediction'):continue
        rr=[r for r in rows if r['id']==s['id'] and r['step']>=6]
        axs[0].semilogy([r['step'] for r in rr],[max(abs(r['X']),1e-16) for r in rr],label=s['id'])
        tt=[t for t in twists if t['id']==s['id']];axs[1].plot([r['step'] for r in tt],[r['splitting'] for r in tt],'-o',ms=2,label=s['id'])
        rr=[r for r in rr if r['charged_heldout_rms'] is not None]
        if rr:axs[2].semilogy([r['step'] for r in rr],[max(r['charged_heldout_rms'],1e-16) for r in rr],label=s['id'])
    axs[0].axhline(.1,color='black',ls=':');axs[0].set_ylabel('|1-4K x01|')
    axs[1].axhline(0,color='gray',ls=':');axs[1].set_ylabel('Signed pi-flux splitting')
    axs[2].axhline(.01,color='black',ls=':');axs[2].set_ylabel('Charge 2,3 held-out RMS')
    for ax in axs:
        ax.set_xlabel('RG step');ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[1]:ax.legend(fontsize=6)
    fig.suptitle(gid+': frozen prediction validation (partial)');savefig(fig,'compensation_partial')
if __name__=='__main__':
    import fcntl
    # Manual refresh and controller completion can coincide.
    with (H/'control/analysis.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
