"""Finite, resumable campaign; remote reservations and frozen predictive tests.

Run with Python 3.11 or later. No existing tensor or reservation is replaced.
The isolated coordination checkout is the only tree this controller commits.
"""
import argparse, concurrent.futures, hashlib, importlib.util, json, math, os
import shutil, subprocess, sys, threading, time, tomllib, uuid
from datetime import datetime, timezone
from pathlib import Path
import toml

H=Path(__file__).resolve().parent
ROOT=H.parents[5]
COORD=Path(os.environ.get('TNR_COORDINATION_ROOT', str(ROOT))).resolve()
PY=sys.executable
JULIA=os.environ.get('JULIA','julia')
TAG='twist_campaign_20261109'
LOCK=threading.Lock()

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def dump(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def manager():
    spec=importlib.util.spec_from_file_location('twist16_manager',COORD/'local_tnr_data/manage.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def git(*args):
    return subprocess.check_output(['git','-C',str(COORD),*args],text=True,stderr=subprocess.STDOUT).strip()
def update_coord():
    if git('status','--porcelain','--untracked-files=no'):raise RuntimeError('Coordination checkout has pending edits')
    git('fetch','origin','main');git('merge','--ff-only','origin/main')
def sources():
    paths=[H/'worker.jl', ROOT/'Yamazaki/runs/followup_B_20261106/common.jl', H.parent/'isotropic_twist_20261109/twist_measure.jl', H/'environment/Project.toml', H/'environment/Manifest.toml']
    return {str(p):sha(p) for p in paths}
def profile():
    p=H/'control/profile.json'
    if p.exists():return json.loads(p.read_text())
    out=subprocess.check_output([JULIA,'+1.13.3','--startup-file=no','--threads=1','--project='+str(H/'environment'),str(H/'worker.jl'),'--profile'],text=True)
    (H/'control/profile.log').write_text(out)
    par=next(l for l in out.splitlines() if l.startswith('LoopParameters{'))
    pkg=next(l.split('=',1)[1] for l in out.splitlines() if l.startswith('PACKAGE_ROOT='))
    m=manager();r=dict(algorithm='LoopTNR',scalar_type='Float64',one_loop_init=True,loop_parameters_repr=par,**m.environment_record(H/'environment',Path(pkg)))
    dump(p,r);return r

def make_case(K,chi,h,label,group,steps,kind='calibration'):
    m=manager();ident=group+'_'+label;variant=TAG+'_'+ident;cv=round(2*math.sin(math.pi/(4*K)),12)
    c=dict(id=ident,group=group,K=K,chi=chi,h=h,a=1+h,b=1-h,c=cv,steps=steps,
           label=label,kind=kind,variant=variant,stop_at_threshold=True,params_repr=profile()['loop_parameters_repr'])
    c['checkpoint_path']=f'local_tnr_data/Six_vertex/symmetry=CU1Irrep/a={m.number(1+h)}_b={m.number(1-h)}/c={m.number(cv)}/LoopTNR/chi={chi}/RGsteps={steps}_init=true/variant={variant}'
    return c

def reuse_isotropic(c):
    if c['K']!=1.5 or c['chi']!=24 or c['h'] not in [0.,.001,-.001]:return c
    old=H.parent/'isotropic_twist_20261109';name={0.:'zero',.001:'plus',-.001:'minus'}[c['h']]
    ms={};src={};last=None
    for p in [old/'data'/f'{name}.toml',old/'data'/f'{name}_from12.toml']:
        d=tomllib.loads(p.read_text());last=d;src[str(p)]=sha(p)
        assert d['config']['K']==c['K'] and d['config']['chi']==c['chi'] and d['config']['h']==c['h']
        assert d['config']['params_repr']==c['params_repr']
        for ck in d['checkpoints']:
            assert sha(ROOT/ck['path'])==ck['sha256'];ms[ck['step']]=ck
    for f in ['Project.toml','Manifest.toml']:assert sha(old/'environment'/f)==sha(H/'environment'/f)
    c.update(reuse=True,reuse_sources=src,reuse_checkpoints=[ms[n] for n in sorted(ms) if n<=c['steps']],
             reuse_reason='Identical archived paper model, default parameters and environment; verified original checkpoints, no repeated RG.')
    if 'n_e' in last:c['reused_n_e']=last['n_e']
    return c

def reserve(cases,batch):
    update_coord();m=manager();shared=m.DEFAULT_ROOT;m.require_shared_current(shared)
    _,catalog=m.read_catalog(shared);prof=profile();ownedpath=shared/'.local/owned_requests.json'
    owned=m.json_read(ownedpath) if ownedpath.exists() else {};pending=[];plans=[]
    existing=[m.json_read(p) for p in (shared/'_requests').glob('*.json')]
    for c in cases:
        q=dict(symmetry='CU1Irrep',a=m.number(c['a']),b=m.number(c['b']),c=m.number(c['c']),chi=c['chi'])
        state,candidates=m.assess(catalog,q,c['steps'],prof)
        related=[dict(key=r['key'],owner=r['owner'],status=r['status'],variant=r.get('variant','')) for r in existing if r['query']==q]
        plans.append(dict(id=c['id'],state=state,candidates=candidates,related_reservations=related,reuse=c.get('reuse',False)))
        if c.get('reuse'):continue
        # Matching archived tensor profiles are reviewed before a repeated run.
        # A variant records the alternating B tensor and the dedicated source/measurement protocol.
        key=m.request_key(q,c['variant']);p=shared/'_requests'/f'{key}.json'
        if p.exists():raise RuntimeError('Reservation already exists: '+str(p))
        reason=('User authorized comprehensive docs16 signed-twist/calibration/prediction calculations. '
                'Isotropic O(2), A=(1+mu,1-mu,c), B=(1-mu,1+mu,c), fixed paper environment; '
                'saved matching isotropic K1.5 chi24 tensors reused separately. Other existing catalogs/runs are '
                'reviewed in campaign plans; spectra-only paper runs have no local checkpoints. '
                'This variant records both sublattices and the complete signed calibration protocol. '
                'No model/seed inference from unsigned spectra alone. '+c['kind'])
        rec=dict(schema_version=1,key=key,token=str(uuid.uuid4()),owner='Fushio',status='reserved',created_at_utc=now(),
                 query=q,target_steps=c['steps'],profile=prof,variant=c['variant'],reason=reason)
        pending.append((p,rec));owned[key]=rec['token'];c['reservation_key']=key
    dump(H/'plans'/f'{batch}.json',plans)
    for p,r in pending:m.json_write(p,r,exclusive=True)
    m.json_write(ownedpath,owned)
    if pending:
        names=[str(p.relative_to(COORD)) for p,r in pending]
        git('add','--',*names);git('commit','-m',f'Reserve Fushio docs16 signed-twist {batch}')
        try:git('push','origin','HEAD:main')
        except subprocess.CalledProcessError:
            # Keep local claims, integrate concurrent additions without force.
            git('fetch','origin','main');git('rebase','origin/main');git('push','origin','HEAD:main')
        dump(H/'control'/f'{batch}_reservation_commit.json',dict(commit=git('rev-parse','HEAD'),paths=names,time=now()))
    for c in cases:
        cfg=dict(coordination_root=str(COORD),source_sha256=sources(),runs=[c])
        (H/'control'/f'{c["id"]}.toml').write_text(toml.dumps(cfg))
    return cases

def prepare():
    if (H/'campaign.json').exists():raise RuntimeError('Campaign already prepared')
    # Prioritize the directly comparable isotope and sign-reversal controls.
    pairs=[(1.5,24),(1.5,32),(1.5,40),(1.,40),(1.25,40)]
    pairs += [(K,ch) for K in [.75,1.,1.25,1.5,1.75] for ch in [24,32,40] if (K,ch) not in pairs]
    pairs += [(K,ch) for K in [1.,1.5] for ch in [48,64]]
    pairs += [(2.5,ch) for ch in [32,40]]
    settings={.75:(12,1e-5),1.:(16,1e-4),1.25:(20,5e-4),1.5:(24,1e-3),1.75:(32,2e-3),2.5:(24,2e-3)}
    groups=[];cases=[]
    for K,ch in pairs:
        ncal,eps=settings[K];group=f'K{K:g}_chi{ch}'
        g=dict(id=group,K=K,chi=ch,calibration_step=ncal,epsilon=eps,predict=K<2,
               calibration_steps=list(range(4,ncal+1,4)),prediction_steps=160 if K>=1.5 else 120)
        ids=[]
        for label,h in [('zero',0.)]+[(f'{sign}{j}',sgn*eps/(2**j)) for j in range(3) for sign,sgn in [('plus',1),('minus',-1)]]:
            c=make_case(K,ch,h,label,group,80 if label=='zero' else ncal)
            c=reuse_isotropic(c);cases.append(c);ids.append(c['id'])
        g['calibration_ids']=ids;groups.append(g)
    protocol=dict(created_at=now(),purpose='Signed first-order spectroscopy and frozen predictive compensation for main_en.pdf',
        groups=groups,calibration_gate=dict(slope_half_quarter_relative_tolerance=.05,even_fraction_tolerance=.05,
        maximum_abs_X=.1,maximum_abs_delta_pi=.2,leading_relative_residual=1e-8,phase_tolerance=1e-5),
        prediction_rule='At preselected ncal use central slope from +/-epsilon/4. Require epsilon/2 consistency, even-response and all numerical/weak gates. Freeze mu=-delta0/s; no retuning. Failed gate holds the prediction.',
        sensitivity_groups=['K1.5_chi24','K1.5_chi32','K1.5_chi40'],sensitivity_factors=[.9,1.1],
        primary_cutoff='First every-step abs(1-4K*x01)>0.1 after n>=6; n_e=n-1. 160/120 step maximum is right-censored.',
        independent_validation='Parity/phase-labelled compact-boson levels excluding x01 and twist pair; RMS<0.01 and abs(c-1)<0.005, with assignment coverage reported.',
        maximum_workers=2,minimum_free_disk_GiB=40,maximum_worker_hours=18,
        caveat='Finite-chi spectral window, not mass convergence; no SPT classification claimed.')
    dump(H/'protocol.json',protocol) # written before any production trajectory
    cases=reserve(cases,'calibration')
    dump(H/'campaign.json',dict(protocol_sha256=sha(H/'protocol.json'),source_sha256=sources(),runs=cases))
    print(dict(groups=len(groups),cases=len(cases),reused=sum(c.get('reuse',False) for c in cases)),flush=True)

def make_prediction(g):
    result=H/'predictions'/f'{g["id"]}.json'
    if result.exists():return json.loads(result.read_text())
    datasets={};src={}
    for ident in g['calibration_ids']:
        p=H/'data'/f'{ident}.toml'
        if not p.exists():return None
        d=tomllib.loads(p.read_text())
        if d['status']=='running':return None
        datasets[d['config']['label']]=d;src[str(p)]=sha(p)
    n=g['calibration_step'];eps=g['epsilon'];checks=[];t={}
    for label,d in datasets.items():
        ts={r['step']:r for r in d['twist_measurements']};ms={r['step']:r for r in d['measurements']}
        if n not in ts or n not in ms or not ts[n]['numerical_gate'] or not ms[n]['numerical_gate']:
            checks.append(label+': calibration missing or numerical gate failed');continue
        t[label]=ts[n]['splitting_x']
        if abs(ms[n]['paper_X'])>.1 or abs(t[label])>.2:checks.append(label+': weak-coupling gate failed')
    out=dict(group=g['id'],created_at=now(),calibration_step=n,source_sha256=src,protocol_sha256=sha(H/'protocol.json'),
             status='held',reasons=checks)
    if len(t)==7:
        slopes=[(t[f'plus{j}']-t[f'minus{j}'])/(2*eps/2**j) for j in range(3)]
        evens=[(t[f'plus{j}']+t[f'minus{j}'])/2-t['zero'] for j in range(3)]
        mismatch=abs(slopes[2]-slopes[1])/max(abs(slopes[2]),1e-30)
        even_fraction=abs(evens[2])/max(abs(slopes[2]*eps/4),1e-30)
        out.update(slopes=slopes,even_response=evens,slope_relative_difference=mismatch,even_fraction=even_fraction)
        if mismatch>.05:checks.append('epsilon/2 and epsilon/4 slopes differ by >5%')
        if even_fraction>.05:checks.append('even response exceeds 5% of odd response')
        if not math.isfinite(slopes[2]) or abs(slopes[2])<1e-12:checks.append('slope unresolved')
        if not checks:
            mu=-t['zero']/slopes[2]
            if abs(mu)>2*eps:checks.append('predicted input outside calibrated range')
            else:out.update(status='frozen',input_h=mu,delta_zero=t['zero'],slope=slopes[2])
    dump(result,out);return out

def run_case(c):
    ident=c['id'];data=H/'data'/f'{ident}.toml';log=H/'control'/f'{ident}.log'
    if data.exists():
        d=tomllib.loads(data.read_text())
        return dict(id=ident,status=d['status'],saved=d['last_saved_step'],existing=True)
    if log.exists():return dict(id=ident,status='launch_previously_attempted_manual_review')
    if shutil.disk_usage(ROOT).free<40*2**30:raise RuntimeError('Free disk below 40 GiB; no new job launched')
    cfg=tomllib.loads((H/'control'/f'{ident}.toml').read_text())
    for p,h in cfg['source_sha256'].items():
        if sha(p)!=h:raise RuntimeError('Frozen source changed: '+p)
    env=os.environ.copy();env.update(PYTHON=PY,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    for k in ['GIT_DIR','GIT_WORK_TREE']:env.pop(k,None)
    cmd=[JULIA,'+1.13.3','--startup-file=no','--threads=1','--project='+str(H/'environment'),str(H/'worker.jl'),str(H/'control'/f'{ident}.toml')]
    with log.open('x') as stream:
        with LOCK:
            if not c.get('reuse'):update_coord()
            p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            dump(H/'control'/f'{ident}.process.json',dict(pid=p.pid,started_at=now(),command=cmd))
            begin=time.monotonic()
            while p.poll() is None and not (H/'control'/f'guard_{ident}.toml').exists():
                if time.monotonic()-begin>600:p.terminate();raise RuntimeError('Guard startup timeout: '+ident)
                time.sleep(1)
        try:code=p.wait(timeout=18*3600)
        except subprocess.TimeoutExpired:
            p.terminate();p.wait(timeout=60);return dict(id=ident,status='walltime_limit_manual_resume',exit_code=p.returncode)
    r=dict(id=ident,exit_code=code,finished_at=now())
    if data.exists():
        d=tomllib.loads(data.read_text());r.update(status=d['status'],saved=d['last_saved_step'],n_e=d.get('n_e'))
    else:r['status']='startup_failed'
    dump(H/'control'/f'{ident}.completion.json',r);print(r,flush=True);return r

def refresh():
    subprocess.run([PY,str(H/'analyze.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)

def launch():
    import fcntl
    with (H/'control/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=json.loads((H/'campaign.json').read_text());protocol=json.loads((H/'protocol.json').read_text())
        assert cfg['protocol_sha256']==sha(H/'protocol.json')
        dump(H/'control/controller.json',dict(pid=os.getpid(),status='running',started_at=now()))
        cases={c['id']:c for c in cfg['runs']};allresults=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for g in protocol['groups']:
                if (H/'control/STOP_AFTER_CURRENT').exists():break
                batch=[cases[i] for i in g['calibration_ids']]
                futures=[pool.submit(run_case,c) for c in batch]
                for f in concurrent.futures.as_completed(futures):
                    allresults.append(f.result());dump(H/'control/completions.json',allresults);refresh()
                if g['predict']:
                    pred=make_prediction(g)
                    if pred and pred['status']=='frozen':
                        pc=H/'predictions'/f'{g["id"]}_campaign.json'
                        if pc.exists():new=json.loads(pc.read_text())['runs']
                        else:
                            factors=[1.]+(protocol['sensitivity_factors'] if g['id'] in protocol['sensitivity_groups'] else [])
                            new=[make_case(g['K'],g['chi'],pred['input_h']*f,'predicted' if f==1 else f'sensitivity{f:g}',g['id'],g['prediction_steps'],kind='prediction' if f==1 else 'sensitivity') for f in factors]
                            for c in new:c['prediction_sha256']=sha(H/'predictions'/f'{g["id"]}.json')
                            with LOCK:new=reserve(new,'prediction_'+g['id'])
                            dump(pc,dict(runs=new,prediction_sha256=sha(H/'predictions'/f'{g["id"]}.json')))
                        for f in concurrent.futures.as_completed([pool.submit(run_case,c) for c in new]):
                            allresults.append(f.result());dump(H/'control/completions.json',allresults);refresh()
                refresh()
        dump(H/'control/controller.json',dict(pid=os.getpid(),status='queue_finished' if not (H/'control/STOP_AFTER_CURRENT').exists() else 'stopped_after_group',finished_at=now(),physical_convergence=False))
        refresh()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','launch','status']);a=ap.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='launch':
        try:launch()
        except Exception as e:
            dump(H/'control/controller_error.json',dict(time=now(),error=repr(e)));raise
    else:refresh();print((H/'status.json').read_text())
