"""Note13: paired h=.02/.01 amplitude comparison, with shared h=0 subtraction."""
import csv,hashlib,json,os,tomllib
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parents[5]
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/note13_h_compare_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
KEYS=['m1','m2','m3','X'];src=H/'analysis.json';allrows=json.loads(src.read_text())['rows']
d={(v['h'],v['step']):v for v in allrows if v['tube']=='141'}
rows=[];sources={};check_errors=[]
for n in range(0,17,2):
 a,b=d[.01,n],d[.02,n];bg=d[0,n]
 # Recompute each estimator directly from the stored sector levels.
 for r in (a,b,bg):
  p=ROOT/r['source'];blob=p.read_bytes();digest=hashlib.sha256(blob).hexdigest();assert digest==r['sha256']
  raw=tomllib.loads(blob.decode());assert raw['shape']==[1.,4.,1.];assert raw['checkpoint_sha256']==r['checkpoint_sha256']
  assert np.allclose([s['x'] for s in raw['sectors'][1:]],r['x'],rtol=0,atol=1e-15)
  sources[r['source']]=dict(sha256=digest,checkpoint_sha256=r['checkpoint_sha256'])
 for key in KEYS:
  for r in (a,b):
   dx=np.array(r['x'])-bg['x'];aa=r['coefficients']['alpha']
   direct=dx[int(key[-1])-1]/aa[int(key[-1])-1] if key!='X' else np.dot([15,-6,1],dx)/r['coefficients']['AX']
   check_errors.append(abs(direct-r['dG'][key]));assert abs(direct-r['dG'][key])<1e-14
  g1,g2=a['dG'][key],b['dG'][key];assert g1!=0
  ratio=g2/g1;rows.append(dict(step=n,reader=key,G_001=g1,G_002=g2,
    G_001_over_h2=g1/.01**2,G_002_over_h2=g2/.02**2,ratio=ratio,ratio_over_four=ratio/4,
    deviation_percent=100*(ratio/4-1),joint_weak_pass=a['pair_pass'] and b['pair_pass'],
    comparison_pass=n>=6 and a['pair_pass'] and b['pair_pass']))
with (H/'h_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
meta=dict(rows=rows,sources=sources,analysis_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),expected_ratio=4,
 checks=dict(input_files_verified=len(sources),direct_estimator_max_difference=max(check_errors)),
 definition='matched h=0-subtracted estimator; ratio G(h=.02)/G(h=.01); shared background; readers not independent')
(H/'h_comparison.json').write_text(json.dumps(meta,indent=2)+'\n')
plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.2,'figure.dpi':160})
fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
# X collapse shows magnitudes and h^2 scaling; all readers retained in the ratio panel.
xx=[r for r in rows if r['reader']=='X']
for h,col,field in [(.01,'C0','G_001_over_h2'),(.02,'C1','G_002_over_h2')]:
 axs[0].plot([r['step'] for r in xx],[r[field] for r in xx],'-',color=col,label=f'h={h:g}')
 for r in xx:axs[0].plot(r['step'],r[field],'o' if r['step']>=6 else 's',ms=5,color=col,mfc=col if r['comparison_pass'] else 'white')
axs[0].set(title=r'$X$ estimator, divided by $h^2$',ylabel=r'$G_X(n,h)/h^2$',yscale='log');axs[0].legend()
for i,key in enumerate(KEYS):
 rr=[r for r in rows if r['reader']==key]
 axs[1].plot([r['step'] for r in rr],[r['ratio'] for r in rr],'-',color=f'C{i}',label=key)
 for r in rr:axs[1].plot(r['step'],r['ratio'],'o' if r['step']>=6 else 's',ms=4,color=f'C{i}',mfc=f'C{i}' if r['comparison_pass'] else 'white')
axs[1].axhline(4,color='black',ls='--',lw=1,label=r'$h^2$ expectation: 4')
axs[1].set(title='Same-RG amplitude ratio',ylabel=r'$G_r(n,0.02)/G_r(n,0.01)$');axs[1].legend(ncol=2,fontsize=9)
for ax in axs:
 ax.axvspan(5,11,color='green',alpha=.07);ax.set_xlabel('RG step');ax.set_xticks(range(0,17,2))
fig.suptitle(r'Tube [1,4,1], $\chi=32$, $K_0=1.4995965$; matched $h=0$ subtraction')
for ext in ('png','svg'):fig.savefig(H/'figures'/f'h_comparison_141.{ext}')
plt.close(fig)
rr=[r for r in rows if r['comparison_pass']]
print('RG6-10 ratio range',min(r['ratio'] for r in rr),max(r['ratio'] for r in rr))
print('max abs relative departure from 4 (%)',max(abs(r['deviation_percent']) for r in rr))
print('X',[(r['step'],r['ratio']) for r in xx]);print(meta['checks'])
