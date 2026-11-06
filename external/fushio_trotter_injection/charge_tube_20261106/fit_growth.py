"""Fit saved positive delta(g_d^2); report window sensitivity, not sampling errors."""
import csv, hashlib, json, os
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/note13_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H=Path(__file__).resolve().parent
INPUT=H/'analysis.json'
KEYS=['m1','m2','m3','X']
rows=json.loads(INPUT.read_text())['rows']
data={h:sorted([r for r in rows if r['tube']=='141' and r['h']==h],key=lambda r:r['step']) for h in (.01,.02)}
K=data[.01][0]['K']; theory=2**(2-K)
# Rule fixed from the pre-existing diagnostics: discard RG0/2/4, then use
# every remaining contiguous point satisfying the matched weak-window gate.
primary={h:(6,max(r['step'] for r in rr if r['step']>=6 and r['pair_pass'])) for h,rr in data.items()}

def fit(rr,key):
    n=np.array([r['step'] for r in rr],float)
    y=np.array([r['dG'][key] for r in rr],float)
    assert len(n)>=3 and np.all(y>0)
    nc=n.mean();z=np.log(y)
    slope,intercept=np.linalg.lstsq(np.column_stack([n-nc,np.ones(len(n))]),z,rcond=None)[0]
    pred=np.exp(intercept+slope*(n-nc))
    return dict(h=rr[0]['h'],reader=key,start=int(n[0]),end=int(n[-1]),points=len(n),
        n_ref=float(nc),amplitude=float(np.exp(intercept)),log_gamma=float(slope),gamma=float(np.exp(slope)),
        gamma_two_steps=float(np.exp(2*slope)),relative_rms=float(np.sqrt(np.mean((pred/y-1)**2))),
        gamma_vs_theory_percent=float(100*(np.exp(slope)/theory-1)),
        admissible=bool(n[0]>=6 and all(r['pair_pass'] for r in rr)),
        primary=(int(n[0]),int(n[-1]))==primary[rr[0]['h']])

fits=[];local=[]
for h,rr in data.items():
    for i in range(len(rr)-2):
        for j in range(i+3,len(rr)+1):
            fits.extend(fit(rr[i:j],k) for k in KEYS)
    for a,b in zip(rr,rr[1:]):
        for k in KEYS:
            ga=a['dG'][k];gb=b['dG'][k]
            local.append(dict(h=h,reader=k,start=a['step'],end=b['step'],
                gamma=float((gb/ga)**(1/(b['step']-a['step']))) if ga>0 and gb>0 else None,
                admissible=a['step']>=6 and a['pair_pass'] and b['pair_pass']))
main=[f for f in fits if f['primary']]
assert len(main)==8 and all(f['admissible'] for f in main)

def save_csv(name,records):
    with (H/name).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
save_csv('growth_fits.csv',fits);save_csv('growth_local_rates.csv',local)
metadata=dict(input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),K0=K,gamma_theory=theory,
    model='G(n)=A*gamma^(n-n_ref)',loss='equal-weight least squares in log(G)',
    interval_rule='n>=6, contiguous, all matched weak gates pass; at least 3 points',
    uncertainty='window and reader sensitivity only; correlated RG steps and readers, no statistical confidence interval',
    theory_source='https://arxiv.org/html/1005.1704#S2.SS2',primary=main,windows=fits,local_rates=local)
(H/'growth_fits.json').write_text(json.dumps(metadata,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'figure.dpi':160})
fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained',gridspec_kw={'height_ratios':[1.1,1]})
windows=[(6,10),(6,12),(6,14),(8,12),(8,14),(10,14)]
for col,(h,rr) in enumerate(data.items()):
    ax=axs[0,col];lo,hi=primary[h]
    ax.axvspan(lo,hi,color='C0',alpha=.06)
    n=np.array([r['step'] for r in rr]);grid=np.linspace(lo,hi,120)
    for i,k in enumerate(KEYS):
        f=next(f for f in main if f['h']==h and f['reader']==k)
        ax.plot(n,[r['dG'][k] for r in rr],color=f'C{i}',lw=.8,alpha=.4)
        for r in rr:ax.plot(r['step'],r['dG'][k],'o',color=f'C{i}',ms=4,mfc=f'C{i}' if lo<=r['step']<=hi else 'white')
        ax.plot(grid,f['amplitude']*np.exp(f['log_gamma']*(grid-f['n_ref'])),'--',color=f'C{i}',label=rf'{k}: $\gamma={f["gamma"]:.4f}$')
    ax.set(yscale='log',xlabel='RG step',ylabel=r'$G=\delta\widehat{g_d^2}$',title=f'h={h:g}: fit RG{lo}-{hi} ({(hi-lo)//2+1} points)')
    ax.set_xticks(range(0,17,2));ax.legend(loc='upper left',fontsize=9)
    ax=axs[1,col]
    for j,(start,end) in enumerate(windows):
        for i,k in enumerate(KEYS):
            f=next(f for f in fits if f['h']==h and f['reader']==k and f['start']==start and f['end']==end)
            marker='D' if f['primary'] else ('o' if f['admissible'] else 'x')
            ax.plot(j+(i-1.5)*.12,f['gamma'],marker,color=f'C{i}',ms=6 if f['primary'] else 5,label=k if j==0 else None)
    ax.axhline(theory,color='black',ls='--',lw=1,label=rf'theory {theory:.4f}')
    ax.set_xticks(range(len(windows)),[f'{a}-{b}' for a,b in windows]);ax.set(xlabel='Fit window (RG steps)',ylabel=r'One-step growth $\gamma$',ylim=(1.39,1.48),title='Window sensitivity')
    ax.legend(fontsize=8,ncol=3,loc='upper left')
fig.suptitle(r'Tube [1,4,1], $\chi=32$, $K_0\simeq1.5$: exponential fits')
for ext in ('png','svg'):fig.savefig(H/'figures'/f'growth_fits_141.{ext}')
plt.close(fig)

def rates(h):return [f['gamma'] for f in main if f['h']==h]
g1=rates(.01);g2=rates(.02)
section=r'''### RGを1段進めたときの増大率

まずは、背景を引いた値 $G_r(n)=\delta\widehat{g_{\mathrm d}^2}_r(n)$ を、次の指数関数でfitするのが自然です。$r$ は $m1,m2,m3,X$ の読み方、$\gamma_r$ はRGを1段進めたときの倍率です。

$$
G_r(n)=A_r\gamma_r^{\,n-n_0},\qquad
\log G_r(n)=\log A_r+(n-n_0)\log\gamma_r.
$$

正の値に対して $\log G_r$ を同じ重みで直線fitします。これで、値が大きい後半だけにfitが引っ張られるのを抑え、各点の相対的なずれを比べられます。$n_0$ はfit区間の中央に置き、$A_r$ と $\gamma_r$ を求めました。

二量体化の演算子の次元は $K_0$ なので、弱い摂動では $g_{\mathrm d}\propto L^{2-K_0}$ と育ちます。演算子の対応は [Takayoshi–Sato, Sec. II.2](https://arxiv.org/html/1005.1704#S2.SS2) を参照してください。今回の $L\propto2^{n/2}$ を代入すると、図の **二乗の量** に対する理論値は

$$
\gamma_{\mathrm{th}}=2^{2-K_0}=THEORY\simeq\sqrt{2}
$$

です。保存点の間隔は2段なので、隣の点では約2倍になります。$g_{\mathrm d}$ 自体の1段あたりの倍率は $\sqrt{\gamma}$ です。

fitには、初期の読み方による差が大きいRG0・2・4を外し、上の弱い摂動の目安を満たす連続した範囲を使いました。$h=0.01$ はRG6〜14の5点、$h=0.02$ はRG6〜10の3点です。理論値への近さで区間を選ぶことはしていません。

上段の破線がfit、塗りつぶした点がfitに使った点です。下段では区間を動かして求めた倍率を比べます。ひし形は上段のfit、丸は目安を満たす別区間、×は弱い摂動の範囲を超える点を含む参考値です。

![指数fitと区間による増大率の変化](../support/charge_tube_20261106/figures/growth_fits_141.png)

RESULTS

$h=0.02$ は3点で2つのパラメータを決めているため、増大率の暫定値として扱います。$h=0.01$ は区間を動かすと、とくに $m3$ と $X$ の値が変わります。この幅を、いま残っている読み方・区間への依存として見ます。4通りは同じ3つの電荷準位から作っているため、平均して独立な4測定として誤差を小さくする扱いは避けます。

定数を足した $G=A\gamma^{n-n_0}+C$ や、さらにdriftを足すfitは、点が増えてから比較するのがよさそうです。とくに3点では $A,\gamma,C$ を動かす余地が大きく、増大率が背景の仮定に強く依存します。背景差し引き後にも残る有限 $\chi$ や高次の補正は、今回の区間依存と理論値との差に含まれます。まずは2パラメータの指数fitと区間比較を採用します。

全区間のfitは[CSV](../support/charge_tube_20261106/growth_fits.csv)、条件と理論値は[JSON](../support/charge_tube_20261106/growth_fits.json)、隣接点の倍率 $[G(n+2)/G(n)]^{1/2}$ は[別CSV](../support/charge_tube_20261106/growth_local_rates.csv)に保存しました。[解析コード](../support/charge_tube_20261106/fit_growth.py)から再現できます。ここで示した幅は読み方の違いで、統計的な信頼区間は付けていません。
'''
result=(f'$h=0.01$ の増大率は4通りで **{min(g1):.4f}〜{max(g1):.4f}**、$h=0.02$ は **{min(g2):.4f}〜{max(g2):.4f}** でした。理論値との差は、それぞれ約{100*(min(g1)/theory-1):.1f}〜{100*(max(g1)/theory-1):.1f}%、約{100*(min(g2)/theory-1):.1f}〜{100*(max(g2)/theory-1):.1f}%です。')
(H/'growth_section.md').write_text(section.replace('THEORY',f'{theory:.7f}').replace('RESULTS',result))
print(json.dumps(main,indent=2))
