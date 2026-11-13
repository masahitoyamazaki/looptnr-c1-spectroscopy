"""Saved-spectrum check of Paper 1a review R-02; no RG or eigensolver runs.

Run from any directory: python3 Fushio/my_notes/kn_drift_validation_20261112/analyze.py
Outputs remain beside this script. Raw collaborator inputs are read-only.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import platform
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'runs/main'
FOLLOW = ROOT / 'runs/additional'
KS = (1.25, 1.5, 2.5)
CHIS = (24, 32, 40, 48, 64)
INPUTS = {}


def read(path):
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        return list(csv.DictReader(f))


GROWTH = {(float(r['K']), int(r['chi'])): r
          for r in read(BASE / 'analysis/o2_fits.csv')}
ORIGINAL = {(float(r['K']), int(r['chi'])): r
            for r in read(BASE / 'analysis/drift_summary.csv')}
INPUTS[str((BASE / 'analysis/drift.py').relative_to(ROOT))] = hashlib.sha256(
    (BASE / 'analysis/drift.py').read_bytes()).hexdigest()


def extract(path, K, chi, tube):
    bystep = {}
    for row in read(path):
        row = dict(row, x=float(row['x']), rank=int(row['rank']))
        bystep.setdefault(int(row['step']), []).append(row)
    out = []
    for n, rows in sorted(bystep.items()):
        sectors = {tag: sorted([r for r in rows if tag in r['sector']], key=lambda r: r['x'])
                   for tag in ('(0, 0)', '(0, 1)', '(1, 2)')}
        even, odd, charge = sectors['(0, 0)'], sectors['(0, 1)'], sectors['(1, 2)']
        if len(even) < 2 or len(odd) < 3 or not charge:
            continue
        e = min(even[1:], key=lambda r: abs(r['x'] - K))
        o = min([r for r in odd if abs(r['x'] - 1) > .02] or odd,
                key=lambda r: abs(r['x'] - K))
        q = charge[0]
        currents = sorted(odd, key=lambda r: abs(r['x'] - 1))[:2]
        # The original script uses only the eigenvalue nearest 1. We retain
        # that convention and also average the two current partners.
        jnear = currents[0]['x']
        jpair = np.mean([r['x'] for r in currents])
        xe = (e['x'] + o['x']) / 2
        if min(q['x'], xe, jnear, jpair) <= 0:
            continue
        kn = .5 * np.sqrt(xe / q['x'])
        rec = dict(K=K, chi=chi, tube=tube, n=n, xq=q['x'], xe=xe,
                   xE=e['x'], xO=o['x'], xJ_nearest=jnear, xJ_pair=jpair,
                   K_original=kn, K_e=xe/jpair, K_m=jpair/(4*q['x']),
                   K_e_nearest=xe/jnear, K_m_nearest=jnear/(4*q['x']),
                   ranks=f"{e['rank']}/{o['rank']}/{q['rank']}/"+
                         ','.join(str(r['rank']) for r in currents),
                   current_split=abs(currents[0]['x']-currents[1]['x']))
        selected = [e, o, q, *currents]
        rec['max_current_distance'] = max(abs(r['x']-1) for r in currents)
        rec['max_electric_distance'] = max(abs(e['x']-K), abs(o['x']-K))
        rec['residual'] = max(float(r.get('residual', 'nan')) for r in selected)
        rec['converged'] = (all(int(r['converged']) >= r['rank'] for r in selected)
                            if 'converged' in selected[0] else None)
        rec['identity_error'] = abs(rec['K_e']*rec['K_m'] / kn**2 - 1)
        assert rec['identity_error'] < 1e-12
        out.append(rec)
    return out


def window(rows, K, chi, start=12, cut=1e-4, end=80):
    g = GROWTH[(K, chi)]
    def in_floor(r):
        if K > 2:
            return True
        assert g.get('A'), (K, chi, 'growth fit missing')
        return abs(float(g['A'])) * float(g['r'])**(r['n']-int(g['n_e'])) < cut
    return [r for r in rows if start <= r['n'] <= end and in_floor(r)]


def fit(rows, key):
    x = np.array([r['n'] for r in rows], float)
    y = np.array([r[key] for r in rows], float)
    x0, y0 = x-x.mean(), y-y.mean()
    slope = np.dot(x0, y0)/np.dot(x0, x0)
    residual = y0-slope*x0
    return float(slope), float(np.sqrt(np.mean(residual**2)))


def summarize(rows, **meta):
    d = dict(meta, points=len(rows), n_first=rows[0]['n'], n_last=rows[-1]['n'])
    for key in ('K_original', 'K_e', 'K_m', 'K_e_nearest', 'K_m_nearest'):
        d['slope_'+key], d['rms_'+key] = fit(rows, key)
    ref = d['slope_K_original']
    d['max_relative_slope_difference'] = max(abs(d['slope_'+k]-ref) for k in ('K_e','K_m'))/abs(ref)
    d['nearest_pair_slope_sensitivity'] = max(abs(d['slope_'+k]-d['slope_'+k+'_nearest'])
                                                for k in ('K_e','K_m'))/abs(ref)
    # Compare changes, not constant offsets, over the identical window.
    d['original_fitted_change'] = ref*(rows[-1]['n']-rows[0]['n'])
    d['max_current_split'] = max(r['current_split'] for r in rows)
    d['max_current_distance'] = max(r['max_current_distance'] for r in rows)
    d['max_electric_distance'] = max(r['max_electric_distance'] for r in rows)
    d['selected_ranks'] = ';'.join(sorted({r['ranks'] for r in rows}))
    d['max_eigen_residual'] = max(r['residual'] for r in rows)
    d['all_selected_converged'] = all(r['converged'] for r in rows) if rows[0]['converged'] is not None else None
    return d


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (HERE/name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def report(summary, variations, wide):
    def table(rows):
        out = ['| $K$ | $\\chi$ | RG 窓 | $10^5 b_{em}$ | $10^5 b_e$ | $10^5 b_m$ | 最大の傾き差 / $\\lvert b_{em}\\rvert$ |',
               '|---|---|---|---|---|---|---|']
        for r in rows:
            vals = ' | '.join(f"{1e5*r['slope_'+k]:+.4f}" for k in ('K_original','K_e','K_m'))
            out.append(f"| {r['K']} | {r['chi']} | {r['n_first']}–{r['n_last']} | {vals} | {r['max_relative_slope_difference']:.2%} |")
        return '\n'.join(out)
    envelope = ['| $\\chi$（$K=2.5$） | 窓・推定法を変えた傾きの全範囲（$10^{-5}$/step） |', '|---|---|']
    for chi in CHIS:
        rows=[r for r in variations if r['K']==2.5 and r['chi']==chi]
        v=[r['slope_'+k]*1e5 for r in rows for k in ('K_original','K_e','K_m')]
        envelope.append(f'| {chi} | {min(v):+.4f} ～ {max(v):+.4f} |')
    wt = ['| $K$ | $\\chi$ | standard の $10^5 b_{em}$ | wide の $10^5 b_{em}$ | 傾き差 / standard の $\\lvert b_{em}\\rvert$ |',
          '|---|---|---|---|---|']
    for r in wide:
        wt.append(f"| {r['K']} | {r['chi']} | {r['standard_slope_K_original']*1e5:+.4f} | {r['slope_K_original']*1e5:+.4f} | {r['relative_tube_difference_K_original']:+.1%} |")
    rr=[r for r in summary if r['K']==2.5]
    exponents={k:float(np.polyfit(np.log([r['chi'] for r in rr]),np.log([-r['slope_'+k] for r in rr]),1)[0])
               for k in ('K_original','K_e','K_m')}
    (HERE/'chi_exponents.json').write_text(json.dumps(exponents,indent=2)+'\n')
    text = r'''# $K_n$ の drift の推定法依存性：R-02 の追加検証

2026-11-12。Fushio の依頼に基づく Codex による解析。**[AI未確認：追加解析・解釈は本人確認待ち]**。
元の R-02 の本人確認と、この追加結果の確認は別に扱う。

## 結果と判断

**$K=2.5$ では負の drift を支持する。標準窓で別の推定法と元の推定法との傾き差は、各 $\chi$ で最大2.3–6.4%であり、窓を変えても全て負だった。**
したがって、少なくともこの条件で「別の準位を使うと drift が消える」という結果ではない。
$K=1.25,1.5$ の $\chi=40$ でも、標準窓の傾きと元の推定法との差はそれぞれ最大0.9%、6.6%だった。
一方、$K=1.25,\chi=64$ と $K=1.5,\chi=24$ は窓によって符号が変わり、一律に同じ精度とは言えない。
wide tube では同じ段数区間でも傾きが変わる。これは readout 条件依存の存在を示すが、wide 自体にも追加切断があり、どちらが正しいかを決めるものではない。

R-02 は「検証が全くない」という形で維持する必要はない。**推定法・窓に対する安定性は支持された範囲を追記し、共通の系統誤差は未評価という点を残す**のが適切。

## データ・方法

- RG・固有値計算は追加せず、[標準スペクトル](../../../Yamazaki/runs/cpu_baseline_20261106/data/)の保存CSVを読む。
- 対象は $K=1.25,1.5,2.5$、$\chi=24,32,40,48,64$ の15条件。
- [既存 drift.py](../../../Yamazaki/runs/cpu_baseline_20261106/analysis/drift.py) と同じ $x_e=(x_E+x_O)/2$、最低電荷1準位 $x_q$ を使用。電気準位はセクター内で入力 $K$ に最も近いものを選ぶ。対象窓では電気準位の順位は各run内で一定だった。
- current は $C$-odd neutral の $x=1$ に近い2準位（順位1,2）の平均を主解析に使用。元コードの「1に最も近い片方」も別に計算し、選択の影響を記録。
- 標準窓は $n\ge12$、既存成長フィットの $|A r^{n-n_e}|<10^{-4}$。$K=2.5$ は $n=12$–80。全推定を同一の点集合で比較する。
- 標準データの選択準位は保存された convergence 数と順位を照合。固有対残差の最大は $1.24\times10^{-12}$。これは固有値計算の精度であり、切断・readout の系統誤差の上限ではない。

比較する量は

$$
K_{em}=\frac12\sqrt{\frac{x_e}{x_q}},\qquad
K_e=\frac{x_e}{x_J},\qquad
K_m=\frac{x_J}{4x_q}.
$$

共通の倍率誤差は全てで消える。各量を $a+bn$ に最小二乗フィットし、傾き $b=dK/dn$ を比較する。
定数オフセットは drift と区別する。$K_{em}^2=K_eK_m$ が厳密に成立するので、独立な3測定ではなく current を追加した整合性検査である。

## 標準窓での全結果

TABLE

![chi40での3推定](drift_comparison.png)

点は各推定の最初の値からの変化、同色の線は一次フィット。誤差棒は付けていない。

![chiごとの傾き](slope_comparison.png)

横方向の微小なずらしは点の重なりを避けるため。結ぶ線は視線誘導であり理論曲線ではない。

## 窓と current の選択への感度

- $K=2.5$：始点12,16,20、終点60,70,80の9窓。
- $K<2$：始点10,12,14、成長項のcut $5\times10^{-5},10^{-4},2\times10^{-4}$ の組合せ。6点未満は傾きの比較から除外した。10開始は過渡補正を含み得るため感度検査として扱う。
- これらは決定論的データの解析感度であり、統計的信頼区間ではない。一次フィットのRMS残差だけから系統誤差を推定しない。

ENVELOPE

$K=2.5$ では current の片方から平均への変更による傾きの差も元の傾きの0.1–3.0%で、負の符号は安定。
ただし $K=1.25,\chi=64$ の元の推定の傾きは窓により $-2.52\times10^{-6}$ ～ $+0.60\times10^{-6}$、
$K=1.5,\chi=24$ は $-5.72\times10^{-5}$ ～ $+2.33\times10^{-4}$ となる。
短い窓・小さい drift では、この確認だけで物理的な符号や大きさを確定しない。

$K=2.5$ の5つの $\chi$ で $|b|\propto\chi^p$ をフィットすると、
EXPONENTS
となる。約 $-3.2$ という経験的な指数は推定法を変えても大きくは変わらないが、理論的導出や漸近指数の確定ではない。

## 保存済み wide-tube 読み出しとの比較

[wide_readout.jl](../../../Yamazaki/runs/followup_B_20261106/wide_readout.jl) による
[W1データ](../../../Yamazaki/runs/followup_B_20261106/data/)から、標準窓と共通する同じ段数だけを使った。
円周は標準の $\sqrt5$ 倍、$\tau=0.1+0.2i$、追加の coarse-graining を同じ $\chi$ で切断する。
元のRG軌道の $\chi$ を増やした検査ではない。標準窓の選択条件をそのまま使っており、wide では relevant 補正の振幅自体も増えるため、差を純粋な測定誤差だけには帰属できない。
W1のCSVには固有対残差・convergence欄がなく、標準と同じ数値精度の検査はできない。
$K=2.5$ の対応するW1データは今回の保存集合にない。

WIDETABLE

$\chi=40$ では $K=1.25$ で31%、$K=1.5$ で10%程度の変化があり、負の符号は共通。
$K=1.5,\chi=48$ では元の推定の傾きが46%変わり、wide側の3推定間にも大きな差がある。
全ての条件について high-precision な drift 測定が確立したとは言えない。

## この検査で分かること・残ること

1. $K=2.5$ の負の drift と、$\chi=40$ の $K=1.25,1.5$ の負の drift は、別の準位を使った推定でも残る。
2. $K_n$ の基準値がずれるだけでなく、傾きそのものに対する推定法・窓の感度を測った。これは R-02 に対する具体的な追加証拠になる。
3. 全準位に「別の $K(n)$」と同じ形で入る共通バイアスは、同じスペクトル内の検査では除去できない。無バイアスの証明ではない。
4. 小さいdriftの条件とwide比較の不一致を残し、原稿は支持された条件・精度を限定して述べるのがよい。新規RGや追加readoutによる全系統誤差評価までは実施していない。

## 再実行・検証

```sh
python3 Fushio/my_notes/kn_drift_validation_20261112/analyze.py
```

- [analyze.py](analyze.py)：解析・図・本ノートを生成。元データは読み取りのみ。
- [slopes.csv](slopes.csv)：15条件の傾き・RMS残差・準位順位・品質確認。
- [estimator_steps.csv](estimator_steps.csv)：各段の選択準位と推定量。
- [window_sensitivity.csv](window_sensitivity.csv)：窓変更の結果。
- [wide_tube_comparison.csv](wide_tube_comparison.csv)：同じ段数区間でのtube比較。
- [chi_exponents.json](chi_exponents.json)：指数の経験的fit。
- [manifest.json](manifest.json)：入力ファイルとスクリプトのSHA-256、実行環境。入力の前後ハッシュ一致を確認。

元の $K_{em}$ の窓・点数と傾きは、保存された `drift_summary.csv` の15条件すべてを再現した（傾き差 $<10^{-12}$）。
恒等式 $K_{em}^2=K_eK_m$ も各段で検査した。図の表示を目視確認した。
'''
    text=text.replace('TABLE',table(summary),1).replace('ENVELOPE','\n'.join(envelope))
    text=text.replace('EXPONENTS',', '.join(f"${label}$: $p={exponents[k]:.3f}$" for k,label in [('K_original','K_{em}'),('K_e','K_e'),('K_m','K_m')]))
    text=text.replace('WIDETABLE','\n'.join(wt))
    (HERE/'README.md').write_text(text)


def main():
    allsteps, summary, variations, wide = [], [], [], []
    grouped = {}
    for K in KS:
        for chi in CHIS:
            p = BASE / f'data/CU1_K{K:.4f}_chi{chi}/levels.csv.gz'
            rows = extract(p, K, chi, 'standard')
            allsteps += rows
            f = window(rows, K, chi)
            assert len(f) >= 4
            assert all(r['converged'] for r in f)
            assert max(r['residual'] for r in f) < 1e-8
            assert max(r['max_current_distance'] for r in f) < .05
            assert max(r['max_electric_distance'] for r in f) < .15
            d = summarize(f, K=K, chi=chi, tube='standard')
            original = ORIGINAL[(K, chi)]
            assert (d['n_first'], d['n_last'], d['points']) == tuple(int(original[k]) for k in ('n_first','n_last','floor_points'))
            d['original_slope_reproduction_error'] = abs(d['slope_K_original']-float(original['dK_per_step']))
            assert d['original_slope_reproduction_error'] < 1e-12
            summary.append(d);grouped[K,chi] = rows
            configs = ([(s, 1e-4, end) for s in (12,16,20) for end in (60,70,80)] if K>2 else
                       [(s, cut, 80) for s in (10,12,14) for cut in (5e-5,1e-4,2e-4)])
            for start,cut,end in configs:
                v = window(rows,K,chi,start,cut,end)
                if len(v) >= 6:
                    variations.append(summarize(v,K=K,chi=chi,start=start,cut=cut,end=end))
            wp = FOLLOW/f'data/W1_K{K:.4f}_chi{chi}/levels_wide.csv.gz'
            if wp.exists():
                wr = extract(wp,K,chi,'wide'); allsteps += wr
                # Fix the step window from the standard data for a paired comparison.
                ns = {r['n'] for r in f}&{r['n'] for r in wr}
                wf = [r for r in wr if r['n'] in ns]
                sf = [r for r in f if r['n'] in ns]
                if len(wf)>=4:
                    wd=summarize(wf,K=K,chi=chi,tube='wide')
                    sd=summarize(sf,K=K,chi=chi,tube='standard')
                    for key in ('K_original','K_e','K_m'):
                        wd['standard_slope_'+key]=sd['slope_'+key]
                        wd['relative_tube_difference_'+key]=(wd['slope_'+key]-sd['slope_'+key])/abs(sd['slope_K_original'])
                    wide.append(wd)
    write_csv('estimator_steps.csv', allsteps)
    write_csv('slopes.csv', summary)
    write_csv('window_sensitivity.csv', variations)
    write_csv('wide_tube_comparison.csv', wide)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,K in zip(axes,KS):
        rows=window(grouped[K,40],K,40)
        for color,(key,label) in zip(('C0','C1','C2'),[('K_original',r'$K_{em}$'),('K_e',r'$K_e=x_e/x_J$'),('K_m',r'$K_m=x_J/(4x_q)$')]):
            n=np.array([r['n'] for r in rows]);y=np.array([r[key] for r in rows]);slope,_=fit(rows,key)
            ax.plot(n,1e3*(y-y[0]),'o',ms=3,label=label,color=color)
            ax.plot(n,1e3*(slope*(n-n.mean())+y.mean()-y[0]),lw=1,color=color)
        ax.set_title(fr'$K={K},\ \chi=40$');ax.set_xlabel('RG step n')
        ax.set_ylabel(r'$10^3[K(n)-K(n_{start})]$');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(HERE/'drift_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    for ax,K in zip(axes,KS):
        data=[r for r in summary if r['K']==K]
        for key,label,offset in [('K_original',r'$K_{em}$',-0.5),('K_e',r'$K_e$',0),('K_m',r'$K_m$',.5)]:
            ax.plot([r['chi']+offset for r in data],[1e5*r['slope_'+key] for r in data],'o-',ms=4,label=label)
        ax.axhline(0,color='k',lw=.6);ax.set_title(fr'$K={K}$');ax.set_xlabel(r'$\chi$');ax.set_ylabel(r'$10^5\,dK/dn$');ax.grid(alpha=.2)
    axes[0].legend();fig.savefig(HERE/'slope_comparison.png',dpi=180);plt.close(fig)
    report(summary,variations,wide)
    before=dict(INPUTS)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in before.items())
    manifest=dict(date='2026-11-12',reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,
                  inputs=before,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_hashes_unchanged=True,standard_runs=len(summary),wide_runs=len(wide),
                  caveats=['Deterministic sensitivity study; no statistical confidence interval or physical-convergence claim.',
                           'Three estimators satisfy K_em^2=K_e*K_m; shared biases can remain.',
                           'Level assignment follows the manuscript by proximity; selected ranks are recorded.',
                           'Wide readout has an additional truncation and no per-level residual/convergence fields.'])
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for r in summary:
        print(f"K={r['K']:4} chi={r['chi']:2} n={r['n_first']}-{r['n_last']} "
              f"slopes={r['slope_K_original']:+.6e},{r['slope_K_e']:+.6e},{r['slope_K_m']:+.6e} "
              f"difference={r['max_relative_slope_difference']:.1%} "
              f"pairSensitivity={r['nearest_pair_slope_sensitivity']:.1%}")
    print('Wide comparisons:',len(wide))


if __name__=='__main__':
    main()
