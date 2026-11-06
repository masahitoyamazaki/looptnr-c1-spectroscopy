"""Refit saved O(2), chi=40 wide-tube readouts for manuscript figures 2 and 3.

Run from repository root:
  MPLCONFIGDIR=/tmp/sixvertex_mpl python3 Fushio/code/six_vertex/scripts/paper_o2_wide_20261101/analyze.py

Analysis and numerical artifacts: Fushio/. Figures and explanation: research_note/.
No RG/CFT work, checkpoint writes, or changes to Otsuki inputs.
"""
from __future__ import annotations
import csv
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[5]
RESULTS = ROOT / 'Fushio/code/six_vertex/results/paper_o2_wide_20261101'
FIGURES = ROOT / 'research_note/paper/latex/figures'
DOC = ROOT / 'research_note/paper/docs/sixvertex_o2_wide_refit_20261101.md'
SOURCE = ROOT / 'Otsuki/tnr/practice/data'
TABLE = ROOT / 'research_note/calculation_table/support/representative_calculations.csv'
FIT_CODE = ROOT / 'Fushio/code/six_vertex/scripts/paper_chi40_x_rate_20261101.py'
spec = importlib.util.spec_from_file_location('previous_fit', FIT_CODE)
method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(method)
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


def plot(records,predictions,reps):
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'dejavuserif','font.size':9,'axes.linewidth':.7,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(6.9,3.9),layout='constrained')
    for ax,k in zip(axes.flat,reps):
        rec=next(row for row in records if abs(row['K']-k)<1e-6)
        pts=[row for row in predictions if abs(row['K']-k)<1e-6]
        ns=np.array([row['n'] for row in pts]);ys=np.array([row['subtracted'] for row in pts])
        fit=np.array([row['fit_subtracted'] for row in pts]);dim=np.array([row['dimer'] for row in pts])
        fitted=rec.get('early',rec.get('control'))
        first,last=fitted['first'],fitted['last']
        inside=(ns>=first)&(ns<=last)
        ax.axvspan(first,last,color='#dddddd',alpha=.35,lw=0,zorder=0)
        def fit_line(values,color,label,style='-'):
            ax.plot(ns[inside],abs(values[inside]),style,color=color,lw=1,label=label)
            for outside in (ns<=first,ns>=last):
                ax.plot(ns[outside],abs(values[outside]),':',color=color,lw=1)
        fit_line(fit,'#777777','full fit')
        fit_line(dim,'#c04d36','fixed decay' if k>2 else 'growth term','--')
        if abs(k-.6)<1e-6:
            fit_line(np.array([row['umklapp'] for row in pts]),'#d68a1b','umklapp','--')
        ax.axhline(fitted['rms'],color='#555555',lw=.7,ls='-.')
        ax.scatter(ns,abs(ys),s=10,color='#2764b2',edgecolors='none',zorder=3)
        ax.set(yscale='log',xlabel=r'RG step $n$',title=fr'$K={k:.2f}$'+(' (control)' if k>2 else ''))
        ax.set(xlim=(0,ns[-1]+1),ylim=(1e-6,1.))
        if ax in axes[:,0]:ax.set_ylabel(r'$|\widetilde X(n)|$')
        ax.tick_params(labelsize=7);ax.grid(alpha=.15,lw=.5)
        ax.legend(fontsize=6.2,frameon=False,loc='upper center')
    for ext in ['pdf','png']:fig.savefig(FIGURES/f'sixvertex_charged_background_wide.{ext}',dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.45,2.7),layout='constrained')
    kk=np.linspace(.52,3.15,400);ax.plot(kk,2**(2-kk),color='#222222',lw=1.2,label=r'$2^{2-K}$')
    rr=[row for row in records if row['status']=='estimated']
    ax.errorbar([row['K'] for row in rr],[row['main']['r'] for row in rr],
                yerr=[[row['main']['r']-row['error_lo'] for row in rr],[row['error_hi']-row['main']['r'] for row in rr]],
                fmt='o',markersize=3.5,capsize=2,elinewidth=.8,color='#215da3',markeredgecolor='white',markeredgewidth=.35,
                label=r'$O(2),\ \chi=40$ (wide)',zorder=3)
    unresolved=[row for row in records if row['status']!='estimated']
    ax.scatter([row['K'] for row in unresolved],[1.]*len(unresolved),marker='v',s=22,color='#888888',label=r'unresolved (at $r=1$)')
    ax.axhline(1,color='#aaaaaa',lw=.7,ls=':')
    ax.set(xlim=(.52,3.18),ylim=(.4,3.25),xlabel=r'$K$',ylabel=r'rate per RG step $r$')
    ax.grid(alpha=.15,lw=.5);ax.legend(fontsize=6.2,frameon=False)
    for ext in ['pdf','png']:fig.savefig(FIGURES/f'sixvertex_growth_rate_wide.{ext}',dpi=220)
    plt.close(fig)


def report(records,sources,reps,validation):
    fitted=[r for r in records if r['status']=='estimated']
    text='''# O(2)・χ=40：広い tube の荷電準位を再フィット

本文の図2・図3を、保存済みの広い tube `[√2,2√2,0]` に統一した。計算表では「中」と呼ぶ形状である。
図2は K=0.60,0.75,1,1.5,1.75,2.5。calculation_table の代表点のうち K=1.25 を外し、umklapp の減衰を示す K=0.60 を追加した。
計算表そのものは変更していない。図3は引き続き保存ログの全22条件を対象とする。
解析コードと数値成果物は Fushio、図とこの説明は research_note に置く。RG/CFT の追加実行は行っていない。

![代表点](../latex/figures/sixvertex_charged_background_wide.png)

## 入力と精度

入力は Otsuki が Fushio の保存テンソルを読んだ `sixvertex_rate_cft_fushio*log` の χ=40 の行。
`Otsuki/tnr/practice/sixvertex_rate_measure.jl` が `CFTData` の既定形状 `[√2,2√2,0]` で測定した出力である。
元の高精度 `.jls` は今回のローカル環境にはない。したがって、この解析は**丸められた測定ログの再解析**である。
各行の `K_m`（小数5桁）と `(K_m-K)/K`（指数形式、小数2桁）の丸め区間の共通部分を取り、
その中点を代表値とした。K 自体はログの丸め表示を使わず、頂点重みから `π/(4 asin(c_v/2))` で計算した。
`X=1-K/K_m=1-4Kx_01` へ変換した区間端点も保存する。中点の桁数は測定精度を増やすものではない。
保存値の有限性と step の連続性を確認したが、ログから固有値残差や checkpoint hash は復元できない。
出典の各行番号と入力 SHA-256 は解析成果物に記録した。元テンソルとの現在の内容照合は行っていない。

## フィットと表示の順序

1. K<1.9 の早期 fit は RG6 から最初の |X|>0.10 の直前まで、`a+bn+A r^(n-n_end)+B s^(n-n_start)+C(-1/2)^(n-n_start)+D(1/2)^(n-n_start)`、`s=2^(2-4K)`。
   r は自由。K=0.75 では s=1/2 なので B,D を一つにまとめる。
2. 図2はその同時 fit の `a+bn` を除いた値の絶対値を、符号を区別せず塗り点で示す。
   データ表示は RG1 から最初の |X|>0.30 の直前まで（超えない条件は最終保存段まで）に広げた。fit に使う区間と入力値は変更していない。
   淡い帯は fit 区間、灰線は背景を除いた全 fit、赤破線は二量体化の項、K=0.60 の橙破線は umklapp の項、一点鎖線は残差 RMS。fit 区間外の線は点線の外挿であり、その範囲でのモデルの妥当性を保証しない。
   特に K>=1 の umklapp 項は初期へ外挿すると急増するため、区間外の全 fit がデータから外れうる。線は同じ fit の分解であり、独立な証拠ではない。
3. 主 fit は RG12 から同じ上限で C,D を外す。早期 fit の n=12 における B 項の絶対振幅が残差 RMS より大きければ B を残す。
   図3には主 fit の r を示す。
4. K>=1.9 では自由な成長率を報告しない。背景表示のための対照 fit は、定数・線形項、±1/2 の項と、倍率を `2^(2-K)` に固定した項。
   K=2 では固定項を定数に吸収する。**K=2.5 の破線は r=2^(-1/2) を仮定した減衰成分であり、その指数を測ったものではない。**
   K=2.5 の背景は従来どおり RG6--70 の同時 fit で求める。初期だけで決めた線形背景を後半へ外挿すると背景推定誤差が蓄積するため、初期区間だけの fit には変更していない。
   後半が残差 RMS 程度にとどまることを示す対照であり、成長項の厳密な不在を証明するものではない。

誤差棒は、主 fit の profile `RSS(r)<=RSS_min[1+1/(N-p)]`、開始段10/12/14・上限5%/10%・代替観測量 `K_m/K-1` の窓感度、
主窓への C,D 再導入、ログ丸め区間の18通りの摂動（両端と固定seedの16通り）の最小・最大の包絡。
丸め摂動でも B の採否は中点データでの選択に固定する。この包絡は統計的信頼区間でも全系統誤差の上限でもない。
丸め摂動の16通りには確率的誤差モデルを与えていない。χ収束、tube 依存、準位同定の誤差は含まない。

![成長率](../latex/figures/sixvertex_growth_rate_wide.png)

## 結果

| K | 主窓 | r | 感度包絡 | 理論 | B | 丸めによる最大変化 | C,D 再導入 |
|---:|:---:|---:|:---:|---:|:---:|---:|---:|
'''
    for r in fitted:
        m=r['main'];delta=max(abs(m['r']-r['rounding_min']),abs(m['r']-r['rounding_max']))
        text+=f"| {r['K']:.5f} | {m['first']}–{m['last']} | {m['r']:.5f} | {r['error_lo']:.5f}–{r['error_hi']:.5f} | {r['theory']:.5f} | {'有' if r['umklapp_term'] else '無'} | {delta:.2g} | {r['lattice_main_r']:.5f} |\n"
    text+='\n未同定条件: '+', '.join(f"K={r['K']:.5f}" for r in records if r['status']!='estimated')+'。図3の灰色三角は r=1 の測定値ではない。\n'
    text+='''
## 再現と成果物

リポジトリ直下で実行する。

```sh
MPLCONFIGDIR=/tmp/sixvertex_mpl python3 Fushio/code/six_vertex/scripts/paper_o2_wide_20261101/analyze.py
```

- [解析コード](../../../Fushio/code/six_vertex/scripts/paper_o2_wide_20261101/analyze.py)
- [全フィット・対照](../../../Fushio/code/six_vertex/results/paper_o2_wide_20261101/summary.json)
- [採用前の入力値・丸め区間・出典](../../../Fushio/code/six_vertex/results/paper_o2_wide_20261101/input_points.csv)
- [背景除去値](../../../Fushio/code/six_vertex/results/paper_o2_wide_20261101/predictions.csv)
- [感度検査](../../../Fushio/code/six_vertex/results/paper_o2_wide_20261101/sensitivity_fits.csv)
- [入力・ソースの hash と検証結果](../../../Fushio/code/six_vertex/results/paper_o2_wide_20261101/provenance.json)

同じ fit 実装の合成データ回復検査、モデル各項からの RSS 再構成、ログ丸め区間の整合性を検証した。
この検証は元の CFT 測定の精度・収束検証に代わるものではない。
'''
    DOC.write_text(text)


def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    validation=check_synthetic()
    groups,sources,reps=load()
    assert len(groups)==22
    records,fits,predictions=analyze(groups)
    assert sum(r['status']=='estimated' for r in records)==15
    write_csv(RESULTS/'input_points.csv',[row for rows in groups.values() for row in rows])
    write_csv(RESULTS/'predictions.csv',predictions)
    write_csv(RESULTS/'sensitivity_fits.csv',fits)
    (RESULTS/'summary.json').write_text(json.dumps(records,indent=2)+'\n')
    plot(records,predictions,reps)
    report(records,sources,reps,validation)
    for path in [FIT_CODE,Path(__file__),TABLE,ROOT/'Otsuki/tnr/practice/sixvertex_rate_measure.jl',ROOT/'Otsuki/tnr/TNRKit/src/utility/cft.jl']:
        sources[str(path.relative_to(ROOT))]=sha(path)
    assert all(sha(ROOT/path)==value for path,value in sources.items())
    (RESULTS/'provenance.json').write_text(json.dumps(dict(inputs_sha256=sources,representative_K=reps,
        validation=validation,geometry='[sqrt(2),2sqrt(2),0]',new_rg_steps=0,new_cft_measurements=0,
        input_precision='intersection of rounded K_m and relative-shift intervals; midpoint used',
        rounding_sensitivity_seed=20261101,rounding_trials=18,
        outputs_sha256={str(path.relative_to(ROOT)):sha(path) for path in [RESULTS/'summary.json',RESULTS/'input_points.csv',RESULTS/'predictions.csv',RESULTS/'sensitivity_fits.csv',DOC,
            *[FIGURES/f'{stem}.{ext}' for stem in ['sixvertex_charged_background_wide','sixvertex_growth_rate_wide'] for ext in ['pdf','png']]]}),indent=2)+'\n')


if __name__=='__main__':
    main()
