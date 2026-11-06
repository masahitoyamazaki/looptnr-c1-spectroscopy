"""Write note13, including the early-RG extension and its saved-point status."""
import json,tomllib,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
D=H.parent.parent/'docs'
rows=json.loads((H/'analysis.json').read_text())['rows']
wide=[r for r in rows if r['tube']=='141'];assert len(wide)>=15
assert all(r['numeric_pass'] for r in wide)
K=wide[0]['K'];factor=2**(4-2*K)
def r(t,h,n):return next(v for v in rows if v['tube']==t and v['h']==h and v['step']==n)
def spread(t,h,n):
    v=r(t,h,n)['spread'];return f'{100*v:.1f}%' if v is not None else '推定値の符号が揃わない'
res=max(v['residual'] for v in wide);agree=max(v['agreement'] for v in wide)
def texnum(value):
    mantissa,exponent=f'{value:.2e}'.split('e')
    return f'${mantissa}\\times10^{{{int(exponent)}}}$'
available={h:[v['step'] for v in wide if v['h']==h and 'dG' in v] for h in (.01,.02)}
eligible={str(h):[n for n in available[h] if r('141',h,n)['pair_pass']] for h in (.01,.02)}
intro=[f'RG6〜8では、小さいtubeに比べて $m=3$ と $X$ のばらつきが小さくなりました。RG8では4通りが正に揃い、最大値と最小値の幅は平均に対して $h=0.01$ で約{spread("141",.01,8)}、$h=0.02$ で約{spread("141",.02,8)}です。']
for h in (.01,.02):
    pairs=[(n,r('141',h,n)['spread']) for n in available[h] if r('141',h,n)['pair_pass'] and r('141',h,n)['spread'] is not None]
    if pairs:
        n,sp=min(pairs,key=lambda v:v[1]);intro.append(f'弱い摂動の目安を満たす範囲では、$h={h:g}$ の幅が最も小さいのはRG{n}で約{100*sp:.1f}%です。同じ段の `[1,2,1]` は{spread("121",h,n)}です。')
body=r'''# 13. 大きいtubeで電荷準位から二量体化を読む

PROGRESS

11と同じ $\chi=32$、$K\simeq1.5$、$h=0.01,0.02$ の保存テンソルを、tube `[1,4,1]` で測り直しました。写真のRG6〜16にRG0・2・4を加え、初期からの変化を比べます。無摂動も同じtubeで測り、$m=1,2,3$ と $X$ の4通りで $g_{\mathrm d}^2$ を読みました。

INTRO

## tubeの選び方

元の空間方向に沿って、2サイトの単位胞を保つ次の周長4を選びます。`[1,4,1]` は、以前の `[1,2,1]` と同じ空間・時間方向を使い、空間の周長を2倍にします。偶数RGでは11と同じ回転補正を入れます。周長と1回の転送の時間幅は次の通りです。

$$
L=8\,2^{n/2},\qquad \Delta t=4\,2^{n/2},\qquad
\tau=\frac{1+\tau_0}{4},\qquad \tau_0=\frac{i v_0\,4}{2}.
$$

ここで $n$ はRG step、$v_0$ は無摂動の有限時間刻みの速度です。時間方向を保ったまま、空間の準位を広い周長で読みます。短い鎖では、この向きと周長の対応を独立なXXZ転送行列で確認しました。

## 電荷準位から得た値

まず、最低電荷準位とCFT値の差 $\Delta x_1=x_{m=1}-1/(4K_0)$ を比べます。引く理論値は3条件とも $1/(4K_0)\simeq0.166712$ です。縦軸は対数、横軸はRG stepです。

![最低電荷準位とCFT値の差](../support/charge_tube_20261106/figures/lowest_charged_141.png)

EARLY_RESULT

下の図は、同じ段・同じtubeの $h=0$ を差し引いた値です。$X$ には11と同じ組み合わせ $15x_1-6x_2+x_3$ を使います。曲線が揃うほど、読み方による違いが小さくなっています。

![大きいtubeでの4通りの結合](../support/charge_tube_20261106/figures/couplings_141.png)

塗りつぶした点は、無摂動・摂動ありの両方で11と同じ弱い摂動の目安を満たす点です。白抜きはその範囲を外れた点です。目安は $m=1$ のCFT値からの差10%未満、電荷準位の位相差0.05 rad未満、固有値の相対残差 $10^{-8}$ 未満です。

EARLY_SPREAD

GROWTH_SECTION

## `[1,2,1]` との比較

周長を2倍にすると、弱い摂動では $g_{\mathrm d}^2$ が $2^{4-2K_0}$ 倍に育ちます。今回は $K_0=KVALUE$ なので約FACTOR倍です。次の図の下段では、その倍率で割ってから11と比較しています。この補正は摂動論の先頭の項に基づきます。

![周長の成長分を除いて比較](../support/charge_tube_20261106/figures/tube_comparison.png)

RG6〜14の下段では、$m=3$ と $X$ が $m=1,2$ に近づき、RGに沿って滑らかに増えています。小さいtubeのRG0では $m=3$ の電荷sectorが存在しないため、$m=3$ と $X$ の点は省いています。

無摂動の準位と4通りのばらつきも比べます。右図の「幅」は最大値と最小値の差を平均で割ったものです。右の縦軸は対数で、4通りが正の点だけを結びます。

![無摂動のずれと読み方によるばらつき](../support/charge_tube_20261106/figures/background_spread.png)

後半では再び差が広がり、$h=0.02$ のRG16では約23%です。同じRGでも周長が2倍になるので、弱い摂動の目安を超える時期も変わります。

今回変えたのは測定の周長です。保存テンソルに残る有限 $\chi$ の影響は引き続き含まれます。4通りの一致は、結合を読む方法どうしの整合性として評価します。

## 計算条件とデータ

- $\delta\tau=1/8$、32行stack、$\chi_{\mathrm{stack}}=64$、$\chi_{\mathrm{loop}}=32$。元のXXZ異方性は $\Delta=-1/2$。
- $h=0,0.01,0.02$ × RG0,2,4,6,8,10,12,14,16の27点。元テンソルとSHA-256は[入力記録](../support/charge_tube_20261106/control/inputs.json)に保存。
- 転送行列の収縮には追加の切断を入れていません。各電荷sectorの最大絶対値の固有値を、独立な2つの初期ベクトルから求めました。
- 最大相対残差は RESIDUAL、2初期値間の最大相対差は AGREEMENT。これは固有値計算の確認です。
- 小さいCU1テンソルで通常の収縮と照合し、差は $1.6\times10^{-15}$ 以下。stack前の短いXXZ鎖との電荷準位の固有値比は、長さ4・8、3つの $h$ で差 $4.8\times10^{-15}$ 以下でした。
- 11の `[1,2,1]` もRG6・8で再測定し、3条件の電荷準位を差ゼロで再現しました。

正確な値は[CSV](../support/charge_tube_20261106/observables.csv)、判定と出典を含む値は[解析JSON](../support/charge_tube_20261106/analysis.json)にあります。[測定コード](../support/charge_tube_20261106/measure.jl)と[作図コード](../support/charge_tube_20261106/analyze.py)を保存しています。新しいRG計算とGoogle Driveへのアップロードは行っていません。
'''
progress='' if len(wide)==27 else f'**初期段を追加計算中：27点中{len(wide)}点を保存済みです。RG0・2・4の測定を追加しています。図には保存済みの点を載せています。**'
early='RG6以降では、無摂動の差は小さいまま推移し、摂動ありの差はRGとともに大きくなります。'
early_spread=''
if all(any(v['h']==h and v['step']==2 for v in wide) for h in (0,.01,.02)):
    early_spread=f'追加したRG2では4通りの幅が $h=0.01$ で約{spread("141",.01,2)}、$h=0.02$ で約{spread("141",.02,2)}あります。初期段では上の目安を満たしていても、結合を読む方法による差が大きく残ります。'
if all(any(v['h']==0 and v['step']==n for v in wide) for n in (0,2,4)):
    early=(f'無摂動の理論値との差は、RG0で {r("141",0,0)["x1_minus_cft"]:.6f}、RG2で {r("141",0,2)["x1_minus_cft"]:.6f}、RG4で {r("141",0,4)["x1_minus_cft"]:.6f} へ小さくなります。'+early)
for a,b in {'PROGRESS':progress,'EARLY_RESULT':early,'EARLY_SPREAD':early_spread,'INTRO':'\n\n'.join(intro),'KVALUE':f'{K:.10f}','FACTOR':f'{factor:.5f}','RESIDUAL':texnum(res),'AGREEMENT':texnum(agree)}.items():body=body.replace(a,b)
if len(wide)<27:
    body=body.replace('の27点。元テンソル','の27点を対象とし、現在'+str(len(wide))+'点を保存済みです。元テンソル')
body=body.replace('\n\n\n\n','\n\n')
growth=H/'growth_section.md'
if growth.exists():
    assert json.loads((H/'growth_fits.json').read_text())['input_sha256']==hashlib.sha256((H/'analysis.json').read_bytes()).hexdigest(),'Re-run fit_growth.py for the current analysis'
body=body.replace('GROWTH_SECTION',growth.read_text().strip() if growth.exists() else '')
target=D/'13_大きいtubeで電荷準位から二量体化を読む.md';target.write_text(body)
print(target)
