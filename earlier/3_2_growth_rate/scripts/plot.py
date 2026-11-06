from pathlib import Path
import csv, json, math, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
FIGURES=ROOT/'figures'
def plot(records,predictions,reps):
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'dejavuserif','font.size':9,'axes.linewidth':.7,'pdf.fonttype':42})
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

