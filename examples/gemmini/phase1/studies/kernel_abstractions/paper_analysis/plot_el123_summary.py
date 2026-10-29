"""Two-panel descriptive figure with the discovered comparator confound visible.

Consumes the pinned source-coherent result JSON. It preserves native verdicts;
it does not turn identical raw outputs into unearned full L2 passes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_el123_comparison import COLORS, LABELS, RUNS, STYLES, WINDOW_S


def render(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5,
        "axes.labelsize":7.5,"xtick.labelsize":7,"ytick.labelsize":7,
        "axes.spines.top":False,"axes.spines.right":False,
        "pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(1,2,figsize=(7.05,3.6),gridspec_kw={"width_ratios":[1.5,1]})
    fig.subplots_adjust(left=.075,right=.98,bottom=.31,top=.72,wspace=.3)
    for name in RUNS:
        r=result["runs"][name]
        curve=r["curve"]+[[WINDOW_S,r["curve"][-1][1]]]
        axes[0].step([p[0]/60 for p in curve],[p[1] for p in curve],where="post",
                     color=COLORS[name],ls=STYLES[name],lw=1.8)
    axes[0].set(xlim=(0,250),ylim=(0,149),xticks=[0,60,120,180,240],
        yticks=[0,32,64,96,128],xlabel="Time since author start (min)",
        ylabel="Cases with a recorded L2 pass")
    axes[0].annotate("135",(240,135),xytext=(-2,7),textcoords="offset points",
                     ha="right",color=COLORS["el3"],fontsize=9,fontweight="bold")
    axes[0].annotate("128 (EL1, EL2)",(240,128),xytext=(-2,-14),textcoords="offset points",
                     ha="right",color=COLORS["el1"],fontsize=7.3)
    axes[0].text(125,32,"Six chain verdicts depend on\nthe comparison path; the\nseventh baseline case was unrun.",
                 fontsize=7,color="#444444",linespacing=1.35)
    axes[0].set_title("(a) Recorded L2 verdicts\nChecker rules differ by ABI",loc="left",
                       fontsize=8.5,fontweight="bold")
    values=[result["runs"][n]["tokens"]["tokens_total"]/1e6 for n in RUNS]
    for i,(name,value) in enumerate(zip(RUNS,values)):
        axes[1].bar(i,value,width=.62,color=COLORS[name])
        axes[1].text(i,value+2,f"{value:.1f}",ha="center",va="bottom",fontsize=9,
                     color=COLORS[name],fontweight="bold")
    axes[1].set(xticks=[0,1,2],xticklabels=["EL1","EL2","EL3"],ylim=(0,145),
                ylabel="Recorded tokens including cache (M)")
    axes[1].set_title("(b) Token use\nFull four-hour window",loc="left",fontsize=8.5,fontweight="bold")
    for ax in axes:
        ax.grid(axis="y",color="#e5e5e5",lw=.45)
        ax.set_axisbelow(True)
    fig.text(.5,.963,"Four-hour campaigns: token use and reported validation verdicts",
             ha="center",va="center",fontsize=10,fontweight="bold")
    fig.legend(handles=[Line2D([],[],color=COLORS[n],ls=STYLES[n],lw=2,label=LABELS[n]) for n in RUNS],
               loc="upper center",bbox_to_anchor=(.5,.929),frameon=False,ncol=3,fontsize=7)
    fig.text(.5,.833,"DRAFT: six verdicts use different comparison rules; the seventh baseline case was unrun.",
             ha="center",fontsize=7,color="#8c4700",fontweight="bold")
    fig.text(.5,.187,"Same four-hour / 15-CPU allowance, references and 149-case roster; one author per condition.",
             ha="center",fontsize=6.9)
    fig.text(.5,.124,"Fresh input + output: 3.01M (EL1), 2.44M (EL2), 2.85M (EL3). Total tokens include cached context.",
             ha="center",fontsize=6.7,color="#555555")
    fig.text(.5,.061,"One source per coverage point. Apply common validation rules before comparing compiler correctness.",
             ha="center",fontsize=6.4,color="#555555")
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for artist in fig.findobj(match=Text):
        if artist.get_visible() and artist.get_text():
            b=artist.get_window_extent(renderer)
            if b.x0 < -1 or b.x1 > fig.bbox.width+1 or b.y0 < -1 or b.y1 > fig.bbox.height+1:
                raise ValueError(f"Figure text would be clipped: {artist.get_text()!r}")
    for ext in ("pdf","svg","png"):
        fig.savefig(out/f"el123_fixed_budget.{ext}",dpi=240,facecolor="white")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=json.loads(args.result.read_text())
    args.out.mkdir(parents=True,exist_ok=True)
    render(result,args.out)


if __name__=="__main__":
    main()
