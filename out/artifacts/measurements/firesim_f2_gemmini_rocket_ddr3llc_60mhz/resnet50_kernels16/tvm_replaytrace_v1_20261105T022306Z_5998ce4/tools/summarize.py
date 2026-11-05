import json
R="/work/exp/tvmfs/f2/results/"
rows=json.load(open(R+"confirm.json")); e2e=json.load(open(R+"e2e.json")); acc=json.load(open("/work/exp/tvmfs/f2/tuning-accounting.json"))
HZ=60e6; out=[]
for r in rows:
    b=[x["cycles"] for x in r["baseline"] if x["ok"]]; t=[x["cycles"] for x in r.get("tuned",[]) if x["ok"]]
    alts={a["plan"]:a["run"]["cycles"] for a in r["alternatives"] if a.get("run",{}).get("ok")}
    alt_fail=[a["plan"] for a in r["alternatives"] if a.get("run") and not a["run"]["ok"] or (not a["build"]["ok"])]
    cand={"default-untuned":b[0] if b else None,"tuned-default-plan":t[0] if t else None}
    cand.update({"tuned-"+k:v for k,v in alts.items()})
    cand={k:v for k,v in cand.items() if v}
    best=min(cand,key=cand.get)
    out.append(dict(unit=r["unit"],kind=r["kind"],baseline_cycles=b[0],baseline_repeats_equal=len(set(b))==1,tuned_cycles=t[0] if t else None,
        tuned_vs_baseline_pct=round(100*(t[0]/b[0]-1),2) if t else None,alt_plans_measured=alts,alt_plans_not_distinct_or_failed=alt_fail,
        best_confirmed=best,best_cycles=cand[best],best_vs_baseline_pct=round(100*(cand[best]/b[0]-1),2),best_seconds_at_60MHz=cand[best]/HZ))
json.dump(dict(units=out,e2e=e2e,clock_hz=HZ),open(R+"summary.json","w"),indent=1)
print("unit  baseline  tuned  tuned%  best(plan)  best%")
for o in out: print(o["unit"],o["baseline_cycles"],o["tuned_cycles"],o["tuned_vs_baseline_pct"],o["best_confirmed"],o["best_vs_baseline_pct"],len(o["alt_plans_measured"]))
b=e2e["baseline"]["cycles"];t=e2e["tuned"]["cycles"];print("E2E",b,t,round(100*(t/b-1),3))
