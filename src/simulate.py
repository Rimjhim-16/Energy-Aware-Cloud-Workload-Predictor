import pandas as pd, numpy as np
from pathlib import Path
R=Path(__file__).resolve().parents[1]; O=R/'outputs'
def simulate(p,servers=10,capacity=10,margin=5,low_steps=6):
 p=p[p.horizon_hours==1].sort_values('timestamp').copy(); active=servers; low=0; out=[]
 for _,r in p.iterrows():
  demand=float(np.clip(r.predicted+margin,0,100)); required=max(1,int(np.ceil(demand/capacity)))
  low=low+1 if demand<active*capacity*.45 else 0
  if low>=low_steps and active>1: active-=1; low=0
  elif required>active: active=min(servers,required)
  available=active*capacity; unserved=max(0,float(r.actual)-available)
  base=servers*180/12/1000; watts=active*180+(servers-active)*15
  out.append({'timestamp':r.timestamp,'predicted':r.predicted,'actual':r.actual,'active_servers':active,'available_capacity':available,'unserved_demand':unserved,'baseline_kwh':base,'policy_kwh':watts/12/1000})
 return pd.DataFrame(out)
def main():
 p=pd.read_csv(O/'predictions.csv'); s=simulate(p); s.to_csv(O/'simulation.csv',index=False)
 b=s.baseline_kwh.sum(); q=s.policy_kwh.sum()
 print(f'Baseline: {b:.3f} kWh | Policy: {q:.3f} kWh | Estimated saving: {(1-q/b)*100:.2f}%')
if __name__=='__main__': main()
