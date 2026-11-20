import numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'data'/'workload_5min.csv'
rng=np.random.default_rng(42); n=4032
t=np.arange(n); ts=pd.date_range('2025-01-01',periods=n,freq='5min')
cpu=np.clip(42+17*np.sin(2*np.pi*(t%288)/288-1)+7*np.sin(2*np.pi*t/(288*7))+rng.normal(0,4,n),4,96)
for s in rng.integers(0,n-20,40): cpu[s:s+12]+=rng.uniform(5,18)
cpu=np.clip(cpu,0,100)
df=pd.DataFrame({'timestamp':ts,'cpu_utilization':cpu,'memory_utilization':np.clip(45+.4*(cpu-40)+rng.normal(0,5,n),5,98),'network_utilization':np.clip(25+.6*cpu+rng.normal(0,7,n),0,100)})
df.to_csv(out,index=False); print('Created demo dataset:',out,'rows:',len(df))
