import pandas as pd
from src.simulate import simulate
def test_simulation_output():
 p=pd.DataFrame({'timestamp':pd.date_range('2025-01-01',periods=20,freq='5min'),'actual':[30.]*20,'predicted':[25.]*20,'horizon_hours':[1]*20})
 s=simulate(p); assert len(s)==20 and (s.policy_kwh>=0).all() and 'unserved_demand' in s
