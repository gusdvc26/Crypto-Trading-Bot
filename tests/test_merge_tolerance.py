import pandas as pd, numpy as np


def test_merge_tolerance_ms():
    dec = pd.DataFrame({'symbol':['BTC-USD']*3, 'ts_ms':[1000,2000,3000]})
    lab = pd.DataFrame({'symbol':['BTC-USD']*3, 'decision_ts_ms':[1010,1995,3030], 'mask_5m':[True,True,True]})
    dec['ts_ms_dt'] = pd.to_datetime(dec['ts_ms'], unit='ms', utc=True)
    lab['decision_ts_ms_dt'] = pd.to_datetime(lab['decision_ts_ms'], unit='ms', utc=True)
    merged = pd.merge_asof(dec.sort_values(['symbol','ts_ms_dt']),
                           lab.sort_values(['symbol','decision_ts_ms_dt']),
                           by='symbol', left_on='ts_ms_dt', right_on='decision_ts_ms_dt',
                           direction='nearest', tolerance=pd.Timedelta(milliseconds=50))
    assert merged['mask_5m'].notna().all()

