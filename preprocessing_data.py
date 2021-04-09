import numpy as np
import pandas as pd

etf_symbols = ['XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']
iv_filenames = ['./data/'+symbol+'_IV.csv' for symbol in etf_symbols]
pr_filenames = ['./data/'+symbol+'_Price.csv' for symbol in etf_symbols]

df = pd.DataFrame()
for symbol, iv_file, pr_file in zip(etf_symbols, iv_filenames, pr_filenames):
    iv_df = pd.read_csv(iv_file)
    pr_df = pd.read_csv(pr_file)
    if 'date' not in df.columns:
        df['date'] = pr_df['date']
    df[symbol+'_iv'] = iv_df['implied_vol']
    df[symbol+'_pr'] = pr_df['price']

# convert date
df.date = pd.to_datetime(df.date)
df.to_parquet('./data/sector_etf_iv_pr.parquet')
#print(pd.read_parquet('./data/sector_etf_iv_pr.parquet'))

num_columns = [col for col in df.columns if col != 'date']
df2 = df[num_columns].apply(lambda x : np.log(x))
df2 = df2 - df2.shift(periods=1)
df2['date'] = df.date
df2.dropna(inplace=True)

df2.to_parquet('./data/sector_etf_log_returns.parquet')
print(pd.read_parquet('./data/sector_etf_log_returns.parquet'))