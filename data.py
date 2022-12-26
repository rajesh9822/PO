import numpy as np
import pandas as pd

import seaborn as sns
import pylab

def load_data():
    df = pd.read_csv("data.csv")
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.value_counts())
    return df

def do_eda(df):
    df = df[['Date','Name', 'Close']]

    low_beta_stock = ['SJM', 'MRK', 'EXR', 'WMT', 'LLY', 'CLX', 'DLR', 'KR'] #,'REGN'] 'PSA',,
    high_beta_stock = ['MS', 'NVDIA', 'C',  'GM', 'MET',  'COF', 'SLB','BAC'] #,] #,'TSLA'] BA,

    # low beta
    lbs = df[df['Name'].isin(low_beta_stock)]
    # high beta
    hbs = df[df['Name'].isin(high_beta_stock)]

    lbs_pivot = lbs.pivot("Date", "Name", "Close" )
    hbs_pivot = hbs.pivot("Date", "Name", "Close")
    #plt.interactive(False)
    lbs.head()
    hbs.head()
    sns.lineplot(data=lbs_pivot, dashes=False, ).set(title='Low Beta Stock').savefig("./save_graph/LowBetaStocks.png")
    sns.lineplot(data=hbs_pivot, dashes=False, ).set(title='High Beta Stock').savefig("./save_graph/HighBetaStocks.png")
    #snsfig2 = sns.lineplot(data=hbs_pivot, dashes=False).set(title='High Beta Stock')
    #snsfig2.savefig("./save_graph/HighBetaStocks.png")

def transform_data(data):
    df_lb = data.loc[data['Name'] == 'SJM']
    df_hb = data.loc[data['Name'] == 'MS']

    df = df_lb.set_index('Date').join(df_hb.set_index('Date'), lsuffix='_lb_stock', rsuffix='_hb_stock')
    df = df [['Name_lb_stock', 'Close_lb_stock','Name_hb_stock','Close_hb_stock']]
    #print(df.columns)
    pd.set_option('display.max_columns', None)
    #print(df.head())
    return df

