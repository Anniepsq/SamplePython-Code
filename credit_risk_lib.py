# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 08:12:03 2016

@author:MyPC
"""

import numpy as np
import pandas as pd



def get_ohlson_o_score(df_ohlson):
    '''
    Compute Ohlson-O score    
    '''
    #http://blog.alphaarchitect.com/2015/07/07/value-investing-research-o-score-and-distress-risk/#gs.KlsBSB8
    # X = 1 if TL > TA, 0 otherwise
    x = np.zeros(np.shape(df_ohlson.index))
    x[df_ohlson.total_lib - df_ohlson.current_asset > 0] = 1
    
    # Y = 1 if a net loss for the last two years, 0 otherwise
    y = np.zeros(np.shape(df_ohlson.index))
    y[df_ohlson.net_income.shift(periods=1) + df_ohlson.net_income.shift(periods=2) < 0] = 1
    y[0:2] = 0
     
    ohlson_o_score = -1.32 - 0.407 * np.log(df_ohlson.total_asset) \
    + 6.03 * df_ohlson.total_lib/df_ohlson.total_asset \
    - 1.43 * df_ohlson.working_cap/df_ohlson.total_asset \
    + 0.076 * df_ohlson.current_lib/df_ohlson.current_asset \
    - 1.72 * x \
    - 2.37 * df_ohlson.net_income/df_ohlson.total_asset \
    - 1.83 * df_ohlson.cash_flow_ops/df_ohlson.total_lib \
    + 0.285 * y \
    - 0.521 * (df_ohlson.net_income - df_ohlson.net_income.shift(1))/(np.abs(df_ohlson.net_income) + np.abs(df_ohlson.net_income.shift(1)))
    return ohlson_o_score

def get_altman_z_score(df_altman):
    '''
    Compute Altman's Z score  where 
    Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5
    Lower z-score suggests higher default probability
    '''    
    altman_z_score = 1.2 * df_altman["x1"] + 1.4 * df_altman["x2"] + \
        3.3 * df_altman["x3"] + 0.6 * df_altman["x4"] + 1.0 * df_altman["x5"]
    
    return altman_z_score;

def get_prob_default_chs(df_chs, predict_horizon=1):
    '''
    CHS logit default probability
    '''
    df_chs = df_chs.shift(predict_horizon)
    score = -9.16 - 20.26 * df_chs["NIMTAAVG"] + 1.42 * df_chs["TLMTA"] \
    - 7.13 * df_chs["EXRETAVG"] + 1.41 * df_chs["SIGMA"] - 0.045 * df_chs["RSIZE"] \
    -2.13 *df_chs["CASHMTA"] + 0.075 * df_chs["MB"] - 0.058 * df_chs["PRICE"]
    
    prob_default = 1/(1+np.exp(-score))
    return (score, prob_default)

def load_chs_data(fundamental_file_name = "Data/24823Q107_COMPUSTAT.csv", 
                  security_market_file_name="Data/24823Q107_CRSP.csv", index_market_file_name = "Data/SP_CRSP.csv"):
    #  Load fundamental data
    df_in=pd.read_csv(fundamental_file_name,skiprows=0,index_col=1,nrows=100)
    df_in.set_index(pd.to_datetime(df_in.index,format="%Y%m%d"),inplace=True)
    df_chs = pd.DataFrame(index=df_in.index)   
    net_income = df_in.niq
     
    total_lib = df_in.ltq
    mkvalt = df_in.mkvaltq
    market_valued_total_assets = mkvalt#df_in.atq # this may not be correct
    
    #NIMTAAVG = Net Income/ Market-valued total assets (profitability ratio)
    df_chs["NIMTAAVG"] = net_income/market_valued_total_assets
    #TLMTA = Total Liability/ Market-valued total assets (measure of leverage)
    df_chs["TLMTA"] = total_lib/market_valued_total_assets
    #CASHMTA: cash and short-term asset to Market-valued total assets (measure of liquidity)
    df_chs["CASHMTA"] = df_in["chq"]/market_valued_total_assets
    #MB: Market to book ratio = Market value of equity (MV)/ Book value of equity =MKVALT / (AT - LT)
    df_chs["MB"] = mkvalt/ (df_in.atq-df_in.ltq)
    
    # Load daily market data
    df_in_mkt = pd.read_csv(security_market_file_name,skiprows=0,index_col=1)
    df_in_mkt.set_index(pd.to_datetime(df_in_mkt.index,format="%Y%m%d"),inplace=True)
    
    df_in_mkt_index = pd.read_csv(index_market_file_name,skiprows=0,index_col=0)
    df_in_mkt_index.set_index(pd.to_datetime(df_in_mkt_index.index,format="%Y%m%d"),inplace=True)
    
    
    df_in_mkt["marketcap_daily"] = df_in_mkt.SHROUT * 1000 * df_in_mkt.PRC
    df_in_mkt["price_log_daily"] = np.log(df_in_mkt.PRC)
    df_in_mkt["ret_daily"] = df_in_mkt.PRC/df_in_mkt.PRC.shift(1)-1
    df_in_mkt["std_daily"] = pd.rolling_std(df_in_mkt.ret_daily, window=21*3) * (252/21/3)**0.5
    
    df_in_mkt_index["index_return_daily"] = df_in_mkt_index.sprtrn
    df_in_mkt_index["index_marketcap_daily"] = df_in_mkt_index.totval * 1000
    
    # merge and compute the daily measure 
    df_in_daily_merged = df_in_mkt.join(df_in_mkt_index, how='inner')
    #PRICE: Each firm’s log price per share, truncated at $15. How to winsorize?
    df_in_daily_merged["PRICE"] = df_in_daily_merged.price_log_daily
     #EXRETAVG = log excess return of equity to S&P index
    df_in_daily_merged["EXRETAVG"] = np.log(df_in_daily_merged.ret_daily+1) - np.log(df_in_daily_merged.index_return_daily+1) 
    #SIGMA: standard deviation of equity return over the past 3 month (using daily data)
    df_in_daily_merged["SIGMA"] = df_in_daily_merged.std_daily
    #RSIZE: relative size of each firm. log ratio of its market cap to that of S&P index
    df_in_daily_merged["RSIZE"] = np.log(df_in_daily_merged.marketcap_daily/df_in_daily_merged.index_marketcap_daily)
    # convert into quarterly measures
    df_chs = df_chs.join(df_in_daily_merged,how='inner')
    
    return df_chs
    
    
    
  

def load_ohlson_o_score_data_from_wrds(wrds_file_name = "136884_DNDNQ.csv"):
    df_in=pd.read_csv(wrds_file_name,skiprows=0,index_col=1,nrows=100)
    # df_altman.columns.values
    df_in.set_index(pd.to_datetime(df_in.index,format="%Y%m%d"),inplace=True)
    df_ohlson=pd.DataFrame(index=df_in.index)    
    df_ohlson["working_cap"] = df_in.wcapq
    df_ohlson["total_asset"] = df_in.atq
    df_ohlson["total_lib"] = df_in.ltq
       
    #current liabilities
    df_ohlson["current_lib"] = df_in.lctq
    #current assets
    df_ohlson["current_asset"] = df_in.actq
    #net income
    df_ohlson["net_income"] = df_in.niq
    #funds from operations/ cash flow from operations
    df_ohlson["cash_flow_ops"] = df_in.oancfy    
    
    return df_ohlson

    
def load_altman_data_from_wrds(wrds_file_name = "136884_DNDNQ.csv"):
    # https://en.wikipedia.org/wiki/Altman_Z-score
    df_in=pd.read_csv(wrds_file_name,skiprows=0,index_col=1,nrows=100)
    # df_altman.columns.values
    df_in.set_index(pd.to_datetime(df_in.index,format="%Y%m%d"),inplace=True)
    df_altman=pd.DataFrame(index=df_in.index)    
    df_altman["working_cap"] = df_in.wcapq
    df_altman["total_asset"] = df_in.atq
    df_altman["retained_earning"] = df_in.req
    df_altman["ebitda"] = df_in.oibdpq
    df_altman["mv_equity"] = df_in.seqq
    # bv_debt = DLCQ + DLTTQ
    df_altman["bv_debt"] = df_in.dlcq + df_in.dlttq
    df_altman["sales"] = df_in.saleq
    
    # X1 = Working Capital / Total Assets
    df_altman["x1"] = df_altman["working_cap"]/df_altman["total_asset"]
    # X2 = Retained Earnings / Total Assets
    df_altman["x2"] = df_altman["retained_earning"]/df_altman["total_asset"]
    # X3 = Earnings Before Interest and Taxes / Total Assets
    df_altman["x3"] = df_altman["ebitda"]/df_altman["total_asset"]
    # X4 = Market Value of Equity / Book Value of Total Liabilities
    df_altman["x4"] = df_altman["mv_equity"]/df_altman["bv_debt"]
    df_altman["x5"] = df_altman["sales"]/df_altman["total_asset"]
    
    return df_altman[["x1","x2","x3","x4","x5"]]
    