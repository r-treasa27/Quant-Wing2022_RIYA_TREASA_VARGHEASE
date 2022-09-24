#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Black-Scholes Model
#Q1


# In[2]:


from math import log,sqrt,exp
from scipy.stats import norm


# In[3]:


def d1(S,K,sigma,r,T,delta=0):
    return (((log(S/K))+(r-delta+(0.5*sigma*sigma)*T))/(sigma*sqrt(T)))


# In[4]:


def d2(S,K,sigma,r,T,delta=0):
    return (d1(S,K,sigma,r,T,delta=0)-(sigma*sqrt(T)))


# In[5]:


#Call option price


# In[6]:


def C(S,K,sigma,r,T,delta=0):
    return ((S*exp(-(delta*T))*norm.cdf(d1(S,K,sigma,r,T,delta=0)))-(K*exp(-(r*T))*norm.cdf(d2(S,K,sigma,r,T,delta=0))))


# In[7]:


#Put Option Price


# In[8]:


def P(S,K,sigma,r,T,delta=0):
    return((K*exp(-(r*T))*norm.cdf(-d2(S,K,sigma,r,T,delta=0)))-(S*exp(-(delta*T))*norm.cdf(-d1(S,K,sigma,r,T,delta=0))))


# In[9]:


C(60,50,12,5,0.5)


# In[10]:


P(60,50,12,5,0.5)


# In[11]:


#Q2(Mean Reversion Strategy using RSI and Bollinger Bands)(Have used reference material for this section!)


# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


# In[18]:


d = yf.Ticker("Marksans.NS")
df = d.history(period = "150d", interval = "1d")


# In[19]:


def BollingerBands (df0, period, dev):
    df = df0
    sma = df["Close"].rolling(window = period).mean()
    rstd = df["Close"].rolling(window = period).std()
    df["Standard Moving Average"] = sma
    df["std"] = rstd
    df["Upper Band"] = sma + dev * rstd
    df["Lower Band"] = sma - dev * rstd
    
    return df


# In[20]:


def RSI(df0, period):
    df = df0
    df["delta"] = df["Close"].diff()
    
    up, down = df["delta"].copy(), df["delta"].copy()
    up[up<0] = 0
    down[down>0] = 0
    df["up"] = up
    df["down"] = down
    
    
    avgain_exp = df["up"].ewm(span=period).mean()
    avloss_exp = df["down"].abs().ewm(span=period).mean()
    
    
    rs_exp = avgain_exp/avloss_exp
    
    rsi_exp = 100.0 - (100.0/(1.0 + rs_exp))
    
    df["EMA RSI"] = rsi_exp
    
    return df


# In[21]:


BollingerBands (df, 10, 1.7)


# In[22]:


RSI(df, 27)


# In[23]:


#Rules
#1. Buy when 10-period RSI below 30 (buy next day) & Price below lower bollinger band
#2. Sell when 10-period RSI above 70 (sell next day) & Price above upper bollinger band


# In[24]:


#buy signal
df['signal'] = np.where(
    (df['EMA RSI'] < 30) &
    (df['Close'] < df["Lower Band"]), 1, np.nan)

#sell signal
df['signal'] = np.where(
    (df["EMA RSI"] > 70) & 
    (df['Close'] > df["Upper Band"]), -1, df['signal'])

#buy/sell next trading day
df['signal'] = df['signal'].shift()
df['signal'] = df['signal'].fillna(0)


# In[25]:


df['Date'] = df.index
df


# In[26]:


def backtest_dataframe(df):
    position = 0
    percentage_change = []
    df['buy_date'] = ''
    df['sell_date'] = ''
    
    for i in df.index:
        close = df['Close'][i]
        date = df['Date'][i]

        #Buy Action

        if df['signal'][i]==1:
            if (position==0):
                buy_price = close
                position = 1
                df.at[i, 'buy_date'] = date
                print("Buying at",buy_price," on ",date)

        #Sell Action

        elif (df['signal'][i]==-1):
            if (position==0):
                sell_price = close
                position = 0
                df.at[i, 'sell_date'] = date
                print("Selling at",sell_price,"on",date)


# In[27]:


backtest_dataframe(df)


# In[28]:


#Pairs Trading Strategy
#Q3


# In[33]:


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib


# In[34]:


T1 = yf.Ticker("PFE")
T2 = yf.Ticker("AAPL")
T3 = yf.Ticker("WBA")
T4 = yf.Ticker("ATVI")
T5 = yf.Ticker("GOOGL")
T6 = yf.Ticker("FCAU")
T7 = yf.Ticker("ARI")
T8 = yf.Ticker("BHARTIARTL.NS")
T9 = yf.Ticker("ATA.PA")
T10 = yf.Ticker("KEL.DE")

list_of_tickers=[T1,T2,T3,T4,T5,T6,T7,T8,T9,T10]
#Taking a stationary level of 0.05
stationary_pairs= []


# In[35]:


def pairs_trade(ticker1,ticker2):
    hist1 = T1.history(period = "100d", interval = "1d")['Close']
    hist2 = T2.history(period = "100d", interval = "1d")['Close']
    hist_df_1 = hist1.to_frame()
    hist_df_2 = hist2.to_frame()
    model = sm.OLS(hist_df_2,hist_df_1).fit()
    influence = model.get_influence()
    standardized_residuals = influence.resid_studentized_internal
    result = adfuller(standardized_residuals, autolag='AIC')
    if result[1]<0.05:
        return True
    else:
        return False
    


# In[39]:


i = 0
while (i<len(list_of_tickers)):
    j = 0
    while (j<len(list_of_tickers)):
        if (i!=j):
            if pairs_trade(list_of_tickers[i],list_of_tickers[j])==True:
                stationary_pairs.append((list_of_tickers[i],list_of_tickers[j]))
        j+=1
    i+=1
           


# In[40]:


print(stationary_pairs)


# In[ ]:




