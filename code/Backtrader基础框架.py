#以sp500为标的，先写的框架，等算法全部弄完再把算法放进去
import pandas as pd
import datetime as dt
import backtrader as bt
import matplotlib as plt
import backtrader.analyzers as btanalyzers
import os

os.getcwd()
#数据加载
df=pd.read_csv(r'.\Backtrader学习\CSI300.csv')
df.index=pd.to_datetime(df.date)
df['openinterest']=0
#对df数据整合以符合backtrader
stock_df=df[['open','high','low','close','volume','openinterest']]
data=bt.feeds.PandasData(dataname=stock_df,fromdate=dt.datetime(2010,1,4),todate=dt.datetime(2021,12,31),timeframe=bt.TimeFrame.Days)

#构建策略-------------------------------
class MaCrossStrategy(bt.Strategy):
    params=(
        ('fast_length',5),
        ('slow_length',25)
    )

    def __init__(self):
        ma_fast=bt.ind.SMA(period=self.params.fast_length)
        ma_slow=bt.ind.SMA(period=self.params.slow_length)

        self.crossover=bt.ind.CrossOver(ma_fast,ma_slow)
    
    def next(self):
        if not self.position:
            if self.crossover>0:
                self.buy()
        elif self.crossover<0:
            self.close()

cerebro=bt.Cerebro()

cerebro.addstrategy(MaCrossStrategy)
cerebro.adddata(data)
cerebro.broker.setcash(100000.0)
cerebro.broker.setcommission(commission=0.0001)

#Analyzer----------------------
cerebro.addanalyzer(btanalyzers.SharpeRatio,_name='sharpe')
cerebro.addanalyzer(btanalyzers.DrawDown,_name='drawdown')
cerebro.addanalyzer(btanalyzers.Returns,_name='returns')

print('start portfolio value{}',format(cerebro.broker.getvalue()))
back=cerebro.run()
print('end portfolio value{}',format(cerebro.broker.getvalue()))



#提取夏普比率、回测、年化以评估策略
ratio_list=[[
    x.analyzers.returns.get_analysis()['rtot'],
    x.analyzers.drawdown.get_analysis()['max']['drawdown'],
    x.analyzers.sharpe.get_analysis()['sharperatio']] for x in back]


ratio_df=pd.DataFrame(ratio_list,columns=['fast_length','slow_length','Total_return','DrawDown','Sharpe_Ratio'])
print(ratio_df)

cerebro.plot(style='candle')

#策略调参时发现使用backtrader内部的opstrategy似乎不是很好，到时候换个方法调
