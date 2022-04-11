import pandas as pd
import datetime as dt
import backtrader as bt
import pyfolio as pf
import matplotlib as plt
import numpy as np
import os

os.getcwd()

#数据加载
df=pd.read_csv('CSI300.csv')
df.index=pd.to_datetime(df.date)
df['openinterest']=0


#随机生成一个-1，0，1的predictions序列
sig=[]
for i in range(len(stock_df)):
    x=np.random.randn()
    if x> 0.2:
        sig.append(1)
    elif x<-0.2:
        sig.append(-1)
    else:
        sig.append(0)
df['signal']=sig



#构建策略-------------------------------
class TestSignalStrategy(bt.Strategy):
    params=(
    )

    def log(self,txt):
        dt= self.datas[0].datetime.date(0)
        print('{},{}'.format(dt.isoformat(),txt)) 

    def __init__(self):
        self.bar_num=0
        self.signal_df=df
        self.signal_df['date']=pd.to_datetime(self.signal_df['date'])
        self.first_trade=True

    def prenext(self):
        self.next()

    
    def next(self):
        #假设有100万资金，每次成分股调整，每个股票使用1万元
        self.bar_num+=1
        current_date=self.datas[0].datetime.date(0).strftime("%Y-%m-%d")
        try:
            next_date =self.datas[0].datetime.date(1)
        except:
            next_date=None
        if next_date != None:
            next_signal_df= self.signal_df[self.signal_df['date']==pd.to_datetime(next_date)]
            if len(next_signal_df)==0:
                self.log("下个交易日的信号不存在")
            else:
                signal = int(next_signal_df['signal'])
                close_price = float(next_signal_df['close'])
                
                #第一次交易没有底仓，只允许做多
                #交易信号的执行方式：
                #signal为1，以开盘价买入45%的资金，然后以收盘价卖出昨日全部底仓
                #signal为-1时，以开盘价卖出45%的资金，以收盘价买入平仓。
                if self.first_trade:
                    if signal==-1:
                        pass
                    if signal == 1:
                        total_value = self.broker.get_value()
                        next_open_price = self.datas[0].open[1]
                        target_size=(0.01*0.45*total_value/next_open_price)*100
                        self.buy(self.datas[0],size=target_size)
                        self.first_trade=False
                else:
                    #现有持仓
                    now_hold_size = self.getposition(self.datas[0]).size
                    if signal == 1:
                        total_value = self.broker.get_value()
                        next_open_price=self.datas[0].open[1]
                        target_size=(0.01*0.45*total_value/next_open_price)*100
                        self.buy(self.datas[0],size=target_size)
                        self.sell(self.datas[0],size=now_hold_size,price=close_price)
                    if signal == -1:
                        total_value = self.broker.get_value()
                        next_open_price =self.datas[0].open[1]
                        target_size=(0.01*0.45*total_value/next_open_price)*100
                        #卖出开仓，手数不能超过底仓
                        if target_size>now_hold_size:
                            target_size=now_hold_size

                        self.sell(self.datas[0],size=target_size)
                        self.buy(self.datas[0],size=now_hold_size,price=close_price)
    #订单情况
    def notify_order(self, order):
        
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status == order.Rejected:
            self.log(f"Rejected : order_ref:{order.ref}  data_name:{order.p.data._name}")
            
        if order.status == order.Margin:
            self.log(f"Margin : order_ref:{order.ref}  data_name:{order.p.data._name}")
            
        if order.status == order.Cancelled:
            self.log(f"Concelled : order_ref:{order.ref}  data_name:{order.p.data._name}")
            
        if order.status == order.Partial:
            self.log(f"Partial : order_ref:{order.ref}  data_name:{order.p.data._name}")
         
        if order.status == order.Completed:
            if order.isbuy():
                self.log(f" BUY : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}")

            else:  # Sell
                self.log(f" SELL : data_name:{order.p.data._name} price : {order.executed.price} , cost : {order.executed.value} , commission : {order.executed.comm}")
    
    def notify_trade(self, trade):
        # 一个trade结束的时候输出信息
        if trade.isclosed:
            self.log('closed symbol is : {} , total_profit : {} , net_profit : {}' .format(
                            trade.getdataname(),trade.pnl, trade.pnlcomm))
            # self.trade_list.append([self.datas[0].datetime.date(0),trade.getdataname(),trade.pnl,trade.pnlcomm])
            
        if trade.isopen:
            self.log('open symbol is : {} , price : {} ' .format(
                            trade.getdataname(),trade.price))
    def stop(self):
        
        pass 

cerebro=bt.Cerebro()

stock_df=df[['open','high','low','close','volume','openinterest']]
data=bt.feeds.PandasData(dataname=stock_df,fromdate=dt.datetime(2010,1,4),todate=dt.datetime(2021,12,31),timeframe=bt.TimeFrame.Days)
cerebro.adddata(data)
cerebro.broker.setcash(1000000.0)
cerebro.broker.setcommission(commission=0.0002)

cerebro.addstrategy(TestSignalStrategy)
cerebro.addanalyzer(bt.analyzers.PyFolio)

#运行回测
results= cerebro.run()
cerebro.plot(style='candle')
