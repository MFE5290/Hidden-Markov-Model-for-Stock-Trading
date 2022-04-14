import pandas as pd
import datetime as dt
import backtrader as bt
import quantstats as qs
import matplotlib as plt
import numpy as np
import os

os.getcwd()

#数据加载-------------------------------------
df=pd.read_csv(r'C:\Users\31145\PycharmProjects\Backtrader学习\CSI300.csv')#绝对路径
df.index=pd.to_datetime(df.date)
df['openinterest']=0

#以下是自己随机添加signal的之后的文件，hmm算法给出的带signal的文件直接用上面的df=pd.read_csv导入，把下面到构建策略之前mute掉即可

#随机生成一个-1，0，1的predictions序列---------
sig=[]
for i in range(len(df)):
    x=np.random.randn()
    if x > 0.2:
        sig.append(1)
    elif x<-0.2:
        sig.append(-1)
    else:
        sig.append(0)
df['signal']=sig



#构建策略-------------------------------------
class TestSignalStrategy(bt.Strategy):
    params=()

    def log(self,txt,dt=None):
        dt= dt or self.datas[0].datetime.date(0)
        print('%s,%s' % (dt.isoformat(),txt)) 

    def __init__(self):
        self.bar_num=0
        self.signal_df=df
        self.signal_df['date']=pd.to_datetime(self.signal_df['date'])
        self.first_trade=True

    def prenext(self):
        self.next()

    def next(self):
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
                
                #第一次交易没有底仓，只允许做多
                #交易信号的执行方式：
                #先不考虑卖空
                #signal为1，以当天开盘价全仓买入
                #signal为-1时，以当天开盘价卖出所有持仓
                if self.first_trade:
                    if signal==-1:
                        pass
                    if signal == 1:
                        total_value = self.broker.get_value()
                        open_price = self.datas[0].open[0]
                        target_size=total_value//open_price
                        self.buy(self.datas[0],size=target_size,price=open_price)
                        self.first_trade=False
                else:
                    #现有持仓
                    now_hold_size = self.getposition(self.datas[0]).size
                    if signal == 1:
                        total_value = self.broker.get_value()
                        open_price = self.datas[0].open[0]
                        target_size=total_value//open_price
                        self.buy(self.datas[0],size=target_size, price=open_price)

                    if signal == -1:
                        total_value = self.broker.get_value()
                        open_price = self.datas[0].open[0]
                        target_size=total_value//open_price
                        #卖出开仓，手数不能超过底仓
                        if target_size>now_hold_size:
                            target_size=now_hold_size

                        self.sell(self.datas[0],size=target_size, price=open_price)

    #订单情况
    def notify_order(self, order):
        
        #订单处于未决，继续
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        #订单已决
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('买单执行,%.2f' % order.executed.price)

            elif order.issell():
                self.log('卖单执行,%2f' % order.executed.price)
        
        elif order.status in [order.Canceled,order.Rejected,order.Expired]:
            self.log('订单 Canceled/Rejected/Expired')

    #交易情况
    def notify_trade(self, trade):
        if trade.isclosed:
            print('毛收益 %0.2f, 扣佣后收益 % 0.2f , 佣金 %.2f'
            % (trade.pnl, trade.pnlcomm, trade.commission))
        


cerebro=bt.Cerebro()

stock_df=df[['open','high','low','close','volume','openinterest']]
data=bt.feeds.PandasData(dataname=stock_df,fromdate=dt.datetime(2010,1,4),todate=dt.datetime(2021,12,31),timeframe=bt.TimeFrame.Days)
cerebro.adddata(data)

#初始资金设置为10万
cerebro.broker.setcash(100000.0)
#手续费万分之二
cerebro.broker.setcommission(commission=0.0002)
#滑点设置 0.5%
cerebro.broker = bt.brokers.BackBroker(slip_perc=0.005)

cerebro.addstrategy(TestSignalStrategy)
cerebro.addanalyzer(bt.analyzers.PyFolio,_name='PyFolio')

#运行回测
print('start portfolio value:',format(cerebro.broker.getvalue()))
results = cerebro.run()
print('end portfolio value:',format(cerebro.broker.getvalue()))


#绩效评价----------------------------
#获取策略实例
start = results[0]
portfolio_stats = start.analyzers.getbyname('PyFolio')
#以下returns为以日期为索引的资产日收益率序列
returns,positions,transactions,gross_lev = portfolio_stats.get_pf_items()
returns.index = returns.index.tz_convert(None)
#画图仍只能在notebook中输出
qs.reports.basic(returns,benchmark=None,rf=0.0,grayscale=False,display=True,compounded = True)
