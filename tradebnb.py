#!/usr/bin/env python

"""
tradebnb.py - Binance trader
Created by Aaron Mekonnen on 12/23/17

"""

import time
import matplotlib.pyplot as plt
import requests
from binance.client import Client
from binance.enums import *
import hashlib
import hmac
from urllib.parse import urlencode



api_key = ""
api_secret =  ""

try:
	client = Client(api_key, api_secret)
except:
	raise


def change(x, y):
	return((y-x)/x)


def get_min(pool, i):
	
	btc_change = change(pool['BTC'][i-1], pool['BTC'][i]) 
	ltc_change = change(pool['LTC'][i-1], pool['LTC'][i]) 
	eth_change = change(pool['ETH'][i-1], pool['ETH'][i])
	xrp_change = change(pool['XRP'][i-1], pool['XRP'][i]) 


	min_change = min(btc_change, ltc_change, eth_change)
	if min_change == btc_change:
		return 'BTC'

	elif min_change == ltc_change:
		return 'LTC'

	elif min_change == eth_change:
		return 'ETH'
	else:
		return 'XRP'


def trade_best_option():
	BTC_candles = client.get_klines(symbol='BTCUSDT', interval=KLINE_INTERVAL_5MINUTE, limit=400)
	LTC_candles = client.get_klines(symbol='LTCUSDT', interval=KLINE_INTERVAL_5MINUTE, limit=400)
	ETH_candles = client.get_klines(symbol='ETHUSDT', interval=KLINE_INTERVAL_5MINUTE, limit=400)
	XRP_candles = client.get_klines(symbol='XRPBTC', interval=KLINE_INTERVAL_5MINUTE, limit=400)

	b_values = [float(BTC_candles[i][1]) for i in range(400)]
	l_values = [float(LTC_candles[i][1]) for i in range(400)]
	e_values = [float(ETH_candles[i][1]) for i in range(400)]
	x_values = [float(XRP_candles[i][1]) for i in range(400)]

	pool ={'BTC': b_values,
			'LTC': l_values,
			'ETH': e_values,
			'XRP': x_values
			} 

	#start with BTC 
	holding = 'BTC'
	hold_amt = 0.1
	min_change = 'BTC'
	to_dollar = 0
	num_of_changes = 0
	eth_amount = (hold_amt*(pool['BTC'][0]))/(pool['ETH'][0])
	ltc_amount = (hold_amt*(pool['BTC'][0]))/(pool['LTC'][0])
	xrp_amount = (hold_amt/(pool['XRP'][0]))

	print('starting with', hold_amt, 'BTC', eth_amount, 'ETH', ltc_amount, 'LTC', xrp_amount, 'XRP')
	for i in range(1, 400):
		if i > 9:
			if ma(pool[holding], i, 10) > pool[holding][i]:
				min_change = get_min(pool, i)
				#also buy fastest growing 
				if holding != min_change:
					to_dollar = (hold_amt*(pool[holding][i]))
					hold_amt = to_dollar/(pool[min_change][i])
					holding = min_change
					num_of_changes += 1
					print(i)

	print(hold_amt, holding)
	print((pool[holding][399]-pool[holding][0])/pool[holding][0])
	print('number of changes ', num_of_changes)


	plt.plot([float(i[1])/60 for i in BTC_candles], label = 'BTC')
	plt.plot([i[1] for i in LTC_candles], label = 'LTC')
	plt.plot([float(i[1])/2  for i in ETH_candles], label = 'ETH')
	plt.legend()
	plt.show()

def buy(balance, rate, delta):

	return [(balance[0]*(1-delta)), balance[1]+((balance[0]*delta)/rate)]

def sell(balance, rate, delta):

	return [balance[0]+((delta*balance[1])*rate), ((1-delta)*balance[1])]





def compare_BTC(coin, interval, limit):

	tradepair = coin + 'BTC'
	interval_dict = {1: KLINE_INTERVAL_1MINUTE,
					5: KLINE_INTERVAL_5MINUTE,
					15: KLINE_INTERVAL_15MINUTE,
					30: KLINE_INTERVAL_30MINUTE }

	alt_candles = client.get_klines(symbol=tradepair, interval=interval_dict[interval], limit=limit)
	btc_candles = client.get_klines(symbol='BTCUSDT', interval=interval_dict[interval], limit=limit)

	altcoin = [float(alt_candles[i][1])*float(btc_candles[i][1]) for i in range(500)]
	bitcoin = [float(i[1]) for i in btc_candles]
	fig, ax1 = plt.subplots()
	ax1.plot(altcoin, 'b')
	ax1.set_xlabel('time')
	ax1.set_ylabel(coin, color='b')
	ax1.tick_params('y', colors='b')

	ax2 = ax1.twinx()
	ax2.plot(bitcoin, 'r')
	ax2.set_ylabel('BTC', color='r')
	ax2.tick_params('y', colors='r')
	fig.tight_layout()
	plt.show()


	'''
		for i in range(500):
			if i > 4:
				m_ave = ma(x_values, i, 5)
				if  m_ave > x_values[i]*1.05:
					print('sell at', i, x_values[i], x_values[i-4], m_ave)

				# if  m_ave < x_values[i]*0.95:
				# 	print('buy at', i, x_values[i], x_values[i-4], m_ave)




	plt.plot(x_values, label = 'IOTA')
	plt.plot(xcandles, label = 'BTC')
	plt.legend()
	plt.show()
	'''


def trade_change(coin, interval, limit, init_amt):

	crypto = 0
	plus = 0
	minus = 0
	interval_dict = {1: KLINE_INTERVAL_1MINUTE,
					5: KLINE_INTERVAL_5MINUTE,
					15: KLINE_INTERVAL_15MINUTE,
					30: KLINE_INTERVAL_30MINUTE,
					60: KLINE_INTERVAL_1HOUR,
					2: KLINE_INTERVAL_2HOUR,
					4: KLINE_INTERVAL_4HOUR,
					6: KLINE_INTERVAL_6HOUR,
					'1': KLINE_INTERVAL_1DAY,
						}

	if coin == 'BTC':
		tradepair = 'BTCUSDT'
		initial = init_amt

	else:
		tradepair = coin + 'BTC'
		btc_candles = client.get_klines(symbol='BTCUSDT', interval=interval_dict[interval], limit=limit)
		btc_values = [float(i[1]) for i in btc_candles]

		initial = init_amt/btc_values[0]


	candles = client.get_klines(symbol=tradepair, interval=interval_dict[interval], limit=limit)
	x_values = [float(i[1]) for i in candles]

	balance = buy([initial, crypto], x_values[0], 0.5)

	print('initial balance:', str(2*balance[0])+'BTC')

	for i in range(1, limit):
		#buy/sell +- based on delta and moveing average method
		'''
		delta = change(x_values[i], x_values[i-1])
		m_ave = ma(x_values, i, 10)
		if delta > 0 and x_values[i] > 1.1*m_ave and m_ave != 0:
			balance = sell(balance, x_values[i], 50*delta)
			print(i, delta)
		elif delta < 0:
			balance = buy(balance, x_values[i], abs(delta))
		'''
		delta = change(x_values[i], x_values[i-1])






	print('final balance:', str(balance[0] + balance[1]*(x_values[limit-1])) + 'BTC')
	# print('dollar amount: $', (balance[0]+ (balance[1]*x_values[limit-1])))
	print('if held:', (str(((1000/btc_values[0])/x_values[0])*x_values[limit-1])) + 'BTC')

	plt.plot(x_values, label = 'BTC')
	plt.legend()
	plt.show()

# def sma_method():
def ma(l, i, n):
	if i < n-1:
		return 0
	else:
		return (sum(l[(i-n+1):(i+1)])/n)


def ema(l, i, n, prev):
	mult = (2/(n+1))
	return (((l[i] - prev) * mult) + prev)


def slope(l, i, n):
	return ((l[i-n]-l[i])/n)


def ema_method(coin, interval, limit, init_amt, points):

	tradepair = coin + 'BTC'
	trades = 0
	interval_dict = {1: KLINE_INTERVAL_1MINUTE,
					5: KLINE_INTERVAL_5MINUTE,
					3: KLINE_INTERVAL_3MINUTE,
					15: KLINE_INTERVAL_15MINUTE,
					30: KLINE_INTERVAL_30MINUTE,
					60: KLINE_INTERVAL_1HOUR,
					2: KLINE_INTERVAL_2HOUR,
					4: KLINE_INTERVAL_4HOUR,
					6: KLINE_INTERVAL_6HOUR,
					'1': KLINE_INTERVAL_1DAY,
						}


	candles = client.get_klines(symbol=tradepair, interval=interval_dict[interval], limit=limit)
	x_values = [float(i[4]) for i in candles]
	curr_state = ""
	prev_state = ""
	cross = 0
	balance = buy([init_amt, 0], x_values[0], 0.5)

	print('initial balance:',str(balance[1])+coin , str(balance[0])+'BTC')
	prev = [ma(x_values, points[2]-1, points[0]), ma(x_values, points[2]-1, points[1]),
				 ma(x_values, points[2]-1, points[2])] 
	ema_vals = [ema(x_values, points[2]-1, points[0], prev[0]), ema(x_values, points[2]-1, points[1], prev[1]),
				 ema(x_values, points[2]-1, points[2], prev[2])]
	print(prev, ema_vals)

	if max(ema_vals[0], ema_vals[1], ema_vals[2]) == ema_vals[0]:
		curr_state = "above"

	elif min(ema_vals[0], ema_vals[1], ema_vals[2]) == ema_vals[0]:
		curr_state = "below"

	prev_state = curr_state

	on = True 
	ema1 = [ema_vals[0]]
	ema2 = [ema_vals[1]]
	ema3 = [ema_vals[2]]

	for i in range(points[2], limit):
		ema_vals = [ema(x_values, i, points[0], prev[0]), ema(x_values, i, points[1], prev[1]),
				 ema(x_values, i, points[2], prev[2])] 

		if max(ema_vals[0], ema_vals[1], ema_vals[2]) == ema_vals[0]:
			curr_state = "above"

		else:
			curr_state = "below"

		if curr_state != prev_state:
			if prev_state == "above":
				if balance[1] > 0 and change(x_values[0], x_values[i]) > 0.05 :
					balance = sell(balance, x_values[i], 1)
					print('Sell at', [i, x_values[i]], "balance: ", balance)

			elif change(x_values[0], x_values[i]) < -0.05 and balance[0] > 0:
				balance = buy(balance, x_values[i], 1)
				print('Buy at', [i, x_values[i]],  "balance: ", balance)

			trades +=1

		prev_state = curr_state


		ema1.append(ema_vals[0])
		ema2.append(ema_vals[1])
		ema3.append(ema_vals[2])

		prev = ema_vals

		#buy/sell +- based on delta and moveing average method
		'''
		delta = change(x_values[i], x_values[i-1])
		m_ave = ma(x_values, i, 10)
		if delta > 0 and x_values[i] > 1.1*m_ave and m_ave != 0:
			balance = sell(balance, x_values[i], 50*delta)
			print(i, delta)
		elif delta < 0:
			balance = buy(balance, x_values[i], abs(delta))
		'''
	growth = change(x_values[0], x_values[limit-1])
	print((1-(0.001*trades))*(sell(balance,x_values[limit-1], 1)[0]), str(growth*100) + '%', init_amt+(init_amt*growth))
	print(trades)

	plt.plot(x_values, label = coin)
	plt.plot(range(points[2], limit+1), ema1, label= str(points[0]))
	# plt.plot(range(55, 500), ema2, label= str(points[1]))
	plt.plot(range(points[2], limit+1), ema3, label= str(points[2]))

	plt.legend()
	plt.show()


def begin_trade(coin, interval, points):
	tradepair = coin + 'BTC'

	interval_dict = {1: KLINE_INTERVAL_1MINUTE,
					5: KLINE_INTERVAL_5MINUTE,
					3: KLINE_INTERVAL_3MINUTE,
					15: KLINE_INTERVAL_15MINUTE,
					30: KLINE_INTERVAL_30MINUTE,
					60: KLINE_INTERVAL_1HOUR,
					2: KLINE_INTERVAL_2HOUR,
					4: KLINE_INTERVAL_4HOUR,
					6: KLINE_INTERVAL_6HOUR,
					'1': KLINE_INTERVAL_1DAY,
						}

	curr_state = ""
	prev_state = ""

	while True:	
		try:
			candles = client.get_klines(symbol=tradepair, interval=interval_dict[interval], limit=points[2]+1)

		except:
			pass

		x_values = [float(i[4]) for i in candles]

		prev = [ma(x_values, points[2]-1, points[0]), ma(x_values, points[2]-1, points[1]),
					 ma(x_values, points[2]-1, points[2])] 
		ema_vals = [ema(x_values, points[2]-1, points[0], prev[0]), ema(x_values, points[2]-1, points[1], prev[1]),
					 ema(x_values, points[2]-1, points[2], prev[2])]

		if max(ema_vals[0], ema_vals[1], ema_vals[2]) == ema_vals[0]:
			curr_state = "above"

		else:
			curr_state = "below"

		prev_state = curr_state

		ema_vals = [ema(x_values, points[2], points[0], ema_vals[0]), ema(x_values, points[2], points[1], ema_vals[1]),
				 ema(x_values, points[2], points[2], ema_vals[2])] 

		if max(ema_vals[0], ema_vals[1], ema_vals[2]) == ema_vals[0]:
			curr_state = "above"

		else:
			curr_state = "below"

		if curr_state != prev_state:
			if prev_state == "above":
				order = client.order_market_sell(
				symbol=tradepair,
				quantity=300)

			else:
				order = client.order_market_buy(
				symbol= tradepair,
				quantity=300)

			print(order)

		else:
			print("HOLD")

		print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')

		time.sleep(interval*60-1)

def streak_method(coin, interval, limit, init_amt, n, stk):

	tradepair = coin + 'BTC'
	trades = 0
	interval_dict = {1: KLINE_INTERVAL_1MINUTE,
					5: KLINE_INTERVAL_5MINUTE,
					3: KLINE_INTERVAL_3MINUTE,
					15: KLINE_INTERVAL_15MINUTE,
					30: KLINE_INTERVAL_30MINUTE,
					60: KLINE_INTERVAL_1HOUR,
					2: KLINE_INTERVAL_2HOUR,
					4: KLINE_INTERVAL_4HOUR,
					6: KLINE_INTERVAL_6HOUR,
					'1': KLINE_INTERVAL_1DAY,
						}


	candles = client.get_klines(symbol=tradepair, interval=interval_dict[interval], limit=limit)
	x_values = [float(i[4]) for i in candles]
	curr_state = ""
	prev_state = ""
	cross = 0
	balance = buy([init_amt, 0], x_values[0], 0.5)

	active = 'hold'
	streak = 0

	print('initial balance:',str(balance[1])+coin , str(balance[0])+'BTC')
	prev = ma(x_values, n-1, n)
	ema_vals = ema(x_values, n-1, n, prev)

	print(prev, ema_vals)

	if x_values[n-1] > ema_vals: 
		curr_state = "above"

	else:
		curr_state = "below"

	prev_state = curr_state

	ema1 = [ema_vals]


	for i in range(n, limit):
		ema_vals = ema(x_values, i, n, prev) 

		print(i, streak)

		curr_state = "above" if ema_vals < x_values[i] else "below"

		if x_values[i] > x_values[i-1]:
			streak += 1

		else:
			streak -= 1

		if streak >= stk :
			active = 'sell'
			balance = sell(balance, x_values[i], 1)
			print('Sell at', [streak, i, x_values[i]], "balance: ", balance)
			trades +=1
			streak = 0
			active = 'hold'

		elif streak <= -1*(stk):
			active = 'buy'
			balance = buy(balance, x_values[i], 1)
			print('Buy at', [streak, i, x_values[i]],  "balance: ", balance)

			trades +=1
			streak = 0
			active = 'hold'


		else:
			active = 'hold'


		'''
		if curr_state != prev_state:
			if prev_state == "above" and active == 'sell':
				balance = sell(balance, x_values[i], 1)
				print('Sell at', [streak, i, x_values[i]], "balance: ", balance)
				trades +=1
				streak = 0
				active = 'hold'

			elif active == 'buy':
				balance = buy(balance, x_values[i], 1)
				print('Buy at', [streak, i, x_values[i]],  "balance: ", balance)

				trades +=1
				streak = 0
				active = 'hold'
		'''
		prev_state = curr_state


		ema1.append(ema_vals)

		prev = ema_vals

	growth = change(x_values[0], x_values[limit-1])
	print((1-(0.001*trades))*(sell(balance,x_values[limit-1], 1)[0]), str(growth*100) + '%', init_amt+(init_amt*growth))
	print(trades)

	plt.plot(x_values, label = coin)
	plt.plot(range(n, limit+1), ema1, label= 'ema_vals')

	plt.legend()
	plt.show()

def find_new_coin():
	prices = client.get_all_tickers()
	symbols = set([i['symbol'] for i in prices])
	prev = set([i['symbol'] for i in prices])
	prev.add('VIBEUSDT')
	print(prev.difference(symbols))
	# while True:
	# 	try:
	# 		'VIBEBTC' in symbols
	# 		symbols.remove("VIBEBTC")
	# 	except ValueError:
	# 		print('True') 

	# 	time.sleep(5)



if __name__ == '__main__':

	# i = 0
	# while True:
	# 	BTC_candles = client.get_klines(symbol='BTCUSDT', interval=KLINE_INTERVAL_5MINUTE, limit=400)
	# 	print(BTC_candles, i)
	# 	time.sleep(3)
	# 	i+=1
	print(client.get_klines(symbol='NANOETH', interval=KLINE_INTERVAL_1MINUTE, limit=60))
	#find_new_coin()
	#begin_trade('XRP', 5, [13, 34, 34])
	#ema_method('RLC', 1 , 90, 0.1, [3, 13, 13])
	#streak_method('XRP', 5 , 480, 0.1, 3, 10)
	#trade_change('TRX', 15 , 500, 1000)
	#[time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(i[0]/1000))
	#trade_best_option()
	#compare_BTC('XVG', 5, 500)
	# print(client.get_historical_trades(symbol='BTCUSDT', limit=1))
	# print([i['price'] for i in client.get_all_tickers() if i['symbol'] == 'BTCUSDT'])
	'''
	order = client.order_market_sell(
    symbol='BTCUSDT',
    quantity=0.001)
	order = client.order_market_buy(
    symbol='BTCUSDT',
    quantity=0.001)
    #Fibonacci series: 1,1,2,3,5,8,13,21,34,55,89….

'''


