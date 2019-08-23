"""
Prototype1

Neural Network Using Keras

Predicts next day values. 48 steps ahead forecast.

Uses daytype,temperature,holidays and utilises autocorrelation.

Data Prep Program
"""
from readfile import excludes
import pandas as pd 
import numpy as np
import os

location='IE' #='NI for northern ireland


def datas(date_begin,date_end):
	print('Loading Data... \n')
	try:
		def df_col_name(date_begin,date_end,var,loc):
			return var+loc+'_'+date_begin+'_'+date_end
		date_rng= pd.date_range(start=date_begin,end=date_end,freq='30min')
		df=pd.DataFrame(date_rng,columns=['Date and Time'])
		col_title=df_col_name(date_begin,date_end,'Load',location)
		df['Load']=pd.read_csv(r'%s.csv'%col_title)
		os.remove(r'%s.csv'%col_title)
		#col_title=df_col_name(date_begin,date_end,'Weather Item=6','') #for NI
		col_title=df_col_name(date_begin,date_end,'Weather Item=1','') #for IE
		df['Temp']=pd.read_csv(r'%s.csv'%col_title)
		os.remove(r'%s.csv'%col_title)
		col_title=df_col_name(date_begin,date_end,'Weather Item=2','') #wind but isn't used in model
		df['Wind']=pd.read_csv(r'%s.csv'%col_title)
		os.remove(r'%s.csv'%col_title)
		
		#data going into neural network needs to be normalised or mapped into [-1,1]
		
		def season(month):
			return month // 4
		
		def excluding(day):
			from datetime import date
			import datetime
			if(str(day) in exclude_days):
				return 1
			else:
				return 0
		def daytype(day):
			days=[1,2,3,4]
			day=day.weekday()
			if(day in days):
				return 1
			else:
				return 0
		def tempfn(temp): #potentially need to alter and add a weather item
			if(temp<=0):
				return 0
			if(temp>0 and temp <=7.5):
				return 0.2
			if(temp>7.5 and temp<=15):
				return 0.4
			if(temp>15 and temp<=22.75):
				return 0.6
			if(temp>22.75 and temp<30):
				return 0.8
			if(temp>=30):
				return 1
		def lunchtime(hour):
			hours=[12,13,14,15]
			if(hour in hours):
				return 1
			else: return 0

		def windfn(wind):
			return wind/10

		df['Lunch']=df['Date and Time'].dt.hour.apply(lunchtime)
		exclude_days=excludes(location)
		df['Excluding']=df['Date and Time'].dt.date.apply(excluding)
		df['Days']=df['Date and Time'].dt.date.apply(daytype)
		df.drop(df.tail(1).index,inplace=True) #one extra row in df
		df['Time']=df['Date and Time'].dt.date
		df['Month']=df['Date and Time'].dt.month
		df['Season']=df['Date and Time'].dt.month.apply(season)
		#--------------------------------------#
		maxs=df.groupby('Time').Temp.max()     #
		df.set_index(['Time'],inplace=True)    #
		df['Maxs']=maxs                        #
		df.reset_index(inplace=True)           #
		df['Temp']=df.Maxs.apply(tempfn)       #
		#--------------------------------------#
		#df['Temp']=df.Temp.apply(tempfn)     #uncomment and comment out above box if rather use just temperature, traintestwould need to be changed though
		df['Wind']=df.Wind.apply(windfn)
		print('Data loaded successfully.\n')

		return df	
	except:
		print('Error Loading Data.')
		print('Make sure load data file is up to date.')
		print('Make sure CSV files generated.')
		return

		
