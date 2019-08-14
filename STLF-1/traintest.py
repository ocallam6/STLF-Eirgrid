"""
Prototype1

Neural Network Using Keras

Predicts next day values. 48 steps ahead forecast.

Uses daytype,temperature,holidays and utilises autocorrelation.



"""

def data_split(df):
	df_test=df.tail(8112+48)
	df.drop(df.tail(48).index,inplace=True)
	#df.drop(df.tail(8112+48).index,inplace=True)#should that just ve 48
	return df,df_test

def predict_data(df_train,date_end,date_predict,temp):
	date_rng=pd.date_range(start=date_end,end=date_predict,freq='30min')
	df=pd.DataFrame(date_rng,columns=['Date and Time'])
	df['Temp']=temp
	df.drop(df.tail(1).index,inplace=True)

	def season(month):
		return month // 4

	def lunchtime(hour):
		hours=[12,13,14,15]
		if(hour in hours):
			return 1
		else: return 0

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

	df['Lunch']=df['Date and Time'].dt.hour.apply(lunchtime)
	exclude_days=excludes('IE')
	df['Excluding']=df['Date and Time'].dt.date.apply(excluding)
	df['Days']=df['Date and Time'].dt.date.apply(daytype)
	#df['Temp']=df.Temp.apply(tempfn)
	df.drop(df.tail(1).index,inplace=True) #one extra row in df
	df['Time']=df['Date and Time'].dt.date
	df['Month']=df['Date and Time'].dt.month

	df['Season']=df['Date and Time'].dt.month.apply(season)


	maxs=df.groupby('Time').Temp.max()
	df.set_index(['Time'],inplace=True)
	df['Maxs']=maxs
	df.reset_index(inplace=True)
	df['Temp']=df.Maxs.apply(tempfn)
	df=df.fillna(0)

	df_train=df_train.append(df,ignore_index=True)
	df_predict=df_train.tail(8112+48)
	df_train.drop(df_train.tail(48).index,inplace=True)
	return df_train,df_predict
