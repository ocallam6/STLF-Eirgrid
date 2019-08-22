"""
Prototype1

Neural Network Using Keras

Predicts next day values. 48 steps ahead forecast.

Uses daytype,temperature,holidays and utilises autocorrelation.

Neural Network

"""


def model_build(df_train,epochs,batch_size,prediction_step):
	#####################################################################################################
	if(epochs==0):
		print('Weights will be loaded from file.')
	import keras
	from sklearn.preprocessing import MinMaxScaler
	'''
	GRU sometimes has benefits over LSTM for a RNN
	Dense is a standard feedforward layer
	Dropout not used but can be implemented, sometimes makes better
	'''
	from keras.layers import GRU, Dense, Input, Flatten, Dropout, Conv1D,LSTM
	from keras.models import Model
	from keras.utils import plot_model
	from keras.callbacks import ModelCheckpoint
	import pandas as pd
	import numpy as np 
	import os	
	from keras.layers.merge import concatenate
	os.environ['TF_CPP_MIN_LOG_LEVEL']='3' #comment out for tf warnings
	'''
	Some of the tensorflow warnings may indicate that if you build tf from source you will get
	a more optimal environment for machine learning so may be work uncommenting every time 
	the code gets run on a new machine
	Moreover if GPU is available then must build tensorflow-GPU
	'''
	#####################################################################################################
	#Preparing Data
	#####################################################################################################
	print('-------------------------------------------------')
	print('Preparing data for training...\n')
	try:
		scaler=MinMaxScaler(feature_range=(-1,1))
		train=pd.DataFrame(df_train['Load'].values)
		train=scaler.fit_transform(train)
		days=df_train['Days'].values
		temp=df_train['Temp'].values
		wind=df_train['Wind'].values
		lunch=df_train['Lunch'].values
		exclude=df_train['Excluding'].values
		season=df_train['Season'].values
		

		inputs=[]
		target=[]
		daytype=[]
		excludes=[]
		temps=[]
		lunchs=[]
		seasons=[]
		lhour=[]
		winds=[]
		loadmonth1=[]
		loadmonth2=[]
		loadmonth3=[]
		loadmonth4=[]
		loadmonth5=[]
		loadmonth6=[]
		loadweek1=[]
		loadweek2=[]
		loadweek3=[]
		loadweek4=[]
		tempmonth1=[]
		tempmonth2=[]
		tempmonth3=[]
		tempmonth4=[]
		tempmonth5=[]
		tempmonth6=[]
		tempweek1=[]
		tempweek2=[]
		tempweek3=[]
		tempweek4=[]
		loadday1=[]
		loadday2=[]
		loadday3=[]
		loadday4=[]
		loadday5=[]
		loadday6=[]
		loadday7=[]
		tempday1=[]
		tempday2=[]
		tempday3=[]
		tempday4=[]
		tempday5=[]
		tempday6=[]
		tempday7=[]
		windmonth1=[]
		windmonth2=[]
		windmonth3=[]
		windmonth4=[]
		windmonth5=[]
		windmonth6=[]
		windweek1=[]
		windweek2=[]
		windweek3=[]
		windweek4=[]
		windday1=[]
		windday2=[]
		windday3=[]
		windday4=[]
		windday5=[]
		windday6=[]
		windday7=[]

		#####################################################################################################
		#Formatting Data
		#####################################################################################################
		#====================================================================================================
		'''
		Predicted value is at i+48, the (i+48+1)th value in the array.

		1152 Corresponds to 6 weeks with 48 intervals
		Likewise with other values
		'''
		#====================================================================================================
		
		
		
		for i in range(1152*7,train.size-prediction_step):
			loadmonth1.append(train[i-1152*7+prediction_step])
			loadmonth2.append(train[i-960*7+prediction_step])
			loadmonth3.append(train[i-768*7+prediction_step])
			loadmonth4.append(train[i-576*7+prediction_step])
			loadmonth5.append(train[i-384*7+prediction_step])
			loadmonth6.append(train[i-192*7+prediction_step])
			loadweek1.append(train[i-192*7+prediction_step])
			loadweek2.append(train[i-144*7+prediction_step])
			loadweek3.append(train[i-96*7+prediction_step])
			loadweek4.append(train[i-48*7+prediction_step])
			lhour.append(train[i-48:i+1,0])
			tempmonth1.append(temp[i-1152*7+prediction_step])
			tempmonth2.append(temp[i-960*7+prediction_step])
			tempmonth3.append(temp[i-768*7+prediction_step])
			tempmonth4.append(temp[i-576*7+prediction_step])
			tempmonth5.append(temp[i-384*7+prediction_step])
			tempmonth6.append(temp[i-192*7+prediction_step])
			windmonth1.append(wind[i-1152*7+prediction_step])
			windmonth2.append(wind[i-960*7+prediction_step])
			windmonth3.append(wind[i-768*7+prediction_step])
			windmonth4.append(wind[i-576*7+prediction_step])
			windmonth5.append(wind[i-384*7+prediction_step])
			windmonth6.append(wind[i-192*7+prediction_step])
			tempweek1.append(temp[i-192*7+prediction_step])
			tempweek2.append(temp[i-144*7+prediction_step])
			tempweek3.append(temp[i-96*7+prediction_step])
			tempweek4.append(temp[i-48*7+prediction_step])
			windweek1.append(wind[i-192*7+prediction_step])
			windweek2.append(wind[i-144*7+prediction_step])
			windweek3.append(wind[i-96*7+prediction_step])
			windweek4.append(wind[i-48*7+prediction_step])
			loadday1.append(train[i+prediction_step-48])
			loadday2.append(train[i-48+prediction_step-48])
			loadday3.append(train[i-96+prediction_step-48])
			loadday4.append(train[i-144+prediction_step-48])
			loadday5.append(train[i-192+prediction_step-48])
			loadday6.append(train[i-240+prediction_step-48])
			loadday7.append(train[i-288+prediction_step-48])
			tempday1.append(temp[i+prediction_step-48])
			tempday2.append(temp[i-48+prediction_step-48])
			tempday3.append(temp[i-96+prediction_step-48])
			tempday4.append(temp[i-144+prediction_step-48])
			tempday5.append(temp[i-192+prediction_step-48])
			tempday6.append(temp[i-240+prediction_step-48])
			tempday7.append(temp[i-288+prediction_step-48])
			windday1.append(wind[i+prediction_step-48])
			windday2.append(wind[i-48+prediction_step-48])
			windday3.append(wind[i-96+prediction_step-48])
			windday4.append(wind[i-144+prediction_step-48])
			windday5.append(wind[i-192+prediction_step-48])
			windday6.append(wind[i-240+prediction_step-48])
			windday7.append(wind[i-288+prediction_step-48])





		for i in range(1152*7,train.size-prediction_step):
			seasons.append(season[i+prediction_step])
			target.append(train[i+prediction_step])
			daytype.append(days[i+prediction_step])
			excludes.append(exclude[i+prediction_step])
			temps.append(temp[i+prediction_step])
			lunchs.append(lunch[i+prediction_step])
			winds.append(wind[i+prediction_step])
		loadday1=np.array(loadday1)
		loadday2=np.array(loadday2)
		loadday3=np.array(loadday3)
		loadday4=np.array(loadday4)
		loadday5=np.array(loadday5)
		loadday6=np.array(loadday6)
		loadday7=np.array(loadday7)
		tempday1=np.array(tempday1)
		tempday2=np.array(tempday2)
		tempday3=np.array(tempday3)
		tempday4=np.array(tempday4)
		tempday5=np.array(tempday5)
		tempday6=np.array(tempday6)
		tempday7=np.array(windday1)
		windday1=np.array(windday1)
		windday2=np.array(windday2)
		windday3=np.array(windday3)
		windday4=np.array(windday4)
		windday5=np.array(windday5)
		windday6=np.array(windday6)
		windday7=np.array(windday7)
		loadmonth1=np.array(loadmonth1)
		loadmonth2=np.array(loadmonth2)
		loadmonth3=np.array(loadmonth3)
		loadmonth4=np.array(loadmonth4)
		loadmonth5=np.array(loadmonth5)
		loadmonth6=np.array(loadmonth6)
		loadweek1=np.array(loadweek1)
		loadweek2=np.array(loadweek2)
		loadweek3=np.array(loadweek3)
		loadweek4=np.array(loadweek4)
		tempmonth1=np.array(tempmonth1)
		tempmonth2=np.array(tempmonth2)
		tempmonth3=np.array(tempmonth3)
		tempmonth4=np.array(tempmonth4)
		tempmonth5=np.array(tempmonth5)
		tempmonth6=np.array(tempmonth6)
		tempweek1=np.array(tempweek1)
		tempweek2=np.array(tempweek2)
		tempweek3=np.array(tempweek3)
		tempweek4=np.array(tempweek4)

		windmonth1=np.array(windmonth1)
		windmonth2=np.array(windmonth2)
		windmonth3=np.array(windmonth3)
		windmonth4=np.array(windmonth4)
		windmonth5=np.array(windmonth5)
		windmonth6=np.array(windmonth6)
		windweek1=np.array(windweek1)
		windweek2=np.array(windweek2)
		windweek3=np.array(windweek3)
		windweek4=np.array(windweek4)
		inputs=np.array(lhour)
		lunchs=np.array(lunchs)
		target,daytype,excludes,temps,winds,seasons=np.array(target),np.array(daytype),np.array(excludes),np.array(temps),np.array(winds),np.array(seasons)
		#only need to really reshape any values going into an RNN
		inputs=np.reshape(inputs,(inputs.shape[0],inputs.shape[1],1))
		daytype=np.reshape(daytype,(daytype.shape[0],))
		temps=np.reshape(temps,(temps.shape[0],))    
		excludes=np.reshape(excludes,(excludes.shape[0],))
		print('Data prepared successfully.\n')
	except:
		print('Data Error:\n')
		print('Possible Error: Data needs to be > 168 days')
		exit()
	#####################################################################################################
	#Model
	#####################################################################################################
	#Saving Weights
	filepath="weights.best.hdf5"
	checkpoint=ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='Max')
	callbacklist=[checkpoint]
	#====================================================================================================
	model_in=Input(shape=(inputs.shape[1],1)) 
	dayin=Input(shape=(1,))
	tempin=Input(shape=(1,))
	windin=Input(shape=(1,))
	lunchin=Input(shape=(1,))
	exin=Input(shape=(1,))
	seasin=Input(shape=(1,))
	loadmonth1i=Input(shape=(1,))
	loadmonth2i=Input(shape=(1,))
	loadmonth3i=Input(shape=(1,))
	loadmonth4i=Input(shape=(1,))
	loadmonth5i=Input(shape=(1,))
	loadmonth6i=Input(shape=(1,))
	loadweek1i=Input(shape=(1,))
	loadweek2i=Input(shape=(1,))
	loadweek3i=Input(shape=(1,))
	loadweek4i=Input(shape=(1,))
	tempmonth1i=Input(shape=(1,))
	tempmonth2i=Input(shape=(1,))
	tempmonth3i=Input(shape=(1,))
	tempmonth4i=Input(shape=(1,))
	tempmonth5i=Input(shape=(1,))
	tempmonth6i=Input(shape=(1,))
	windmonth1i=Input(shape=(1,))
	windmonth2i=Input(shape=(1,))
	windmonth3i=Input(shape=(1,))
	windmonth4i=Input(shape=(1,))
	windmonth5i=Input(shape=(1,))
	windmonth6i=Input(shape=(1,))
	tempweek1i=Input(shape=(1,))
	tempweek2i=Input(shape=(1,))
	tempweek3i=Input(shape=(1,))
	tempweek4i=Input(shape=(1,))
	windweek1i=Input(shape=(1,))
	windweek2i=Input(shape=(1,))
	windweek3i=Input(shape=(1,))
	windweek4i=Input(shape=(1,))
	loadday1i=Input(shape=(1,))
	loadday2i=Input(shape=(1,))
	loadday3i=Input(shape=(1,))
	loadday4i=Input(shape=(1,))
	loadday5i=Input(shape=(1,))
	loadday6i=Input(shape=(1,))
	loadday7i=Input(shape=(1,))
	tempday1i=Input(shape=(1,))
	tempday2i=Input(shape=(1,))
	tempday3i=Input(shape=(1,))
	tempday4i=Input(shape=(1,))
	tempday5i=Input(shape=(1,))
	tempday6i=Input(shape=(1,))
	tempday7i=Input(shape=(1,))
	windday1i=Input(shape=(1,))
	windday2i=Input(shape=(1,))
	windday3i=Input(shape=(1,))
	windday4i=Input(shape=(1,))
	windday5i=Input(shape=(1,))
	windday6i=Input(shape=(1,))
	windday7i=Input(shape=(1,))
	day1=concatenate([loadday1i,tempday1i,windday1i])
	day2=concatenate([loadday2i,tempday2i,windday2i])
	day3=concatenate([loadday3i,tempday3i,windday3i])
	day4=concatenate([loadday4i,tempday4i,windday4i])
	day5=concatenate([loadday5i,tempday5i,windday5i])
	day6=concatenate([loadday6i,tempday6i,windday6i])
	day7=concatenate([loadday7i,tempday7i,windday7i])
	day1=Dense(10)(day1)
	day2=Dense(10)(day2)
	day3=Dense(10)(day3)
	day4=Dense(10)(day4)
	day5=Dense(10)(day5)
	day6=Dense(10)(day6)
	day7=Dense(10)(day7)
	day=concatenate([day1,day2,day3,day4,day5,day6,day7])
	datcon1=concatenate([exin,seasin,lunchin])
	seasoncon1=Dense(5)(datcon1)
	seasoncon2=Dense(5)(datcon1)
	month1=concatenate([loadmonth1i,tempmonth1i,windmonth1i])
	month2=concatenate([loadmonth2i,tempmonth2i,windmonth2i])
	month3=concatenate([loadmonth3i,tempmonth3i,windmonth3i])
	month4=concatenate([loadmonth4i,tempmonth4i,windmonth4i])
	month5=concatenate([loadmonth5i,tempmonth5i,windmonth5i])
	month6=concatenate([loadmonth6i,tempmonth6i,windmonth6i])
	month1=Dense(10)(month1)
	month2=Dense(10)(month2)
	month3=Dense(10)(month3)
	month4=Dense(10)(month4)
	month5=Dense(10)(month5)
	month6=Dense(10)(month6)
	month=concatenate([month1,month2,month3,month4,month5,month6]) #p do the denses after the concats
	week1=concatenate([loadweek1i,tempweek1i,windweek1i])
	week2=concatenate([loadweek2i,tempweek2i,windweek2i])
	week3=concatenate([loadweek3i,tempweek3i,windweek3i])
	week4=concatenate([loadweek4i,tempweek4i,windweek4i])
	week1=Dense(10)(week1)
	week2=Dense(10)(week2)
	week3=Dense(10)(week3)
	week4=Dense(10)(week4)
	week=concatenate([week1,week2,week3,week4])
	fc1=concatenate([week,month,day,dayin,seasoncon1])
	hourly=LSTM(50)(model_in)
	#hourly=Flatten()(hourly)  #uncomment if above line becomes densely connected
	fc2=concatenate([hourly,seasoncon2])
	fc1=Dense(10)(fc1)
	fc2=Dense(10)(fc2)
	merge=concatenate([fc1,fc2,tempin,windin])
	merge=Dense(10)(merge)
	output=Dense(1)(merge)
	#====================================================================================================

	#----------------------------------------------------------------------------------------------------
	model=Model(inputs=[model_in,dayin,tempin,windin,lunchin,exin,seasin,loadmonth1i,loadmonth2i,loadmonth3i,loadmonth4i,
		loadmonth5i,loadmonth6i,tempmonth1i,tempmonth2i,tempmonth3i,tempmonth4i,tempmonth5i,tempmonth6i,
		loadweek1i,loadweek2i,loadweek3i,loadweek4i,tempweek1i,tempweek2i,tempweek3i,tempweek4i,loadday1i,
		loadday2i,loadday3i,loadday4i,loadday5i,loadday6i,loadday7i,tempday1i,tempday2i,tempday3i,tempday4i,
		tempday5i,tempday6i,tempday7i,windmonth1i,windmonth2i,windmonth3i,windmonth4i,windmonth5i,windmonth6i,
		windweek1i,windweek2i,windweek3i,windweek4i,windday1i,windday2i,windday3i,windday4i,windday5i,windday6i,
		windday7i],outputs=output)
	#----------------------------------------------------------------------------------------------------
	print('-------------------------------------------------')
	print('Compiling model...\n')
	try:
		model.compile(optimizer='adam',loss='mean_squared_error',metrics=[keras.losses.mean_absolute_percentage_error])
		print('Model compiled successfully.')	
	except:
		print('Error Compiling Model.')
		model.summary()
		print('Comment out error exceptions?')
		exit()
	#----------------------------------------------------------------------------------------------------
	print('-------------------------------------------------')
	print('Fitting model...\n')
	try:
		model.fit([inputs,daytype,temps,winds,lunchs,excludes,seasons,loadmonth1,loadmonth2,loadmonth3,loadmonth4,loadmonth5,
			loadmonth6,tempmonth1,tempmonth2,tempmonth3,tempmonth4,tempmonth5,tempmonth6,loadweek1,loadweek2,
			loadweek3,loadweek4,tempweek1,tempweek2,tempweek3,tempweek4,loadday1,loadday2,loadday3,loadday4,
			loadday5,loadday6,loadday7,tempday1,tempday2,tempday3,tempday4,tempday5,tempday6,tempday7,
			windmonth1,windmonth2,windmonth3,windmonth4,windmonth5,windmonth6,windweek1,windweek2,windweek3,
			windweek4,windday1,windday2,windday3,windday4,windday5,windday6,windday7],
			target,epochs=epochs,callbacks=callbacklist,batch_size=batch_size)
	except:
		print('Error fitting model.\n')
		print('Comment out error exceptions?')
		exit()			
	#----------------------------------------------------------------------------------------------------
	return model
	#====================================================================================================



def predict(model,df_train,prediction_step):#should be test but it works
	from keras.utils import plot_model
	from sklearn.preprocessing import MinMaxScaler
	import pandas as pd
	import numpy as np 
	print('-------------------------------------------------')
	print('Preparing data for predicitons...\n')
	try:
		scaler=MinMaxScaler(feature_range=(-1,1))
		train=pd.DataFrame(df_train['Load'].values)
		train=scaler.fit_transform(train)
		days=df_train['Days'].values
		temp=df_train['Temp'].values
		wind=df_train['Wind'].values
		lunch=df_train['Lunch'].values
		exclude=df_train['Excluding'].values
		season=df_train['Season'].values
		

		inputs=[]
		target=[]
		daytype=[]
		excludes=[]
		temps=[]
		lunchs=[]
		seasons=[]
		lhour=[]
		winds=[]
		loadmonth1=[]
		loadmonth2=[]
		loadmonth3=[]
		loadmonth4=[]
		loadmonth5=[]
		loadmonth6=[]
		loadweek1=[]
		loadweek2=[]
		loadweek3=[]
		loadweek4=[]
		tempmonth1=[]
		tempmonth2=[]
		tempmonth3=[]
		tempmonth4=[]
		tempmonth5=[]
		tempmonth6=[]
		tempweek1=[]
		tempweek2=[]
		tempweek3=[]
		tempweek4=[]
		loadday1=[]
		loadday2=[]
		loadday3=[]
		loadday4=[]
		loadday5=[]
		loadday6=[]
		loadday7=[]
		tempday1=[]
		tempday2=[]
		tempday3=[]
		tempday4=[]
		tempday5=[]
		tempday6=[]
		tempday7=[]
		windmonth1=[]
		windmonth2=[]
		windmonth3=[]
		windmonth4=[]
		windmonth5=[]
		windmonth6=[]
		windweek1=[]
		windweek2=[]
		windweek3=[]
		windweek4=[]
		windday1=[]
		windday2=[]
		windday3=[]
		windday4=[]
		windday5=[]
		windday6=[]
		windday7=[]

		#####################################################################################################
		#Formatting Data
		#####################################################################################################
		#====================================================================================================
		'''
		Predicted value is at i+48, the (i+48+1)th value in the array.

		1152 Corresponds to 6 weeks with 48 intervals
		Likewise with other values
		'''
		#====================================================================================================
		
		
		
		for i in range(1152*7,train.size-prediction_step):
			loadmonth1.append(train[i-1152*7+prediction_step])
			loadmonth2.append(train[i-960*7+prediction_step])
			loadmonth3.append(train[i-768*7+prediction_step])
			loadmonth4.append(train[i-576*7+prediction_step])
			loadmonth5.append(train[i-384*7+prediction_step])
			loadmonth6.append(train[i-192*7+prediction_step])
			loadweek1.append(train[i-192*7+prediction_step])
			loadweek2.append(train[i-144*7+prediction_step])
			loadweek3.append(train[i-96*7+prediction_step])
			loadweek4.append(train[i-48*7+prediction_step])
			lhour.append(train[i-48:i+1,0])
			tempmonth1.append(temp[i-1152*7+prediction_step])
			tempmonth2.append(temp[i-960*7+prediction_step])
			tempmonth3.append(temp[i-768*7+prediction_step])
			tempmonth4.append(temp[i-576*7+prediction_step])
			tempmonth5.append(temp[i-384*7+prediction_step])
			tempmonth6.append(temp[i-192*7+prediction_step])
			windmonth1.append(wind[i-1152*7+prediction_step])
			windmonth2.append(wind[i-960*7+prediction_step])
			windmonth3.append(wind[i-768*7+prediction_step])
			windmonth4.append(wind[i-576*7+prediction_step])
			windmonth5.append(wind[i-384*7+prediction_step])
			windmonth6.append(wind[i-192*7+prediction_step])
			tempweek1.append(temp[i-192*7+prediction_step])
			tempweek2.append(temp[i-144*7+prediction_step])
			tempweek3.append(temp[i-96*7+prediction_step])
			tempweek4.append(temp[i-48*7+prediction_step])
			windweek1.append(wind[i-192*7+prediction_step])
			windweek2.append(wind[i-144*7+prediction_step])
			windweek3.append(wind[i-96*7+prediction_step])
			windweek4.append(wind[i-48*7+prediction_step])
			loadday1.append(train[i+prediction_step-48])
			loadday2.append(train[i-48+prediction_step-48])
			loadday3.append(train[i-96+prediction_step-48])
			loadday4.append(train[i-144+prediction_step-48])
			loadday5.append(train[i-192+prediction_step-48])
			loadday6.append(train[i-240+prediction_step-48])
			loadday7.append(train[i-288+prediction_step-48])
			tempday1.append(temp[i+prediction_step-48])
			tempday2.append(temp[i-48+prediction_step-48])
			tempday3.append(temp[i-96+prediction_step-48])
			tempday4.append(temp[i-144+prediction_step-48])
			tempday5.append(temp[i-192+prediction_step-48])
			tempday6.append(temp[i-240+prediction_step-48])
			tempday7.append(temp[i-288+prediction_step-48])
			windday1.append(wind[i+prediction_step-48])
			windday2.append(wind[i-48+prediction_step-48])
			windday3.append(wind[i-96+prediction_step-48])
			windday4.append(wind[i-144+prediction_step-48])
			windday5.append(wind[i-192+prediction_step-48])
			windday6.append(wind[i-240+prediction_step-48])
			windday7.append(wind[i-288+prediction_step-48])





		for i in range(1152*7,train.size-prediction_step):
			seasons.append(season[i+prediction_step])
			target.append(train[i+prediction_step])
			daytype.append(days[i+prediction_step])
			excludes.append(exclude[i+prediction_step])
			temps.append(temp[i+prediction_step])
			lunchs.append(lunch[i+prediction_step])
			winds.append(wind[i+prediction_step])
		loadday1=np.array(loadday1)
		loadday2=np.array(loadday2)
		loadday3=np.array(loadday3)
		loadday4=np.array(loadday4)
		loadday5=np.array(loadday5)
		loadday6=np.array(loadday6)
		loadday7=np.array(loadday7)
		tempday1=np.array(tempday1)
		tempday2=np.array(tempday2)
		tempday3=np.array(tempday3)
		tempday4=np.array(tempday4)
		tempday5=np.array(tempday5)
		tempday6=np.array(tempday6)
		tempday7=np.array(windday1)
		windday1=np.array(windday1)
		windday2=np.array(windday2)
		windday3=np.array(windday3)
		windday4=np.array(windday4)
		windday5=np.array(windday5)
		windday6=np.array(windday6)
		windday7=np.array(windday7)
		loadmonth1=np.array(loadmonth1)
		loadmonth2=np.array(loadmonth2)
		loadmonth3=np.array(loadmonth3)
		loadmonth4=np.array(loadmonth4)
		loadmonth5=np.array(loadmonth5)
		loadmonth6=np.array(loadmonth6)
		loadweek1=np.array(loadweek1)
		loadweek2=np.array(loadweek2)
		loadweek3=np.array(loadweek3)
		loadweek4=np.array(loadweek4)
		tempmonth1=np.array(tempmonth1)
		tempmonth2=np.array(tempmonth2)
		tempmonth3=np.array(tempmonth3)
		tempmonth4=np.array(tempmonth4)
		tempmonth5=np.array(tempmonth5)
		tempmonth6=np.array(tempmonth6)
		tempweek1=np.array(tempweek1)
		tempweek2=np.array(tempweek2)
		tempweek3=np.array(tempweek3)
		tempweek4=np.array(tempweek4)

		windmonth1=np.array(windmonth1)
		windmonth2=np.array(windmonth2)
		windmonth3=np.array(windmonth3)
		windmonth4=np.array(windmonth4)
		windmonth5=np.array(windmonth5)
		windmonth6=np.array(windmonth6)
		windweek1=np.array(windweek1)
		windweek2=np.array(windweek2)
		windweek3=np.array(windweek3)
		windweek4=np.array(windweek4)
		inputs=np.array(lhour)
		lunchs=np.array(lunchs)
		target,daytype,excludes,temps,winds,seasons=np.array(target),np.array(daytype),np.array(excludes),np.array(temps),np.array(winds),np.array(seasons)
		#only need to really reshape any values going into an RNN
		inputs=np.reshape(inputs,(inputs.shape[0],inputs.shape[1],1))
		daytype=np.reshape(daytype,(daytype.shape[0],))
		temps=np.reshape(temps,(temps.shape[0],))    
		excludes=np.reshape(excludes,(excludes.shape[0],))
		print('Data prepared successfully.\n')
	except:
		print('Data Error:\n')
		print('Possible Error: Data needs to be > 168 days')
		exit()
	print('-------------------------------------------------')
	print('Predicting...\n')
	predictions = model.predict([inputs,daytype,temps,winds,lunchs,excludes,seasons,loadmonth1,loadmonth2,loadmonth3,loadmonth4,loadmonth5,
			loadmonth6,tempmonth1,tempmonth2,tempmonth3,tempmonth4,tempmonth5,tempmonth6,loadweek1,loadweek2,
			loadweek3,loadweek4,tempweek1,tempweek2,tempweek3,tempweek4,loadday1,loadday2,loadday3,loadday4,
			loadday5,loadday6,loadday7,tempday1,tempday2,tempday3,tempday4,tempday5,tempday6,tempday7,
			windmonth1,windmonth2,windmonth3,windmonth4,windmonth5,windmonth6,windweek1,windweek2,windweek3,
			windweek4,windday1,windday2,windday3,windday4,windday5,windday6,windday7])
	predictions=scaler.inverse_transform(predictions)
	try:
		plot_model(model, to_file='model.png',show_shapes=True,show_layer_names=True)
		return pd.DataFrame(predictions)
	except:
		return pd.DataFrame(predictions)



