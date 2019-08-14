def model_build(df_train,epochs,batch_size):
	import keras
	from sklearn.preprocessing import MinMaxScaler
	from keras.layers import GRU, Dense, Input, Flatten, Dropout, Conv1D,LSTM
	from keras.models import Model
	from keras.utils import plot_model
	from keras.callbacks import ModelCheckpoint
	import pandas as pd
	import numpy as np 
	import os	
	scaler=MinMaxScaler()
	train=pd.DataFrame(df_train['Load'].values)
	train=scaler.fit_transform(train)

	days=df_train['Days'].values
	temp=df_train['Temp'].values
	exclude=df_train['Excluding'].values
	inputs=[]
	target=[]
	daytype=[]
	excludes=[]
	temps=[]
	from keras.layers.merge import concatenate
	#checkpoint
	filepath="weights.best.hdf5"
	checkpoint=ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='Max')
	callbacklist=[checkpoint]
	for i in range(336,train.size-48):
		inputs.append(train[i-336:(i+1),0])
		#variant======target.append(train[i:i+48,0])
		target.append(train[i+48])
		daytype.append(days[i+48])
		excludes.append(exclude[i+48])
		temps.append(temp[i+48])
	inputs,target,daytype,excludes,temps=np.array(inputs),np.array(target),np.array(daytype),np.array(excludes),np.array(temps)
	inputs=np.reshape(inputs,(inputs.shape[0],inputs.shape[1],1))
	daytype=np.reshape(daytype,(daytype.shape[0],))
	temps=np.reshape(temps,(temps.shape[0],))    
	excludes=np.reshape(excludes,(excludes.shape[0],))

	model_in=Input(shape=(inputs.shape[1],1))
	dayin=Input(shape=(1,))
	tempin=Input(shape=(1,))
	#in2=concatenate([dayin,tempin])
	exin=Input(shape=(1,))
		


	x=LSTM(50,return_sequences=True)(model_in)
	x=Dense(20)(x)
	x=Flatten()(x)

	y=Conv1D(100,kernel_size=10,activation='relu',padding='same')(model_in)
	y=Conv1D(75,kernel_size=8,activation='relu',padding='same')(y)
	y=Conv1D(75,kernel_size=5,activation='relu',padding='same')(y)
	y=Dense(20)(y)
	y=Flatten()(y)

	z=Dense(100)(tempin)
	z=Dense(20)(z)
	#output3=Flatten()(output3)


	merge = concatenate([x,y,z,dayin,exin])
	#merge=Dropout(0.6)(merge)

	hidden6=Dense(60)(merge)
	output=Dense(1)(hidden6)


	model=Model(inputs=[model_in,dayin,tempin,exin],outputs=output)
	model.compile(optimizer='adam',loss='mean_squared_error',metrics=[keras.losses.mean_absolute_percentage_error])
	model.fit([inputs,daytype,temps,excludes],target,epochs=epochs,callbacks=callbacklist,batch_size=batch_size)
	return model
def predict(model,df_test):
	from keras.utils import plot_model
	import os
	os.environ["PATH"] += os.pathsep + '/Documents/bin'
	from sklearn.preprocessing import MinMaxScaler
	import pandas as pd
	import numpy as np
	scaler=MinMaxScaler(feature_range=(0,1))
	test=pd.DataFrame(df_test['Load'].values)
	test=scaler.fit_transform(test)
	day=df_test['Days'].values
	temp=df_test['Temp'].values
	exclude=df_test['Excluding'].values
	test_data=[]
	daytype=[]
	temps=[]
	excludes=[]
	for i in range(336,test.size-48):
		test_data.append(test[i-336:(i+1),0])
		daytype.append(day[i+48])
		temps.append(temp[(i+48)])
		excludes.append(exclude[i+48])
	test_data=np.array(test_data)
	test_data=np.reshape(test_data,(test_data.shape[0],test_data.shape[1],1))
	daytype=np.array(daytype)
	daytype=np.reshape(daytype,(daytype.shape[0],))
	temps=np.array(temps)
	temps=np.reshape(temps,(temps.shape[0],))
	excludes=np.array(excludes)
	excludes=np.reshape(excludes,(excludes.shape[0]),)
	predictions = model.predict([test_data,daytype,temps,excludes])
	predictions=scaler.inverse_transform(predictions)
	plot_model(model, to_file='model.png',show_shapes=True,show_layer_names=True)

	return pd.DataFrame(predictions)


