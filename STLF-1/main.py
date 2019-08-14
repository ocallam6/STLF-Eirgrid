"""
Prototype1

NB: Deletes first 168 days for prediction. 

Neural Network Using Keras

Predicts next day values. 48 steps ahead forecast.

Uses daytype,temperature,holidays and utilises autocorrelation.

Main Program
"""
#===========================================================================
from cleancsv import clean_csv
from traintest import data_split,predict_data
from data import datas
import datetime
from network import model_build,predict
from analysis import plot_values, errors
import pandas as pd
import time as time
import keras
import warnings
import tensorflow as tf 
warnings.filterwarnings("ignore") #comment out to display warnings
#===========================================================================
start_time = time.time()
#===========================================================================
date_begin='16-Jul-2017'
date_predict='01-Feb-2019'
date_end=datetime.datetime.strftime(datetime.datetime.strptime(date_predict,
	'%d-%b-%Y')-datetime.timedelta(days=1),'%d-%b-%Y')
clean_csv(date_begin,date_end)
#===========================================================================
df_train=datas(date_begin,date_end)
#===========================================================================

#---------------------------------------------------------------------------
df_train,df_test=data_split(df_train)
#---------------------------------------------------------------------------
'''
data_split for analysis
predict_data for real life

tempmax=23
df_train,df_predict=predict_data(df_train,date_end,date_predict,tempmax_predict)
'''
#---------------------------------------------------------------------------

#===========================================================================
model=model_build(df_train,epochs=1,batch_size=32)
model.load_weights("weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=[keras.losses.mean_absolute_percentage_error])
#===========================================================================
predictions=predict(model,df_test)
df_test.drop(df_test.head(8112).index,inplace=True)
df_test['Predicted']=predictions.values
#===========================================================================
plot_values(df_test)
errors(df_test)
#===========================================================================
print("          RUNTIME       \n --- %s seconds ---" % (time.time() - start_time))
#===========================================================================

