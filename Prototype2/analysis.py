"""
Prototype1

Neural Network Using Keras

Predicts next day values. 48 steps ahead forecast.

Uses daytype,temperature,holidays and utilises autocorrelation.

Plots and Analysis
"""
def plot_values(df):
	import pandas as pd 
	import matplotlib.pyplot as plt 
	plt.plot('Date and Time','Load',data=df,marker='',color='blue',linewidth=1)
	plt.plot('Date and Time','Predicted',data=df,marker='',color='red',linewidth=1)
	plt.legend()
	plt.show()

def errors(df):
	import pandas as pd 
	import matplotlib.pyplot as plt 
	import numpy as np
	difference = df.Load.values-df.Predicted.values
	df['Diff']=np.absolute(difference)
	abs_val=np.absolute(difference).mean()
	print('-------------------------------------------------\n')
	print('Mean Error=%f'%abs_val)
	mape=np.absolute(difference/df.Load.values).mean()
	print('MAPE=%f'%mape)
	print('-------------------------------------------------\n')
