# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:24:33 2019

@author: ocallaghan_m
"""
#Ireland Data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Raw_To_CSV import raw_to_csv
import os
from Operations_on_CSV import plot_dataframe
from Operations_on_CSV import addcol
import datetime


def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

#Create Data We Want
#17-May-2014 for 6 months

start_date='05-Jun-2018'
end_date='06-Jun-2019'



raw_to_csv(start_date,end_date,'Load','IE') #create csv should really do this elsewhere
def df_col_name(date_begin,date_end,var,loc):
    return var+loc+'_'+date_begin+'_'+date_end
col_title=df_col_name(start_date,end_date,'Load','IE')
date_rng = pd.date_range(start = start_date, end= end_date,freq='30min') #timestamp
df = pd.DataFrame(date_rng, columns=['Date and Time'])
new_col = pd.read_csv(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s.csv'%col_title)     #doesn't write anything new
df['%s'%col_title] = new_col
plt.figure(1)
plot_dataframe(df,'Time Series',col_title,'')   
#Data Cleaning
#Step 1: Missing Values Load
orig= open(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s.csv'%col_title,"r+")
lines = orig.readlines()
orig.close()
new= open(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s_temp.csv'%col_title,"w+")
for i in range(0, len(lines)):
    line = lines[i]
    su=0
    for word in line.split():
        if(word=='0.00'):
                
                su=str(round((float(lines[i-2].split(' ',1)[0])+float(lines[i+2].split(' ',1)[0]))/2,2))
                    
                #new.write('%s \n'%word)
               # print('0.00 at %d'%(i+1))
                
                #print('Changed to: %s'%su)
                new.write('%s \n'%su)
                
        else:
            new.write("%s \n" % word) #use this for replacing
new.close()
orig= open(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s.csv'%col_title,"w+")
new= open(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s_temp.csv'%col_title,"r+")
lines = new.readlines()
new.close()
for i in range(0, len(lines)):
    line = lines[i]
    for word in line.split():
        orig.write("%s \n" %word)
os.remove(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s_temp.csv'%col_title)
orig.close()
#Step2: Repeated Values
orig= open(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s.csv'%col_title,"r+")
lines = orig.readlines()
orig.close()
new= open(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s_temp.csv'%col_title,"w+")
for i in range(0, len(lines)):
    line = lines[i]
    if (lines[i-1]==lines[i]):
        
        for word in line.split():
            su=str(round((float(lines[i-1].split(' ',1)[0])+float(lines[i+1].split(' ',1)[0]))/2,2))
            #print('Repeat at %d'%(i-1))
            #print('Time = %s'%df['Date and Time'][(i-1)])
            #print('Changed to %s'%su )
            new.write("%s \n" %su)
    else: 
        for word in line.split():
            new.write("%s \n" %word)
new.close()
orig= open(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s.csv'%col_title,"w+")
new= open(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s_temp.csv'%col_title,"r+")
lines = new.readlines()
new.close()
for i in range(0, len(lines)):
    line = lines[i]
    for word in line.split():
        orig.write("%s \n" % word)
os.remove(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s_temp.csv'%col_title)

orig.close()

raw_to_csv(start_date,end_date,'Weather Item=2','')


col2=df_col_name(start_date,end_date,'Weather Item=2','')
date_rng = pd.date_range(start = start_date, end= end_date,freq='30min') #timestamp
df_new = pd.DataFrame(date_rng, columns=['Date and Time'])
new_col = pd.read_csv(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s.csv'%col_title)     #doesn't write anything new
df_new['%s'%col_title] = new_col
addcol(col2,df_new)
#plt.figure(2)
#plot_dataframe(df_new,'Time Series',col_title,'')


#scatter stuff---------------------------------


plt.figure(2)
plot_dataframe(df_new, 'Scatter',col_title,col2)

    
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
