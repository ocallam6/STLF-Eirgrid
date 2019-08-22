"""
Prototype1

Neural Network Using Keras

Predicts next day values. 48 steps ahead forecast.

Uses daytype,temperature,holidays and utilises autocorrelation.

CSV Create
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Raw_To_CSV import raw_to_csv
import os

location='IE'           #location variable, NI is northern ireland

def clean_csv(start_date,end_date):

    def isfloat(value):
      try:
        float(value)
        return True
      except ValueError:
        return False
    #csv creation
    raw_to_csv(start_date,end_date,'Load',location) 
    def df_col_name(date_begin,date_end,var,loc):
        return var+loc+'_'+date_begin+'_'+date_end
    col_title=df_col_name(start_date,end_date,'Load',location)
    date_rng = pd.date_range(start = start_date, end= end_date,freq='30min') #timestamp
    df = pd.DataFrame(date_rng, columns=['Date and Time'])
    new_col = pd.read_csv(r'%s.csv'%col_title)     #doesn't write anything new
    df['%s'%col_title] = new_col
    #Data Cleaning
    #Step 1: Missing Values Load
    orig= open(r'%s.csv'%col_title,"r+")
    lines = orig.readlines()
    orig.close()
    new= open(r'%s_temp.csv'%col_title,"w+")
    for i in range(0, len(lines)):
        line = lines[i]
        su=0
        for word in line.split():
            if(word=='0.00'): #find zeros
                    
                    su=str(round((float(lines[i-2].split(' ',1)[0])+float(lines[i+2].split(' ',1)[0]))/2,2)) #linearly interpolate

                    new.write('%s \n'%su)
                    
            else:
                new.write("%s \n" % word) #use this for replacing
    new.close()
    orig= open(r'%s.csv'%col_title,"w+")
    new= open(r'%s_temp.csv'%col_title,"r+")
    lines = new.readlines()
    new.close()
    for i in range(0, len(lines)):
        line = lines[i]
        for word in line.split():
            orig.write("%s \n" %word)
    os.remove(r'%s_temp.csv'%col_title)
    orig.close()
    #Step2: Repeated Values
    orig= open(r'%s.csv'%col_title,"r+")
    lines = orig.readlines()
    orig.close()
    new= open(r'%s_temp.csv'%col_title,"w+")
    for i in range(0, len(lines)):
        line = lines[i]
        if (lines[i-1]==lines[i]):
            
            for word in line.split():
                su=str(round((float(lines[i-1].split(' ',1)[0])+float(lines[i+1].split(' ',1)[0]))/2,2)) #linearly interpolate

                new.write("%s \n" %su)
        else: 
            for word in line.split():
                new.write("%s \n" %word)
    new.close()
    orig= open(r'%s.csv'%col_title,"w+")
    new= open(r'%s_temp.csv'%col_title,"r+")
    lines = new.readlines()
    new.close()
    for i in range(0, len(lines)):
        line = lines[i]
        for word in line.split():
            orig.write("%s \n" % word)
    os.remove(r'%s_temp.csv'%col_title)

    orig.close()

    #create the weather variable csv
    #raw_to_csv(start_date,end_date,'Weather Item=6','') #for northern ireland temperature
    raw_to_csv(start_date,end_date,'Weather Item=1','')
    raw_to_csv(start_date,end_date,'Weather Item=2','')
    
