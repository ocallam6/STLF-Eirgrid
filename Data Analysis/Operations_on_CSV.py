# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:22:46 2019
Given Raw SCADA data this programme will read it in, create a dataframe and csv file
It also has a function to plot two variables of dataframe
To see weather item numbering check readme.txt
@author: ocallaghan_m
"""
from Raw_To_CSV import raw_to_csv
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
#Column names of dataframe given same inputs as plot_data
def df_col_name(date_begin,date_end,var,loc):
    return var+loc+'_'+date_begin+'_'+date_end

#name of raw data manually inputted

#add column to dataframe
def addcol(col_title,data_fr):
    new_col = pd.read_csv(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s.csv'%col_title)     
    data_fr['%s'%col_title] = new_col

#before using this need to create a csv file with the columns etc.
def plot_data(date_begin,date_end,var1,loc1,var2,loc2):    
    #Used to be here for creating csv here but no longer, delete at some point
    #write_csv(extract_data('%s'%date_begin,'%s'%date_end,'%s'%var1,'%s'%loc1))
    #write_csv(extract_data('%s'%date_begin,'%s'%date_end,'%s'%var2,'%s'%loc2))
    date_rng = pd.date_range(start = '%s'%date_begin, end='%s'%date_end,freq='30min')
    
    df = pd.DataFrame(date_rng, columns=['Date and Time'])
    
    df = pd.DataFrame(date_rng, columns=['Date and Time'])
    a=var1+loc1+'_'+date_begin+'_'+date_end
    b=var2+loc2+'_'+date_begin+'_'+date_end
    addcol(a,df)   #this is where reads in cs file
    addcol(b,df)
    return df
#plot data
def plot_dataframe(df,plot_type,col_a,col_b):
    if(plot_type == 'Time Series'):
        df = df.set_index('Date and Time')
        #plt.figure(1)
        plt.clf()
        sns.set(rc={'figure.figsize':(11, 4)})
        df[col_a].plot(linewidth=0.5);
        plt.title('%s Load Time Series'%col_a)
        plt.xlabel('Time')
        plt.ylabel('Load')
        plt.show()
        
    if(plot_type == 'Basic Scatter'):   
        #plt.figure(2)
        plt.scatter(df[col_b], df[col_a])
        plt.title(col_a+' vs '+col_b)
        plt.xlabel(col_b)
        plt.ylabel(col_a)
        plt.show()
        
        
        
    if(plot_type == 'Heat Map'):
        #plt.figure(3)
        plt.clf()
        plt.title(col_a+' vs '+col_b)
        plt.xlabel(col_b)
        plt.ylabel(col_a)
        x=df[col_b]
        y=df[col_a]
        #Heat map sort of example
        plt.hexbin(x,y,gridsize=(75,75))
        plt.colorbar()
        plt.show()
        #reducing dot size
    if(plot_type == 'Scatter'):
        #plt.figure(4)
        plt.clf()
        plt.title(col_a+' vs '+col_b)
        plt.xlabel(col_b)
        plt.ylabel(col_a)
        plt.plot(col_b,col_a,data=df,linestyle='',marker='o',markersize=1)
        plt.show()
        #transperancy
    if(plot_type == 'Transperancy'):
        #plt.figure(5)
        plt.clf()
        plt.title(col_a+' vs '+col_b)
        plt.xlabel(col_b)
        plt.ylabel(col_a)
        plt.plot(col_b,col_a,data=df,linestyle='',marker='o',markersize=3,alpha=0.05,color='purple')   
        plt.show()
        
        #density plot
        
        #plt.figure(6)
        #sns.kdeplot(x, y, cmap="Reds", shade=True)
        #plt.show
    if(plot_type == 'Sample'):
        #sampling
        df_sample=df.sample(1000)
        #plt.figure(6)
        plt.clf()
        plt.title(col_a+' vs '+col_b)
        plt.xlabel(col_b)
        plt.ylabel(col_a)
        plt.plot(col_b,col_a,data=df_sample,linestyle='',marker='x',markersize=3)
        plt.show()
    
    
    




#df=plot_data('19-May-2014','19-May-2016','Load','IE','Weather Item=1','')
#col_a= df_col_name('19-May-2014','19-May-2016','Load','IE')
#col_b= df_col_name('19-May-2014','19-May-2016','Weather Item=1','')
#plot_dataframe(df,'Transperancy',col_a,col_b)



