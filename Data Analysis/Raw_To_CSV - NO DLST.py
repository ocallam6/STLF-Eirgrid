# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:05:53 2019

@author: ocallaghan_m
"""
import os
def extract_data(date_begin,date_end,var,loc):
    new_file=var+loc+'_'+date_begin+'_'+date_end
    k= open(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\TextFiles\%s.txt'%new_file,"w+")
    f= open(r'C:/Users/ocallaghan_m/Desktop/Forecasting/Data/Raw Data/lf_archived_20190605_20140517.txt','r')
    searchlines = f.readlines()
    for i, line in enumerate(searchlines):
        if "%s"%var in line and "%s"%loc in line:
            while ';Date=%s'%date_begin not in searchlines[i]:
                i=i+1
            if ';Date=%s'%date_begin in searchlines[i]:
                for l in searchlines[i:len(searchlines)]: 
                    if ';Date=%s'%date_end not in l:   
                        k.write(l)
                        #print(l)
                    else: break
        
    return new_file

def write_csv(x):
    k= open(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s.csv'%x,"w+")
    f= open(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\TextFiles\%s.txt'%x)
    k.write("X \n")

    lines = f.readlines()
    for i in range(0, len(lines)):
        line = lines[i]
        for word in line.split():
            if(len(line.split())<3):
                    continue
            else:
                k.write("%s \n" % word)


def raw_to_csv(date_begin,date_end,var,loc):
    name = extract_data(date_begin,date_end,var,loc)
    write_csv(name)
    os.remove(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\TextFiles\%s.txt'%name)
    
#if want to write csv file from Dataframe its using df.write_csv or whatever,
    #example in mean analysis
#raw_to_csv('17-Dec-2014','17-May-2015','Weather Item=1','')