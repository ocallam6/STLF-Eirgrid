"""
Prototype1

Neural Network Using Keras

Predicts next day values. 48 steps ahead forecast.

Uses daytype,temperature,holidays and utilises autocorrelation.


"""

import os
def extract_data(loc):
    new_file=loc
    k= open(r'%s.txt'%new_file,"w+")
    f= open(r'raw_data.txt','r')
    searchlines = f.readlines()
    for i, line in enumerate(searchlines):
        if "[EXCLUDED DATES]" in line:
            while '[LOAD IDENTS]' not in searchlines[i]:
                i=i+1
                k.write(searchlines[i])

        


def write_csv(x):
    k= open(r'%s.csv'%x,"w+")
    f= open(r'%s.txt'%x)
    k.write("X \n")
    s='FCA=%s'%x
    lines = f.readlines()
    for i in range(0, len(lines)):
        line = lines[i]
        for word in line.split():
            
            if(s in line.split()):
                k.write('%s \n'%word)
                break
        


def raw_to_csv(loc):
    extract_data(loc)
    write_csv(loc)
    os.remove(r'%s.txt'%loc)



def excludes(loc):
    raw_to_csv(loc)
    import pandas as pd
    dates=pd.read_csv(r'%s.csv'%loc)
    def rem(date):
        return date[5:]
    dates['Date']=dates.ix[:,0].apply(rem)
    Dates=dates['Date'].values.astype(str)
    import datetime 
    a=[]
    formats="%d-%b-%Y"
    for date in Dates:
        date = date[:-1] #space in data
        a.append(datetime.datetime.strptime(date,formats).strftime("%Y-%m-%d"))
  
    return a 



