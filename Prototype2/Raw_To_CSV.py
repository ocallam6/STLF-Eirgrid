"""
Prototype1

Neural Network Using Keras

Predicts next day values. 48 steps ahead forecast.

Uses daytype,temperature,holidays and utilises autocorrelation.

"""

import os

#Spring Forward, Fall Back
forward=[';Date=30-Mar-2014',';Date=29-Mar-2015',';Date=27-Mar-2016',';Date=26-Mar-2017',';Date=25-Mar-2018',';Date=31-Mar-2019']
backward=[';Date=26-Oct-2014',';Date=25-Oct-2015',';Date=30-Oct-2016'';Date=29-Oct-2017'';Date=28-Oct-2018'';Date=27-Oct-2019']

def extract_data(date_begin,date_end,var,loc):
    new_file=var+loc+'_'+date_begin+'_'+date_end
    k= open(r'%s.txt'%new_file,"w+")
    f= open(r'raw_data.txt','r')
    searchlines = f.readlines()
    for i, line in enumerate(searchlines):
        if "%s"%var in line and "%s"%loc in line:
            while ';Date=%s'%date_begin not in searchlines[i]:
                i=i+1
            if ';Date=%s'%date_begin in searchlines[i]:
                for l in searchlines[i:len(searchlines)]: 
                    if ';Date=%s'%date_end not in l:  
                        if('Item=1' in l or 'Item=2' in l or 'Item=3' in l or 'Item=4' in l or 'Item=5' in l or 'Item=6' in l or 'Item=7' in l or 'Item=8' in l): 
                            break
                        else:
                            k.write(l)
                    else: break
        
    return new_file

def write_csv(x):
    k= open(r'%s.csv'%x,"w+")
    f= open(r'%s.txt'%x)
    k.write("X \n")

    lines = f.readlines()
    i=0
    while i < len(lines):
        line = lines[i]
        if(len(line.split())<3):
            for word in line.split():
                if(word in forward):
                    line1=lines[i+1]
                    appending=[]
                    for word in line1.split():
                        appending.append(word)                  
                    for j in range(0,2):
                        k.write("%s \n" %appending[j]) #mightneed to be .split
                    k.write("0.00 \n")
                    k.write("0.00 \n")#i think string
                    for j in range(2,len(appending)):
                        k.write("%s \n" %appending[j])
                    for m in range(2,6):
                        line1=lines[i+m]
                        for word in line1.split():
                            k.write("%s \n" % word)
                    line1=lines[i+6]
                    appending=[]
                    for word in line1.split():
                        appending.append(word)
                        
                    for j in range(0,6):
                        k.write("%s \n" %appending[j])
                    i=i+8
                elif(word in backward):
                    line1=lines[i+1]
                    appending=[]
                    for word in line1.split():
                        appending.append(word)                    
                    for j in range(0,4):
                        k.write("%s \n" %appending[j]) #mightneed to be .split
                    for j in range(6,len(appending)):
                        k.write("%s \n" %appending[j])
                    for m in range(2,8):
                        line1=lines[i+m]
                        for word in line1.split():
                            k.write("%s \n" % word)
                    i=i+8
                else: i=i+1
        else:
            for word in line.split():
                k.write("%s \n" % word)
            i=i+1

def raw_to_csv(date_begin,date_end,var,loc):
    name = extract_data(date_begin,date_end,var,loc)
    write_csv(name)
    os.remove(r'%s.txt'%name)



