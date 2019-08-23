
import pandas as pd 
import matplotlib.pyplot as plt
from Operations_on_CSV import plot_dataframe

start_date='01-Jan-2015'
end_date='01-Jan-2019'
#for next hour
#data
#==============================================================================================
#full plot
def df_col_name(date_begin,date_end,var,loc):
    return var+loc+'_'+date_begin+'_'+date_end
col_title=df_col_name(start_date,end_date,'Load','IE')
date_rng = pd.date_range(start = start_date, end= end_date,freq='30min') #timestamp
df = pd.DataFrame(date_rng, columns=['Date and Time'])
new_col = pd.read_csv(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s.csv'%col_title)     #doesn't write anything new
df['%s'%col_title] = new_col
col_title2=df_col_name(start_date,end_date,'Weather Item=1','')
new_col = pd.read_csv(r'C:\Users\ocallaghan_m\Desktop\Forecasting\Data\CSV_Files\%s.csv'%col_title2)     #doesn't write anything new
df['%s'%col_title2] = new_col

df.set_index('Date and Time',inplace=True)
df['Date and Time']=df.index
#df=df.loc[df.index.minute==0] #df now hourly
df.drop(df.tail(1).index,inplace=True)
###############################################################################################
#TIME SERIES PLOT
###############################################################################################



plot_dataframe(df,'Time Series',col_title,col_title2)


###############################################################################################
#DAYTYPE SCATTER PLOT
###############################################################################################



df['Day of Week']=df['Date and Time'].dt.day_name()
def day_to_num(day):
    days=['Monday','Tuesday','Wednesday','Thursday','Friday']
    if day in days:
        return 1
    else: return 0
df['Day of Week']=df['Date and Time'].dt.day_name().apply(day_to_num)
df_weekday=df.loc[df['Day of Week']==1]
df_weekend=df.loc[df['Day of Week']==0]
plot_dataframe(df_weekend,'Scatter',col_title,col_title2)
#-------------------------------------------------------------------------------------
df_weekday['Hour']=df_weekday['Date and Time'].dt.hour
df_hour=df_weekday.loc[df.Hour==20]
plot_dataframe(df_hour,'Scatter',col_title,col_title2)
print(df_hour[col_title].corr(df_hour[col_title2]))

df_weekend['Hour']=df_weekend['Date and Time'].dt.hour
df_hour=df_weekend.loc[df.Hour==20]
plot_dataframe(df_hour,'Scatter',col_title,col_title2)
#-------------------------------------------------------------------------------------
###############################################################################################
#CORRELATION PER HOUR PER WEEKDAY
###############################################################################################

mean=[]
for i in range(0,24):
    df_weekday['Hour']=df_weekday['Date and Time'].dt.hour
    df_hour=df_weekday.loc[df.Hour==i]
    print( df_hour[col_title].corr(df_hour[col_title2]))
    mean.append(df_hour[col_title].corr(df_hour[col_title2]))
x=mean[0]
for i in range(1,24):
    x=x+mean[i]

print(x/24)    



###############################################################################################
#HOUR ON ORGINAL
###############################################################################################


df['Hour']=df['Date and Time'].dt.hour
df_hour=df.loc[df.Hour==20]
plot_dataframe(df_hour,'Scatter',col_title,col_title2)


df_hour['Month']=df['Date and Time'].dt.month
df_test=df_hour.loc[df_hour['Month']==11]
plot_dataframe(df_test,'Scatter',col_title,col_title2)



###############################################################################################
#CORRELATION ON ALL
###############################################################################################

print(df[col_title].corr(df[col_title2]))#0.04
correlation=[]
for i in range(0,24):
    df['Hour']=df['Date and Time'].dt.hour
    df_hour=df.loc[df.Hour==i]
    print('Corrolation at %d is:' %i)
    print((df_hour[col_title].corr(df_hour[col_title2])))
    correlation.append((df_hour[col_title].corr(df_hour[col_title2])))

r=range(0,24)
plt.plot(r,correlation)

###############################################################################################
#AUTOCORRELATION
###############################################################################################

autocorrelation=[]
for i in range(1,1800):
    autocorrelation.append(df[col_title].autocorr(lag=i))
r=range(1,1800)
plt.title('Autocorrelation')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.plot(r,autocorrelation,color='red')
    
###############################################################################################
#MAX LOAD VS MAX TEMP
###############################################################################################
maximum=df.resample('d', on='Date and Time').max()
minimum=df.resample('d', on='Date and Time').min()
plot_dataframe(maximum,'Scatter',col_title,col_title2)
plot_dataframe(minimum,'Scatter',col_title,col_title2)
print(maximum[col_title].corr(maximum[col_title2]))
print(minimum[col_title].corr(minimum[col_title2]))


maximum=df_weekday.resample('d', on='Date and Time').max()
minimum=df_weekday.resample('d', on='Date and Time').min()
plot_dataframe(maximum,'Scatter',col_title,col_title2)
plot_dataframe(minimum,'Scatter',col_title,col_title2)
print(maximum[col_title].corr(maximum[col_title2]))
print(minimum[col_title].corr(minimum[col_title2]))


maximum=df_weekend.resample('d', on='Date and Time').max()
minimum=df_weekend.resample('d', on='Date and Time').min()
plot_dataframe(maximum,'Scatter',col_title,col_title2)
plot_dataframe(minimum,'Scatter',col_title,col_title2)
print(maximum[col_title].corr(maximum[col_title2]))
print(minimum[col_title].corr(minimum[col_title2]))







