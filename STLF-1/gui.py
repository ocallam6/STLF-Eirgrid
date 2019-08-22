'''
Beginning of a GUI that doesn't really work well but with a few tweaks could be made work
'''




from main import mains
import tkinter
from tkinter import *


main=tkinter.Tk()
main.title("EirGrid Short Term Load Forecasting")
main.geometry('600x400')
main.configure(background="#bce6ff")
#-------------------------------------------------------------
#Opening Page
#-------------------------------------------------------------
labl=tkinter.Label(main,text="Day Ahead Load Forecast \n Predictions",bg="#bce6ff",font=("Arial Bold",14))
labl.grid(column=1,row=0,rowspan=1)
lbl=tkinter.Label(main,text="Enter Start Date: \n \n dd-mon-year",bg="#bce6ff",pady=20)
lbl.grid(column=0,row=1)
lble=tkinter.Label(main,text="Enter Prediction Date:\n \n dd-mon-year",bg="#bce6ff",pady=20)
lble.grid(column=0,row=2)


txt=tkinter.Entry(main, width=20)
txt.focus()
txt.grid(column=1,row=1)
txt1=tkinter.Entry(main, width=20)
txt1.grid(column=1,row=2)

'''
loc = [
    ("Ireland"),
    ("Northern Ireland")
]

#Location dropbox

selec=tkinter.Label(main, text="Location:",pady = 20,bg="#bce6ff")
selec.grid(column=0,row=3)
variable1=tkinter.StringVar(main)
variable1.set(loc[0])
w=tkinter.OptionMenu(main,variable1,*loc,)
w.grid(column=1,row=3)
'''

loc = [
    ("Yes"),
    ("No")
]

#Location dropbox

selec=tkinter.Label(main, text="New Train?",pady = 20,bg="#bce6ff")
selec.grid(column=0,row=4)
variable2=tkinter.StringVar(main)
variable2.set(loc[0])
w=tkinter.OptionMenu(main,variable2,*loc,)
w.grid(column=1,row=4)

def clicked():
    start_date=txt.get()
    end_date=txt1.get()
    #loc=loc_item(variable1.get())
    train=variable2.get()
    mains(start_date,end_date,train)

btn=tkinter.Button(main, text='Run Report',command=clicked,)#,bg="orange,fg="red")
btn.grid(column=1,row=5)











main.mainloop()
