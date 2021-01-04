from tkinter import *
import os

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)
        
        text = Label(self, text="Face Recognition", fg="Red", font=("Helvetica", 18))
        text.place(x=160,y=20)
        btn1 = Button(self,text='Train Model', bg="red",fg="white",command=training)
        btn1.place(x=205,y=80)
        btn2 = Button(self, text="Run Inference", bg="red", fg="white", command=inference)
        btn2.place(x=200,y=120)
        #text.pack()

def inference():
    os.system('inference.py')

def training():
    os.system('Train_model.py')


root = Tk()
app = Window(root)
root.wm_title("Face Recognition")
root.geometry("480x240")
root.mainloop()