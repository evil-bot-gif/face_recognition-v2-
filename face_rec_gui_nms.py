from tkinter import *
import os


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

############# Row 0 ###################
        # Project Title Label
        text = Label(self, text="Face Recognition", fg="Red", font=("Helvetica", 18))
        text.pack(pady=3)

############# Row 1 ###################
        # Step 0 instruction label
        i1 = Label(self,text='Enter IP webcam URL for video input src',fg='red',font=('Arial',10))
        i1.pack(pady=0)
        # Label beside src input text field
        src_label = Label(self,text="(Required) Src:",font=("Arial",10))
        src_label.place(x=340,y=65)
        # Input field for user to enter the src 
        s = Entry(self,width=30,font=("Arial",10))
        s.pack(pady=3)
        s.insert(0,"URL e.g.https://192.168.0.238/video")

############# Row 2 ###################
        # Step 1 instruction label 
        text1 = Label(self,text="Enter name of face image dataset to build and press 'Build Image Dataset' button to start collecting images with IP camera devices.",fg='Red',font=("Helvetica",10))
        text1.pack(pady=3)

############# Row 3 ###################
        # Label beside name input text field
        name_label = Label(self,text="(Required) Name:",font=("Arial",10))
        name_label.place(x=345,y=120)
        # Input field for user to enter the path 
        n = Entry(self,width=25,font=("Arial",10))
        n.pack(pady=3)
        n.insert(0,"e.g.xaiver_lim(replace ' ' to '_')")

############# Row 4 ###################        
        # Button to run build_face_dataset script
        btn = Button(self, text="Build Image Dataset", bg='red', fg='white',command= lambda:dataset(n.get(),s.get()))
        btn.pack(pady=3)

############# Row 5 ###################
        # Step 2 instruction label 
        text2 = Label(self,text="Retrain the model with newly created dataset by pressing 'Train Model' button.",fg='Red',font=("Helvetica",10))
        text2.pack(pady=3)

############# Row 6 ###################
        # Button to run train_model script
        btn1 = Button(self,text='Train Model', bg="red",fg="white",command=training)
        btn1.pack(pady=3)

############# Row 7 ###################
        # Step 3 instruction label
        text3 = Label(self,text="Press the 'Run Inference' button to run face recognition.",fg='Red',font=("Helvetica",10))
        text3.pack(pady=3)

############# Row 8 ###################
        # Button to run inference script
        btn2 = Button(self, text="Run Inference", bg="red", fg="white",command=lambda:inference(s.get()))
        btn2.pack(pady=3)
        #text.pack()

def inference(src):
    os.system(f'inference.py -i "{src}"')

def training():
    os.system('Train_model.py')

def dataset(name,src):
    os.system(f'build_face_dataset.py -o known_faces/{name} -i "{src}"')



root = Tk()
app = Window(root)
root.wm_title("Face Recognition")
root.resizable(0,0)
root.geometry("1080x420")
root.mainloop()