from tkinter import *
from PIL import ImageTk, Image
import os


class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=1)

        # Project Title Label
        text = Label(self, text="Face Recognition GUI", fg="#37474f", font=("Helvetica", 18))
        text.pack(pady=3)

        # Image icon for face rec gui

        image1 = Image.open('face_rec_icon.png')
        image1 = image1.resize((50,50),Image.ANTIALIAS)
        i = ImageTk.PhotoImage(image1)
        icon = Label(image=i)
        icon.image = i
        icon.place(x=550,y=0)

        # Instruction label for entering video source
        i1 = Label(self,text='Enter IP webcam URL for video input src',fg='#37474f',font=('Arial',10))
        i1.pack(pady=0)
        # Src label beside input text field
        src_label = Label(self,text="(Required) Src:",font=("Arial",10),fg="#37474f")
        src_label.place(x=230,y=65)
        # Input field for user to enter the src 
        s = Entry(self,width=30,font=("Arial",10))
        s.pack(pady=3)
        s.insert(0,"http://151.192.128.132:18888/video")

        # Instruction label for building dataset
        text1 = Label(self,text="Enter name of face image dataset to build and press 'Build Image Dataset' button to start collecting images with IP camera devices.",fg='#37474f',font=("Helvetica",10))
        text1.pack(pady=3)

        # Name label beside input text field
        name_label = Label(self,text="(Required) Name:",font=("Arial",10),fg="#37474f")
        name_label.place(x=230,y=120)
        # Input field for user to enter name of dataset folder created 
        n = Entry(self,width=25,font=("Arial",10))
        n.pack(pady=3)
        n.insert(0,"xavier lim")
       
        # Button to run build_face_dataset script
        btn = Button(self, text="Build Image Dataset", bg='#37474f', fg='#00e5ff',bd=-2,command= lambda:dataset(n.get(),s.get()))
        btn.pack(pady=3)

        # Instruction label for generating encodings
        text2 = Label(self,text="Generate face encodings from new dataset by pressing 'Generate encodings' button.",fg='#37474f',font=("Helvetica",10))
        text2.pack(pady=3)

        # Button to run Generate_encodings script
        btn1 = Button(self,text='Generate encodings', bg="#37474f",fg="#00e5ff",bd=-2,command=generate)
        btn1.pack(pady=3)


        # Instruction label for running inference
        text3 = Label(self,text="Press the 'Run Inference' button to run face recognition.",fg='#37474f',font=("Helvetica",10))
        text3.pack(pady=3)


        # Button to run inference script
        btn2 = Button(self, text="Run Inference", bg="#37474f", fg="#00e5ff",bd=-2,command=lambda:inference(s.get()))
        btn2.pack(pady=3)

def inference(src):
    os.system(f'python3 inference.py -i "{src}"')

def generate():
    os.system('python3 generate_encodings.py')

def dataset(name,src):
    os.system(f'python3 build_face_dataset.py -o known_faces/"{name}" -i "{src}"')



root = Tk()
app = Window(root)
root.wm_title("Face Recognition")
root.resizable(0,0)
root.geometry("860x350+800+0")
root.mainloop()