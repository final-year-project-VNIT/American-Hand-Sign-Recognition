# Importing Libraries

import numpy as np

import cv2
import os, sys
import time
import operator

from tkinter import Frame
from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

from hunspell import Hunspell
import enchant

from keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

#Application :

class Application:

    def __init__(self):

        self.hs = Hunspell('en_US')
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        self.json_file = open("Models\model-bw.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()

        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("Models\model-bw.h5")

        self.json_file_dru = open("Models\model-dru.json" , "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()

        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights("Models\model-dru.h5")

        self.json_file_kdi = open("Models\model-kdi.json" , "r")
        self.model_json_kdi = self.json_file_kdi.read()
        self.json_file_kdi.close()
        self.loaded_model_kdi = model_from_json(self.model_json_kdi)
        self.loaded_model_kdi.load_weights("Models\model-kdi.h5")

        self.json_file_smn = open("Models\model-smn.json" , "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()

        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights("Models\model-smn.h5")

        self.count = {}
        self.count['blank'] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
          self.count[i] = 0
        
        print("Starting the Interface")

        self.root = tk.Tk()
        frame1 = Frame(self.root, highlightbackground="blue", highlightthickness=5,width=800, height=800, bd= 0)
        frame1.pack()

        self.root.title("ASL Recognition")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("800x650")

        self.window = tk.Label(self.root)
        self.window.place(x = 100, y = 10, width = 580, height = 580)
        
        self.window2 = tk.Label(self.root) # initialize image panel
        self.window2.place(x = 400, y = 65, width = 275, height = 275)

        self.text = tk.Label(self.root)
        self.text.place(x = 100, y = 5)
        self.text.config(text = "ASL Recognition", font = ("Arial", 30, "bold"))

        self.window3 = tk.Label(self.root) # Current Symbol
        self.window3.place(x = 500, y = 540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x = 100, y = 540)
        self.T1.config(text = "Prediction :", font = ("Arial", 30, "bold"))

        self.window4 = tk.Label(self.root) # Word
        self.window4.place(x = 230, y = 595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x = 100,y = 595)
        self.T2.config(text = "Word :", font = ("Arial", 30, "bold"))


        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.video_processing()


    def video_processing(self):
        ok, frame = self.vs.read()

        if ok:
            cv2image = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0) ,1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image = self.current_image)

            self.window.imgtk = imgtk
            self.window.config(image = imgtk)

            cv2image = cv2image[y1 : y2, x1 : x2]

            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            self.predict(res)

            self.current_image2 = Image.fromarray(res)

            imgtk = ImageTk.PhotoImage(image = self.current_image2)

            self.window2.imgtk = imgtk
            self.window2.config(image = imgtk)

            self.window3.config(text = self.current_symbol,fg = "red", font = ("Arial", 30))

            self.window4.config(text = self.word, font = ("Arial", 30))


            predicts = self.hs.suggest(self.word)
            
       

        self.root.after(5, self.video_processing)

    def predict(self, test_image):

        test_image = cv2.resize(test_image, (128, 128))

        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))


        result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))

        result_kdi = self.loaded_model_kdi.predict(test_image.reshape(1 , 128 , 128 , 1))

        result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))

        prediction = {}

        prediction['blank'] = result[0][0]

        inde = 1

        for i in ascii_uppercase:

            prediction[i] = result[0][inde]

            inde += 1

        #LAYER 1

        prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

        self.current_symbol = prediction[0][0]


        #LAYER 2

        if(self.current_symbol == 'D' or self.current_symbol == 'R' or self.current_symbol == 'U'):

        	prediction = {}

        	prediction['D'] = result_dru[0][0]
        	prediction['R'] = result_dru[0][1]
        	prediction['U'] = result_dru[0][2]

        	prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

        	self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'D' or self.current_symbol == 'I' or self.current_symbol == 'K' ):

        	prediction = {}

        	prediction['D'] = result_kdi[0][0]
        	prediction['I'] = result_kdi[0][1]
        	prediction['K'] = result_kdi[0][2]

        	prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

        	self.current_symbol = prediction[0][0]

        if(self.current_symbol == 'M' or self.current_symbol == 'N' or self.current_symbol == 'S'):

        	prediction1 = {}

        	prediction1['M'] = result_smn[0][0]
        	prediction1['N'] = result_smn[0][1]
        	prediction1['S'] = result_smn[0][2]

        	prediction1 = sorted(prediction1.items(), key = operator.itemgetter(1), reverse = True)

        	if(prediction1[0][0] == 'S'):

        		self.current_symbol = prediction1[0][0]

        	else:

        		self.current_symbol = prediction[0][0]
        
        if(self.current_symbol == 'blank'):

            for i in ascii_uppercase:
                self.count[i] = 0

        self.count[self.current_symbol] += 1

        if(self.count[self.current_symbol] > 25):

            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = self.count[self.current_symbol] - self.count[i]

                if tmp < 0:
                    tmp *= -1

                if tmp <= 20:
                    self.count['blank'] = 0

                    for i in ascii_uppercase:
                        self.count[i] = 0
                    return

            self.count['blank'] = 0


            if self.current_symbol == 'blank':

                if self.blank_flag == 0:
                    self.blank_flag = 1


                    self.word = ""

            else:

                self.blank_flag = 0

                self.word += self.current_symbol
            
            for i in ascii_uppercase:
                self.count[i] = 0

    def action1(self):

    	predicts = self.hs.suggest(self.word)

    	if(len(predicts) > 0):

            self.word = ""

    def action2(self):

    	predicts = self.hs.suggest(self.word)

    	if(len(predicts) > 1):
            self.word = ""


    def action3(self):

    	predicts = self.hs.suggest(self.word)

    	if(len(predicts) > 2):
            self.word = ""


    def action4(self):

    	predicts = self.hs.suggest(self.word)

    	if(len(predicts) > 3):
            self.word = ""


    def action5(self):

    	predicts = self.hs.suggest(self.word)

    	if(len(predicts) > 4):
            self.word = ""
            
    def destructor(self):

        print("Closing Application...")

        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()
    
print("Starting Application...")

(Application()).root.mainloop()