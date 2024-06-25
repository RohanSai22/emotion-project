import threading
import webbrowser
from tkinter import *    # GUI
from tkinter import ttk  # GUI
from PIL import Image, ImageTk    # pip install pillow for image
from tkinter import messagebox
import requests
import cv2
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image


def fetch_joke():
    try:
        response = requests.get("https://official-joke-api.appspot.com/random_joke")
        if response.status_code == 200:
            joke = response.json()
            return f"{joke['setup']} - {joke['punchline']}"
        else:
            return "No joke available at the moment."
    except:
        return "Failed to fetch joke."


def fetch_songs(genre):
    try:
        api_key = 'f9b011c9'  # Replace with your actual API key
        response = requests.get(f"https://api.jamendo.com/v3.0/tracks/?client_id={api_key}&format=json&limit=1&tags={genre}")
        if response.status_code == 200:
            song = response.json()
            return song['results'][0]['audio']
        else:
            return "No songs available at the moment."
    except:
        return "Failed to fetch songs."


def get_suggestion(emotion):
    suggestions = {
        'Happy': [fetch_songs('Pop'), fetch_joke()],
        'Sad': [fetch_songs('Classical'), fetch_joke()],
        'Angry': [fetch_songs('Jazz'), 'Practice deep breathing'],
        'Surprise': [fetch_songs('Rock'), 'Watch a thriller movie'],
        'Neutral': [fetch_songs('Any'), 'Take a short walk'],
        'Fear': [fetch_songs('Ambient'), 'Try meditation'],
        'Disgust': [fetch_songs('Soul'), 'Watch a funny video']
    }
    return suggestions.get(emotion, ["No suggestion available"])[0]


def open_url(url):
    webbrowser.open_new(url)


def call_GUI1():  # for Static Button
    win2 = Toplevel(root)
    Second_Window(win2)
    return


def call_GUI2():  # For Real Time Button
    win3 = Toplevel(root)
    Third_Window(win3)
    return


class First_Window:
    def __init__(self, root):
        self.root = root
        self.root.title("Main")

        screen_width = root.winfo_screenwidth()  # Fetching screen width
        screen_height = root.winfo_screenheight()  # Fetching screen height
        root.geometry(f'{screen_width}x{screen_height - 100}')  # Geometry For main window and -100 so that it will not loss any part of main window

        img1 = Image.open("images/2-AI-invades-automobile-industry-in-2019.jpeg")  # AI Hand Image
        img1 = img1.resize((1530, 800), Image.ANTIALIAS)
        self.photoImg1 = ImageTk.PhotoImage(img1)
        bg_lbl = Label(self.root, image=self.photoImg1)
        bg_lbl.place(x=0, y=0, width=1530, height=800)

        title = Label(bg_lbl, text="Emotion Detection ", font=("times new roman", 35, "bold"), bg="white", fg="red")  # White strip of main
        title.place(x=0, y=120, width=1550, height=45)

        myname = Label(self.root, text="Developed By: team impossible", fg="black", bg="white",
                       font=("times new roman", 18, "bold"))  # Developed by
        myname.place(x=0, y=0)

        img10 = Image.open("images/facial-recognition_0.jpg")  # Image displaying of facial recognition
        img10 = img10.resize((500, 120), Image.ANTIALIAS)
        self.photoImg10 = ImageTk.PhotoImage(img10)
        bg_lbl1 = Label(bg_lbl, image=self.photoImg10)
        bg_lbl1.place(x=0, y=0, width=500, height=120)

        img11 = Image.open("images/facialrecognition.png")
        img11 = img11.resize((500, 120), Image.ANTIALIAS)
        self.photoImg11 = ImageTk.PhotoImage(img11)
        bg_lbl22 = Label(bg_lbl, image=self.photoImg11)
        bg_lbl22.place(x=500, y=0, width=500, height=120)

        img13 = Image.open("images/smart-attendance.jpg")
        img13 = img13.resize((550, 120), Image.ANTIALIAS)
        self.photoImg13 = ImageTk.PhotoImage(img13)
        bg_lbl12 = Label(bg_lbl, image=self.photoImg13)
        bg_lbl12.place(x=1000, y=0, width=550, height=120)

        frame = Frame(self.root, bg="black")
        frame.place(x=610, y=200, width=340, height=430)

        img1 = Image.open("images/LoginIconAppl.png")
        img1 = img1.resize((90, 90), Image.ANTIALIAS)
        self.photoimage1 = ImageTk.PhotoImage(img1)
        lblimg1 = Label(image=self.photoimage1, bg="black", borderwidth=0)
        lblimg1.place(x=730, y=200, width=90, height=90)

        get_str = Label(frame, text="Get Started", font=("times new roman", 20, "bold"), fg="white", bg="black")
        get_str.place(x=95, y=85)

        # LoginButton
        btn_login = Button(frame, text="STATIC", borderwidth=5, relief=RAISED, command=call_GUI1, cursor="hand2",
                           font=("times new roman", 20, "bold"), fg="white", bg="red", activebackground="#B00857")
        btn_login.place(x=75, y=160, width=200, height=50)

        btn_login1 = Button(frame, text="REAL TIME", borderwidth=5, relief=RAISED, command=call_GUI2, cursor="hand2",
                            font=("times new roman", 20, "bold"), fg="white", bg="red", activebackground="#B00857")
        btn_login1.place(x=75, y=270, width=200, height=50)


class Second_Window:
    def __init__(self, root):
        self.root = root
        self.root.title("Static")
        screen_width = root.winfo_screenwidth()  # Fetching screen width
        screen_height = root.winfo_screenheight()  # Fetching screen height
        root.geometry(f'{screen_width}x{screen_height - 100}')

        frame = Frame(self.root, bg="black")
        frame.place(x=610, y=200, width=340, height=430)

        self.var_SecurityA = StringVar()
        entry1 = Entry(frame, textvariable=self.var_SecurityA, bd=5, relief=GROOVE, width=20, font=("times new roman", 18))
        entry1.grid(row=7, column=1, padx=20, pady=3)

        def getLink():
            x1 = entry1.get()
            emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

            # Load json and create model
            json_file = open('model/emotion_model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            emotion_model = model_from_json(loaded_model_json)

            # Load weights into new model
            emotion_model.load_weights("model/emotion_model.h5")
            print("Loaded model from disk")

            # Pass here your video path
            cap = cv2.VideoCapture(x1)

            while True:
                # Find haar cascade to draw bounding box around face
                ret, frame = cap.read()
                frame = cv2.resize(frame, (1280, 720))
                if not ret:
                    break
                face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces available on camera
                num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

                # Take each face available on the camera and Preprocess it
                for (x, y, w, h) in num_faces:
                    cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
                    roi_gray_frame = gray_frame[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                    # Predict the emotions
                    emotion_prediction = emotion_model.predict(cropped_img)
                    maxindex = int(np.argmax(emotion_prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('Emotion Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()

        button1 = Button(frame, text="Get Link", borderwidth=5, relief=RAISED, command=getLink, cursor="hand2",
                         font=("times new roman", 20, "bold"), fg="white", bg="red", activebackground="#B00857")
        button1.place(x=75, y=160, width=200, height=50)

        button2 = Button(frame, text="QUIT", borderwidth=5, relief=RAISED, command=root.destroy, cursor="hand2",
                         font=("times new roman", 20, "bold"), fg="white", bg="red", activebackground="#B00857")
        button2.place(x=75, y=270, width=200, height=50)


class Third_Window:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time")

        screen_width = root.winfo_screenwidth()  # Fetching screen width
        screen_height = root.winfo_screenheight()  # Fetching screen height
        root.geometry(f'{screen_width}x{screen_height - 100}')

        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.classifier = load_model('model.h5')

        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

        self.frame = Frame(self.root, bg="black")
        self.frame.place(x=610, y=200, width=340, height=430)

        self.label = Label(self.frame, text="Emotion: ", font=("times new roman", 18, "bold"), fg="white", bg="black")
        self.label.place(x=10, y=10)

        self.suggestion_label = Label(self.frame, text="Suggestion: ", font=("times new roman", 18, "bold"), fg="white", bg="black")
        self.suggestion_label.place(x=10, y=80)

        self.button = Button(self.frame, text="Open Music Site", command=lambda: open_url(self.suggestion_url), state=DISABLED, font=("times new roman", 18, "bold"), fg="white", bg="red", activebackground="#B00857")
        self.button.place(x=10, y=150)

        self.thread = threading.Thread(target=self.detect_emotions)
        self.thread.start()

    def detect_emotions(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            labels = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = self.classifier.predict(roi)[0]
                    label = self.emotion_labels[prediction.argmax()]
                    suggestion = get_suggestion(label)
                    self.update_gui(label, suggestion)
                else:
                    self.update_gui('No Faces', '')

            cv2.imshow('Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def update_gui(self, emotion, suggestion):
        self.label.config(text=f"Emotion: {emotion}")
        self.suggestion_label.config(text=f"Suggestion: {suggestion}")
        if "http" in suggestion:
            self.suggestion_url = suggestion
            self.button.config(state=NORMAL)
        else:
            self.button.config(state=DISABLED)


if __name__ == "__main__":
    root = Tk()
    app = First_Window(root)  # Corrected class name
    root.mainloop()
