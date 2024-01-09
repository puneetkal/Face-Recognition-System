import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import os

# Global variables
dataset_path = 'Dataset'
trainer_path = 'trainer/trainer.yml'
cascade_path = 'haarcascade_frontalface_default.xml'
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cascade_path)
font = cv2.FONT_HERSHEY_SIMPLEX

# GUI App
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        self.create_widgets()

    def create_widgets(self):
        # Label for displaying video feed
        self.video_label = ttk.Label(self.root)
        self.video_label.grid(row=0, column=0, columnspan=2)

        # Buttons
        self.capture_button = ttk.Button(self.root, text="Capture Faces", command=self.capture_faces)
        self.capture_button.grid(row=1, column=0, padx=5, pady=5)

        self.train_button = ttk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_button.grid(row=1, column=1, padx=5, pady=5)

        self.recognize_button = ttk.Button(self.root, text="Recognize Faces", command=self.recognize_faces)
        self.recognize_button.grid(row=2, column=0, columnspan=2, pady=5)

    def capture_faces(self):
        capture_faces_gui = CaptureFacesGUI(self.root)

    def train_model(self):
        faces, ids, names = self.get_images_and_labels(dataset_path)
        recognizer.train(faces, np.array(ids))
        recognizer.write(trainer_path)
        print(f"\n [INFO] {len(np.unique(ids))} faces trained.")
        return names

    def recognize_faces(self):
        names = self.train_model()  # Get names from the training
        recognize_faces_gui = RecognizeFacesGUI(self.root, names)

    def get_images_and_labels(self, path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []
        ids = []
        names = []

        for image_path in image_paths:
            pil_img = Image.open(image_path).convert('L')
            img_numpy = np.array(pil_img, 'uint8')

            id = int(os.path.split(image_path)[-1].split(".")[1])
            name = os.path.split(image_path)[-1].split(".")[2]

            faces = face_cascade.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
                names.append(name)

        return face_samples, ids, names


class CaptureFacesGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Capture Faces")

        self.name_entry_label = ttk.Label(self.root, text="Enter name:")
        self.name_entry_label.grid(row=0, column=0, padx=5, pady=5)

        self.name_entry = ttk.Entry(self.root)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)

        self.capture_button = ttk.Button(self.root, text="Capture", command=self.capture_face)
        self.capture_button.grid(row=1, column=0, columnspan=2, pady=5)

    def capture_face(self):
        name = self.name_entry.get()
        if not name:
            return

        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)

        face_detector = cv2.CascadeClassifier(cascade_path)

        print("\n [INFO] Initializing face capture. Look at the camera and wait...")

        count = 0

        # Ensure the dataset folder exists
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                count += 1

                cv2.putText(img, name, (x+5, y-5), font, 1, (255, 255, 255), 2)

                cv2.imshow('image', img)

                # Save the captured face image
                face_img = gray[y:y+h, x:x+w]
                img_name = f"subject.{count}.{name}.png"
                img_path = os.path.join(dataset_path, img_name)
                cv2.imwrite(img_path, face_img)

            k = cv2.waitKey(100) & 0xff
            if k == 27 or count >= 150:
                break

        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()

class RecognizeFacesGUI:
    def __init__(self, root, names):
        self.root = root
        self.root.title("Recognize Faces")
        self.names = names  # Store names for later use

        self.recognize_faces()

    def recognize_faces(self):
        cam = cv2.VideoCapture(0)
        cam.set(3, 640)
        cam.set(4, 480)

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(30, 30),
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 100:
                    name = self.names[id]  # Use self.names to access the names
                    confidence_text = f"  {round(confidence)}%"
                else:
                    name = "unknown"
                    confidence_text = f"  {round(confidence)}%"

                cv2.putText(img, name, (x+5, y-5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, confidence_text, (x+5, y+h-5), font, 1, (255, 255, 0), 1)

            cv2.imshow('camera', img)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
