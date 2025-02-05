import cv2
import os
import datetime
from tkinter import Tk, Label, Button
from threading import Thread
video_capture = None
running = False
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
output_dir = "captured_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def detect_and_capture():
    """Detect faces and smiles, capture images, and display the live feed."""
    global video_capture, running
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open camera.")
        return

    while running:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=15, minSize=(25, 25))
            if len(smiles) > 0:
                print("Smile detected, capturing image...")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = os.path.join(output_dir, f'smile_{timestamp}.jpg')
                cv2.imwrite(image_filename, frame)
                cv2.putText(frame, "Smile Captured!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow("Smile Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

def start_camera():
    """Start the camera and smile detection."""
    global running
    if not running:
        running = True
        Thread(target=detect_and_capture).start()

def stop_camera():
    """Stop the camera and smile detection."""
    global running, video_capture
    running = False
    if video_capture is not None:
        video_capture.release()
    cv2.destroyAllWindows()

def create_gui():
    """Create the GUI for the application."""
    root = Tk()
    root.title("Smile Detector")
    root.geometry("300x200")
    label = Label(root, text="Smile Detector", font=("Helvetica", 16))
    label.pack(pady=10)

    start_button = Button(root, text="Start Camera", command=start_camera, width=15)
    start_button.pack(pady=10)

    stop_button = Button(root, text="Stop Camera", command=stop_camera, width=15)
    stop_button.pack(pady=10)

    exit_button = Button(root, text="Exit", command=root.destroy, width=15)
    exit_button.pack(pady=10)

    root.mainloop()

create_gui()
