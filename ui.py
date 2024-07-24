import face_recognition
import cv2
import csv
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

# Function to capture and save image with entered name
def capture_and_save():
    def capture_frame():
        nonlocal frame_saved
        ret, frame = capture.read()
        if ret:
            cv2.imshow("Capture", frame)
            frame_saved = frame.copy()

    def save_image():
        nonlocal frame_saved
        global known_face_encodings
        global known_faces_names

        name = name_entry.get()
        if name.strip() == "":
            messagebox.showerror("Error", "Please enter a name.")
            return

        if frame_saved is not None:
            cv2.imwrite(f"{name}.jpg", frame_saved)
            messagebox.showinfo("Success", f"Image saved as {name}.jpg")

            # Reload known faces list
            image = face_recognition.load_image_file(f"{name}.jpg")
            encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(encoding)
            known_faces_names.append(name)

    # Create new window for image capture
    new_window = tk.Toplevel(root)
    new_window.title("Capture Image")

    # Label and entry for name input
    name_label = tk.Label(new_window, text="Enter Name:")
    name_label.pack()
    name_entry = tk.Entry(new_window)
    name_entry.pack()

    # Initialize video capture
    capture = cv2.VideoCapture(0)
    frame_saved = None

    # Button to capture image
    capture_button = tk.Button(new_window, text="Capture Frame", command=capture_frame)
    capture_button.pack()

    # Button to save image
    save_button = tk.Button(new_window, text="Save Image", command=save_image)
    save_button.pack()

    # Function to close window when finished
    def on_closing():
        capture.release()
        cv2.destroyAllWindows()
        new_window.destroy()

    new_window.protocol("WM_DELETE_WINDOW", on_closing)

# Function to reload known faces list
def reload_known_faces():
    global known_face_encodings
    global known_faces_names

    # Clear existing known face encodings and names
    known_face_encodings.clear()
    known_faces_names.clear()

    # Load known face encodings
    for name, path in known_faces_paths.items():
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_faces_names.append(name)

# Function to perform face detection on updated data
def detect_faces():
    # Initialize variable to track whether any known face is detected
    known_face_detected = False

    while True:
        # Read frame from video capture
        ret, frame = video_capture.read()

        # Convert frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_frame)

        # Encode faces in the current frame
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Iterate over detected faces
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            # Compare face encoding with known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                known_face_detected = True

                # Get the name of the detected face
                name = known_faces_names[matches.index(True)]

                # Draw a rectangle and text around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load known face encodings and names
known_face_encodings = []
known_faces_names = []

# Define paths to known faces
known_faces_paths ={
}

# Load known face encodings
reload_known_faces()

# GUI
root = tk.Tk()
root.title("Face Detection")
root.geometry("300x200")

# Button to open image capture window
capture_button = tk.Button(root, text="Save Image", command=capture_and_save)
capture_button.pack()

# Button to start face detection
detect_button = tk.Button(root, text="Detect Faces", command=detect_faces)
detect_button.pack()

root.mainloop()

# Release video capture
video_capture.release()
cv2.destroyAllWindows()
