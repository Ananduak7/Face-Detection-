import face_recognition
import cv2
import csv
from datetime import datetime

# Load known face encodings and names
known_face_encodings = []
known_faces_names = []

# Define paths to known faces
known_faces_paths = {
    "AYESWARYA LAKSHMI": "D:/FACE/mal.jpg",
    "ANANDAKRISHNAN A_21BCE1682": "D:/FACE/IMG20240319150838.jpg",
    "ASWIN MENON_21BMV1076": "D:/FACE/IMG20240319150753.jpg",
    "AYDIN ABDUL_21CE1076": "D:/FACE/WhatsApp Image 2024-03-19 at 21.14.58_5067337d.jpg",
    "VASIREDDY SURYA_21BCE": "D:/FACE/V.jpg",
    "Nikhil Shinde_21BCE1743": "D:\FACE\WhatsApp Image 2024-04-16 at 16.37.53_f9c1759f.jpg"
}

# Load known face encodings
for name, path in known_faces_paths.items():
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_faces_names.append(name)

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Initialize CSV writer
current_date = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"{current_date}.csv"
f = open(csv_filename, 'w', newline='')
lnwriter = csv.writer(f)

# Initialize variable to track whether any known face is detected
known_face_detected = False

while True:
    # Read frame from video capture
    ret, frame = video_capture.read()

    # Convert frame to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find face locations
    face_locations = face_recognition.face_locations(rgb_frame)

    # Iterate over detected faces
    for top, right, bottom, left in face_locations:
        # Extract the detected face
        face_image = frame[top:bottom, left:right]

        # Convert face image to RGB format
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Encode face
        face_encoding = face_recognition.face_encodings(face_image_rgb)

        # Compare face encoding with known face encodings
        for encoding in face_encoding:
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            if True in matches:
                known_face_detected = True

                # Get the name of the detected face
                name = known_faces_names[matches.index(True)]

                # Write the name to CSV
                current_time = datetime.now().strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])

                # Draw a rectangle and text around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close CSV file
video_capture.release()
cv2.destroyAllWindows()
f.close()
