import cv2
import os
import numpy as np
from datetime import datetime
from tkinter import Tk, simpledialog
from PIL import Image, ImageDraw, ImageFont
from functions import to_grayscale, resize_image, calculate_histogram, compare_histograms, face_detection

# Setup directories and files
saved_faces_dir = "saved_faces"
attendance_file = "attendance.txt"
template_path = "assets/templates"
os.makedirs(saved_faces_dir, exist_ok=True)

# Load Roboto font
font_path = "assets/Roboto-Regular.ttf"
font_size = 20
font = ImageFont.truetype(font_path, font_size)

# Global variables
mode = "Attendance"  # Modes: "Attendance", "Add Student"
logged_today = set()  # Track students logged during the current session
templates = dict(
    left_eye=to_grayscale(cv2.imread(os.path.join(template_path, 'template_left_eye.jpg'))),
    right_eye=to_grayscale(cv2.imread(os.path.join(template_path, 'template_right_eye.jpg'))),
    mouth=to_grayscale(cv2.imread(os.path.join(template_path, 'template_mouth.jpg')))
)

# Function to load saved faces and calculate their histograms
def load_saved_faces():
    saved_faces = []
    saved_names = []

    for filename in os.listdir(saved_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image in BGR format
            face_image = cv2.imread(os.path.join(saved_faces_dir, filename))
            
            if face_image is not None:
                # Convert to grayscale
                gray_face = to_grayscale(face_image)

                # Resize to a standard size
                resized_face = resize_image(gray_face, 150, 150)

                # Calculate histogram
                face_hist = calculate_histogram(resized_face)

                # Append histogram and name
                saved_faces.append(face_hist)
                saved_names.append(os.path.splitext(filename)[0])  # Use filename (without extension) as name

    return saved_faces, saved_names


# Function to mark app launch in attendance file
def mark_app_launch():
    with open(attendance_file, "a") as file:
        now = datetime.now()
        timestamp = now.strftime("%d-%m-%Y / %H:%M:%S")
        file.write(f"Lesson Attendance - {timestamp}\n")


# Function to log attendance
def log_attendance(name):
    global logged_today
    if name not in logged_today:
        now = datetime.now()
        timestamp = now.strftime("%d-%m-%Y / %H:%M:%S")
        with open(attendance_file, "a") as file:
            file.write(f"- {name} - {timestamp}\n")
        logged_today.add(name)
        print(f"Logged: {name} at {timestamp}")


# Function to compare histograms for face matching
def compare_faces(detected_face_hist, saved_faces, saved_names, threshold=0.8):
    best_match_name = None
    best_match_score = 0  # Default score for the best match

    # Compare the histogram with saved faces
    for face_hist, name in zip(saved_faces, saved_names):
        score = compare_histograms(detected_face_hist, face_hist)
        if score > best_match_score:
            best_match_score = score
            best_match_name = name

    # If the best match score is below the threshold, don't assign a name
    if best_match_score < threshold:
        best_match_name = None

    # Convert score to confidence percentage (score normalized to 0-100%)
    confidence_percentage = round(best_match_score * 100, 2)
    return best_match_name, confidence_percentage



# Function to draw buttons for mode selection
def draw_buttons(frame):
    global mode
    # Button dimensions
    button_width, button_height = 150, 40
    x_center = frame.shape[1] // 2

    # Button positions
    attendance_button = (x_center - 160, 10, x_center - 10, 50)  # Left
    add_student_button = (x_center + 10, 10, x_center + 160, 50)  # Right

    # Draw Attendance button background
    cv2.rectangle(frame, attendance_button[:2], attendance_button[2:], (0, 255, 0) if mode == "Attendance" else (255, 255, 255), -1)
    frame = draw_text_on_frame(frame, "Attendance", (attendance_button[0] + 20, attendance_button[1] + 5), (0, 0, 0))

    # Draw Add Student button background
    cv2.rectangle(frame, add_student_button[:2], add_student_button[2:], (0, 255, 0) if mode == "Add Student" else (255, 255, 255), -1)
    frame = draw_text_on_frame(frame, "Add Student", (add_student_button[0] + 20, add_student_button[1] + 5), (0, 0, 0))

    return attendance_button, add_student_button, frame


# Function to draw Roboto text on the frame using Pillow
def draw_text_on_frame(frame, text, position, color):
    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(frame_pil)


# Function to add a new face
def add_face(detected_face, parent_window_geometry):
    root = Tk()
    root.withdraw()
    # Center the input window relative to the main window
    x, y = [int(i) for i in parent_window_geometry.split("+")[1:]]
    root.geometry(f"+{x + 100}+{y + 100}")
    student_name = simpledialog.askstring("Input", "Enter the student's name:", parent=root)
    root.destroy()

    if student_name:
        file_path = os.path.join(saved_faces_dir, f"{student_name}.jpg")
        cv2.imwrite(file_path, detected_face)
        print(f"Face saved as: {file_path}")
        return True
    return False


# Main program
def main():
    global mode

    # Mark app launch in attendance file
    mark_app_launch()

    # Load saved faces and names
    saved_faces, saved_names = load_saved_faces()

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create the main window
    cv2.namedWindow("Face Detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = to_grayscale(frame)

        # Detect faces
        faces = face_detection(gray, templates)
        detected_face = None
        detection_status = "Processing..."

        for (x, y, w, h) in faces:
            detected_face = gray[y:y+h, x:x+w]

            if len(detected_face.shape) != 2:
                continue

            
            if detected_face.shape[0] == 0  or detected_face.shape[1] == 0:
                continue

            # Resize the detected face to match saved face dimensions
            resized_face = resize_image(detected_face, 150, 150)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

            # Calculate the histogram for the detected face
            detected_face_hist = calculate_histogram(resized_face)

            # Match the detected face with saved faces
            if mode == "Attendance":
                name, confidence = compare_faces(detected_face_hist, saved_faces, saved_names)
                if name:
                    detection_status = f"Detected: {name}"
                    log_attendance(name)
                    # Display name and confidence percentage above the face rectangle
                    frame = draw_text_on_frame(frame, f"{name} ({confidence:.1f}%)", (x, y - 25), (0, 255, 0))
                else:
                    detection_status = "Not Found"

        # Draw buttons
        attendance_button, add_student_button, frame = draw_buttons(frame)

        # Handle window close
        if cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:
            break

        # Show detection status in Attendance mode
        if mode == "Attendance":
            frame = draw_text_on_frame(frame, detection_status, (20, frame.shape[0] - 50), (255, 255, 255))

        # Show "Press A" in Add Student mode when a face is detected
        if mode == "Add Student" and detected_face is not None:
            frame = draw_text_on_frame(frame, "Press 'A' to Add Face", (frame.shape[1] // 2 - 100, 70), (0, 0, 255))

        # Display the frame
        cv2.imshow("Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key to exit
            break
        elif key == ord('a') and mode == "Add Student" and detected_face is not None:
            parent_window_geometry = f"+{cv2.getWindowImageRect('Face Detection')[0]}+{cv2.getWindowImageRect('Face Detection')[1]}"
            if add_face(detected_face, parent_window_geometry):
                saved_faces, saved_names = load_saved_faces()  # Reload saved faces after adding

        # Check mouse clicks only if the window exists
        if cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) >= 1:
            def mouse_callback(event, x, y, flags, param):
                global mode
                if event == cv2.EVENT_LBUTTONDOWN:
                    if attendance_button[0] < x < attendance_button[2] and attendance_button[1] < y < attendance_button[3]:
                        mode = "Attendance"
                    elif add_student_button[0] < x < add_student_button[2] and add_student_button[1] < y < add_student_button[3]:
                        mode = "Add Student"

            cv2.setMouseCallback("Face Detection", mouse_callback)

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    if not os.path.exists(attendance_file):
        open(attendance_file, 'w').close()
    main()
