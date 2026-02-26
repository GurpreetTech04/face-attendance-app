# main.py - Modified for Flask integration
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import json
import time
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use absolute paths
IMAGES_PATH = os.path.join(SCRIPT_DIR, 'images')
ATTENDANCE_FILE = os.path.join(SCRIPT_DIR, 'attendance.csv')
TOLERANCE = 0.5

# Fix Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def log_message(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}")
    # Also try to write to a log file for debugging
    try:
        with open(os.path.join(SCRIPT_DIR, 'debug.log'), 'a') as f:
            f.write(f"[{timestamp}] {msg}\n")
    except:
        pass

def load_encodings():
    """Loads images and generates face encodings."""
    known_encodings = []
    known_names = []

    log_message("="*50)
    log_message("LOADING KNOWN FACES")
    log_message("="*50)
    log_message(f"Images path: {IMAGES_PATH}")
    
    if not os.path.exists(IMAGES_PATH):
        log_message(f"Creating directory: {IMAGES_PATH}")
        os.makedirs(IMAGES_PATH)
        return [], []

    files = os.listdir(IMAGES_PATH)
    log_message(f"Found {len(files)} files")
    
    if not files:
        log_message("No images found!")
        return [], []

    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(IMAGES_PATH, filename)
            try:
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)
                
                if encodings:
                    encoding = encodings[0]
                    name = os.path.splitext(filename)[0]
                    known_encodings.append(encoding)
                    known_names.append(name)
                    log_message(f"  ✅ Loaded: {name}")
                else:
                    log_message(f"  ⚠️ No face in {filename}")
            except Exception as e:
                log_message(f"  ❌ Error: {e}")

    return known_encodings, known_names

def mark_attendance(name):
    """Logs attendance in CSV file."""
    try:
        log_message(f"Marking attendance for: {name}")
        
        # Create file with header if needed
        if not os.path.exists(ATTENDANCE_FILE):
            log_message(f"Creating new file: {ATTENDANCE_FILE}")
            with open(ATTENDANCE_FILE, 'w') as f:
                f.write('Name,Time,Date\n')

        now = datetime.now()
        today_date = now.strftime('%Y-%m-%d')
        current_time = now.strftime('%H:%M:%S')

        # Check if already marked
        already_marked = False
        if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
            with open(ATTENDANCE_FILE, 'r') as f:
                lines = f.readlines()
            
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) >= 3 and parts[0] == name and parts[2] == today_date:
                        already_marked = True
                        break

        if not already_marked:
            with open(ATTENDANCE_FILE, 'a') as f:
                f.write(f'{name},{current_time},{today_date}\n')
            
            log_message(f"  ✅✅ ATTENDANCE MARKED: {name}")
            
            # Verify file
            if os.path.exists(ATTENDANCE_FILE):
                size = os.path.getsize(ATTENDANCE_FILE)
                log_message(f"  CSV size: {size} bytes")
            return True
        else:
            log_message(f"  ⚠️ {name} already marked today")
            return False
            
    except Exception as e:
        log_message(f"  ❌ Error: {e}")
        return False

def main():
    log_message("\n" + "="*50)
    log_message("FACE ATTENDANCE SYSTEM STARTING")
    log_message("="*50)
    log_message(f"Script directory: {SCRIPT_DIR}")
    log_message(f"Images path: {IMAGES_PATH}")
    log_message(f"Attendance file: {ATTENDANCE_FILE}")
    
    # Load faces
    known_encodings, known_names = load_encodings()
    
    if not known_encodings:
        log_message("\n❌ No faces loaded!")
        log_message("Please add photos to 'images' folder")
        input("\nPress Enter to exit...")
        return

    log_message(f"\n✅ Loaded {len(known_names)} faces: {', '.join(known_names)}")
    
    # Initialize camera
    log_message("\n" + "="*50)
    log_message("STARTING CAMERA")
    log_message("="*50)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log_message("❌ Cannot open camera!")
        input("\nPress Enter to exit...")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    log_message("✅ Camera initialized")
    log_message("\nPress 'q' to quit")
    log_message("="*50 + "\n")
    
    process_this_frame = True
    font = cv2.FONT_HERSHEY_DUPLEX
    face_locations = []
    face_names = []
    marked_in_session = set()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log_message("Frame read error")
                break
            
            # Process every other frame
            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, TOLERANCE)
                    
                    name = "Unknown"
                    if True in matches:
                        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            name = known_names[best_match_index]
                            
                            if name not in marked_in_session:
                                if mark_attendance(name):
                                    marked_in_session.add(name)
                    
                    face_names.append(name)
            
            process_this_frame = not process_this_frame
            
            # Draw results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            
            # Status
            status = f"Faces: {len(face_locations)} | Marked: {len(marked_in_session)}"
            cv2.putText(frame, status, (10, 30), font, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Attendance System', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        log_message(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        log_message("\n" + "="*50)
        log_message("SESSION ENDED")
        log_message("="*50)
        log_message(f"Marked today: {len(marked_in_session)}")
        
        # Show CSV content
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, 'r') as f:
                log_message("\nFinal CSV content:")
                log_message(f.read())
        
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()