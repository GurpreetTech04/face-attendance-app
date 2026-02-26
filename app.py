import os
import cv2
import numpy as np
import face_recognition
import csv
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import base64
from io import BytesIO
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
IMAGES_FOLDER = 'images'
ATTENDANCE_FILE = 'attendance.csv'
JSON_FILE = 'attendance.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create images folder if it doesn't exist
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Load known faces
def load_known_faces():
    """Load all face images from the images folder and encode them"""
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(IMAGES_FOLDER):
        logger.warning(f"Images folder '{IMAGES_FOLDER}' does not exist")
        return known_face_encodings, known_face_names
    
    for filename in os.listdir(IMAGES_FOLDER):
        if any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            try:
                # Load image
                image_path = os.path.join(IMAGES_FOLDER, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    # Use the first face found
                    known_face_encodings.append(face_encodings[0])
                    # Use filename without extension as name
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
                    logger.info(f"Loaded face: {name}")
                else:
                    logger.warning(f"No face found in {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
    
    logger.info(f"Loaded {len(known_face_names)} known faces")
    return known_face_encodings, known_face_names

# Load faces at startup
known_face_encodings, known_face_names = load_known_faces()

def mark_attendance(name):
    """Mark attendance in CSV and JSON files"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date = datetime.now().strftime("%Y-%m-%d")
    
    # Check if already marked today (simple check - in production use database)
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2 and row[0] == name and row[1].startswith(date):
                    logger.info(f"{name} already marked attendance today")
                    return False
    
    # Mark in CSV
    file_exists = os.path.isfile(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Timestamp'])
        writer.writerow([name, timestamp])
    
    # Mark in JSON
    attendance_data = []
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            try:
                attendance_data = json.load(f)
            except json.JSONDecodeError:
                attendance_data = []
    
    attendance_data.append({
        'name': name,
        'timestamp': timestamp,
        'date': date
    })
    
    with open(JSON_FILE, 'w') as f:
        json.dump(attendance_data, f, indent=2)
    
    logger.info(f"Marked attendance for {name} at {timestamp}")
    return True

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    """Handle captured image from webcam"""
    try:
        # Get image data from request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image data received'}), 400
        
        # Extract base64 image data
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'status': 'error', 'message': 'Failed to decode image'}), 400
        
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Find faces in the image
        face_locations = face_recognition.face_locations(rgb_img)
        
        if not face_locations:
            return jsonify({'status': 'error', 'message': 'No face detected in image'})
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        
        if not face_encodings:
            return jsonify({'status': 'error', 'message': 'Could not encode face'})
        
        # Compare with known faces
        matches = []
        names = []
        
        for face_encoding in face_encodings:
            if known_face_encodings:
                # Compare with all known faces
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(distances)
                
                # Threshold for face matching (lower = stricter)
                if distances[best_match_index] < 0.6:
                    name = known_face_names[best_match_index]
                    names.append(name)
                    matches.append(True)
                else:
                    names.append("Unknown")
                    matches.append(False)
            else:
                names.append("No known faces")
                matches.append(False)
        
        # Mark attendance for the first recognized face
        result = {'status': 'success', 'matches': []}
        
        for i, name in enumerate(names):
            match_result = {
                'name': name,
                'recognized': name not in ["Unknown", "No known faces"]
            }
            
            if match_result['recognized']:
                # Mark attendance
                marked = mark_attendance(name)
                match_result['attendance_marked'] = marked
                match_result['message'] = f"Attendance marked for {name}" if marked else f"{name} already marked today"
            else:
                match_result['attendance_marked'] = False
                match_result['message'] = f"Face not recognized as {name}" if name == "Unknown" else "No reference faces in database"
            
            result['matches'].append(match_result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in capture: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/attendance')
def view_attendance():
    """View attendance records"""
    attendance_data = []
    if os.path.exists(JSON_FILE):
        with open(JSON_FILE, 'r') as f:
            try:
                attendance_data = json.load(f)
            except json.JSONDecodeError:
                attendance_data = []
    
    return jsonify(attendance_data)

@app.route('/download-attendance')
def download_attendance():
    """Download attendance CSV file"""
    if os.path.exists(ATTENDANCE_FILE):
        return send_file(ATTENDANCE_FILE, as_attachment=True)
    else:
        return jsonify({'error': 'No attendance file found'}), 404

@app.route('/faces')
def list_faces():
    """List all known faces"""
    return jsonify({
        'count': len(known_face_names),
        'names': known_face_names
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'faces_loaded': len(known_face_names),
        'attendance_file_exists': os.path.exists(ATTENDANCE_FILE)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)