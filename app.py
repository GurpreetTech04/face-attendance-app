# app.py - Updated with better path handling
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import json
from flask import Flask, render_template, jsonify, request, send_file
import time
import subprocess
import signal
import psutil
import pandas as pd
import sys

app = Flask(__name__)

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_TO_RUN = os.path.join(BASE_DIR, 'main.py')
PYTHON_EXECUTABLE = sys.executable
IMAGES_PATH = os.path.join(BASE_DIR, 'images')
ATTENDANCE_FILE = os.path.join(BASE_DIR, 'attendance.csv')

# Global variable
BACKGROUND_PROCESS = None

# Ensure directories exist
os.makedirs(IMAGES_PATH, exist_ok=True)

def find_attendance_process():
    """Finds the PID of the running attendance script"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline_args = proc.info.get('cmdline')
            if isinstance(cmdline_args, list):
                cmd_str = ' '.join(cmdline_args).lower()
                if 'main.py' in cmd_str and 'python' in cmd_str:
                    return proc.pid
        except:
            pass
    return None

def read_attendance_data():
    """Read attendance data from CSV safely"""
    try:
        if not os.path.exists(ATTENDANCE_FILE):
            return []
        
        if os.path.getsize(ATTENDANCE_FILE) == 0:
            return []
        
        df = pd.read_csv(ATTENDANCE_FILE)
        return df.to_dict('records') if not df.empty else []
    except Exception as e:
        print(f"Error reading attendance: {e}")
        return []

def get_students_list():
    """Get list of registered students"""
    students = []
    if os.path.exists(IMAGES_PATH):
        for file in os.listdir(IMAGES_PATH):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                students.append(os.path.splitext(file)[0])
    return students

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_attendance', methods=['POST'])
def start_attendance():
    global BACKGROUND_PROCESS
    
    # Check if already running
    existing_pid = find_attendance_process()
    if existing_pid:
        return jsonify({
            'status': 'already_running', 
            'pid': existing_pid,
            'message': 'System already running'
        })

    try:
        print("\n" + "="*50)
        print("STARTING ATTENDANCE SYSTEM")
        print("="*50)
        print(f"Base dir: {BASE_DIR}")
        print(f"Script: {SCRIPT_TO_RUN}")
        print(f"Images: {IMAGES_PATH}")
        print(f"CSV: {ATTENDANCE_FILE}")
        
        command = [PYTHON_EXECUTABLE, SCRIPT_TO_RUN]
        env = os.environ.copy()
        
        if sys.platform == 'win32':
            BACKGROUND_PROCESS = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=BASE_DIR  # Important: Set working directory
            )
        else:
            BACKGROUND_PROCESS = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=BASE_DIR
            )
        
        print(f"Process started with PID: {BACKGROUND_PROCESS.pid}")
        print("="*50 + "\n")
        
        time.sleep(3)
        
        return jsonify({
            'status': 'running', 
            'pid': BACKGROUND_PROCESS.pid,
            'message': 'System started'
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop_attendance', methods=['POST'])
def stop_attendance():
    global BACKGROUND_PROCESS
    
    pid_to_stop = None
    if BACKGROUND_PROCESS and BACKGROUND_PROCESS.poll() is None:
        pid_to_stop = BACKGROUND_PROCESS.pid
    
    if not pid_to_stop:
        pid_to_stop = find_attendance_process()
    
    if pid_to_stop:
        try:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/PID', str(pid_to_stop)], 
                             capture_output=True)
            else:
                os.kill(pid_to_stop, signal.SIGTERM)
            
            BACKGROUND_PROCESS = None
            return jsonify({'status': 'stopped', 'message': 'System stopped'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    else:
        return jsonify({'status': 'not_running'})

@app.route('/api/status', methods=['GET'])
def status_attendance():
    if BACKGROUND_PROCESS and BACKGROUND_PROCESS.poll() is None:
        return jsonify({'status': 'running', 'pid': BACKGROUND_PROCESS.pid})
    
    pid = find_attendance_process()
    if pid:
        return jsonify({'status': 'running', 'pid': pid})
    
    return jsonify({'status': 'stopped'})

@app.route('/api/attendance/today', methods=['GET'])
def get_today_attendance():
    try:
        data = read_attendance_data()
        today = datetime.now().strftime('%Y-%m-%d')
        today_data = [r for r in data if r.get('Date') == today]
        return jsonify({'status': 'success', 'data': today_data, 'count': len(today_data)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/attendance/all', methods=['GET'])
def get_all_attendance():
    try:
        data = read_attendance_data()
        return jsonify({'status': 'success', 'data': data, 'count': len(data)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/students', methods=['GET'])
def get_students():
    try:
        students = get_students_list()
        return jsonify({'status': 'success', 'data': students, 'count': len(students)})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        data = read_attendance_data()
        students = get_students_list()
        
        if not data:
            return jsonify({
                'total_students': len(students),
                'total_records': 0,
                'today_count': 0,
                'unique_students': 0
            })
        
        df = pd.DataFrame(data)
        today = datetime.now().strftime('%Y-%m-%d')
        
        stats = {
            'total_students': len(students),
            'total_records': len(df),
            'today_count': len(df[df['Date'] == today]) if 'Date' in df.columns else 0,
            'unique_students': df['Name'].nunique() if 'Name' in df.columns else 0
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("="*50)
    print("FACE ATTENDANCE SYSTEM - FLASK SERVER")
    print("="*50)
    print(f"Base directory: {BASE_DIR}")
    print(f"Images: {IMAGES_PATH}")
    print(f"CSV: {ATTENDANCE_FILE}")
    print(f"Python: {PYTHON_EXECUTABLE}")
    print("="*50)
    print("Server: http://localhost:5000")
    print("="*50)
    
    app.run(debug=True, host='127.0.0.1', port=5000)