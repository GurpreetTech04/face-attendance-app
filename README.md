# Face Attendance App

## Setup Instructions

1.  **Install Requirements:**
    Open terminal in this folder and run:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `face-recognition` needs `dlib`, which can be tricky on Windows. You might need to install Visual Studio C++ Build Tools first.*

2.  **Add Photos:**
    Put clear photos of people in the `images/` folder.
    *   Example: `Raman.jpg`, `Friend.jpg`
    *   The filename will be used as the name in attendance.

3.  **Run the App:**
    ```bash
    python main.py
    ```

4.  **View Attendance:**
    Attendance will be saved automatically in `attendance.csv` (created on first run).

## For Google Colab

If running on Colab, replace `main.py` with the specialized JS-based camera code because `cv2.VideoCapture(0)` won't work in the cloud.
