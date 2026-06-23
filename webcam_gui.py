# Install dependencies if necessary
# %pip install opencv-python numpy "mediapipe<0.10.21" PySide6

import sys
import cv2
import numpy as np
import time
import os
import threading
from pathlib import Path

import mediapipe as mp
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox)
from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtGui import QImage, QPixmap

# Core data processing functions matching the main pipeline format
MAX_SEQ_LENGTH = 120
FEATURES = 141
FPS = 30
WIDTH = 640
HEIGHT = 480

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
POSE = [0, 11, 12, 13, 14]

def normalize_frame(vec):
    vec = vec.copy()
    pose = vec[:15].reshape(5, 3)
    lh = vec[15:78].reshape(21, 3)
    rh = vec[78:].reshape(21, 3)

    if np.any(pose):
        ls = pose[1]
        rs = pose[2]
        center = (ls + rs) / 2
        dist = np.linalg.norm(ls - rs)
        if dist > 1e-6:
            pose = (pose - center) / dist
            if np.any(lh): lh = (lh - center) / dist
            if np.any(rh): rh = (rh - center) / dist

    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])

def extract(results):
    pose = []
    if results.pose_landmarks:
        for idx in POSE:
            lm = results.pose_landmarks.landmark[idx]
            pose.extend([lm.x, lm.y, lm.z])
    else: pose = [0] * 15

    lh = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            lh.extend([lm.x, lm.y, lm.z])
    else: lh = [0] * 63

    rh = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            rh.extend([lm.x, lm.y, lm.z])
    else: rh = [0] * 63

    return normalize_frame(np.array(pose + lh + rh, dtype=np.float32))

def pad(seq):
    if len(seq) < MAX_SEQ_LENGTH:
        seq = np.vstack([seq, np.zeros((MAX_SEQ_LENGTH - len(seq), FEATURES))])
    return seq[:MAX_SEQ_LENGTH]

def reconstruct(data, outfile):
    writer = cv2.VideoWriter(str(outfile), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for i, frame in enumerate(data):
        canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
        if np.all(frame == 0):
            cv2.putText(canvas, "PAD", (280, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            writer.write(canvas)
            continue

        pose = frame[:15].reshape(5, 3)
        lh = frame[15:78].reshape(21, 3)
        rh = frame[78:].reshape(21, 3)

        def draw(points, color):
            for p in points:
                if p[0] == 0 and p[1] == 0 and p[2] == 0:
                    continue
                x = int((p[0] * 0.25 + 0.5) * WIDTH)
                y = int((p[1] * 0.25 + 0.5) * HEIGHT)
                cv2.circle(canvas, (x, y), 3, color, -1)

        draw(pose, (0, 0, 255))
        draw(lh, (0, 255, 0))
        draw(rh, (255, 255, 0))

        cv2.putText(canvas, f"Frame {i}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
        writer.write(canvas)
    writer.release()

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bridging Silence - Webcam Collector (PySide6)")
        
        self.vid = None
        
        self.is_recording = False
        self.holo = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.init_ui()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.delay = 15
        self.timer.start(self.delay)
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        control_layout = QHBoxLayout()
        
        control_layout.addWidget(QLabel("Word:"))
        self.word_entry = QLineEdit("HELLO")
        self.word_entry.setFixedWidth(100)
        control_layout.addWidget(self.word_entry)
        
        control_layout.addWidget(QLabel("Samples:"))
        self.samples_entry = QLineEdit("3")
        self.samples_entry.setFixedWidth(50)
        control_layout.addWidget(self.samples_entry)
        
        control_layout.addWidget(QLabel("Frames:"))
        self.frames_entry = QLineEdit("90")
        self.frames_entry.setFixedWidth(50)
        control_layout.addWidget(self.frames_entry)
        
        control_layout.addWidget(QLabel("Camera:"))
        self.cam_entry = QLineEdit("0")
        self.cam_entry.setFixedWidth(40)
        control_layout.addWidget(self.cam_entry)
        
        self.start_cam_btn = QPushButton("Open Camera")
        self.start_cam_btn.clicked.connect(self.open_camera)
        control_layout.addWidget(self.start_cam_btn)

        self.btn_record = QPushButton("Start Rec")
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self.start_recording)
        control_layout.addWidget(self.btn_record)
        
        self.btn_stop = QPushButton("Stop Rec")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_recording)
        control_layout.addWidget(self.btn_stop)
        
        main_layout.addLayout(control_layout)
        
        self.video_label = QLabel()
        self.video_label.setFixedSize(WIDTH, HEIGHT)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label)
        
        self.status_label = QLabel("Ready. Adjust Camera Index and click Open Camera.")
        main_layout.addWidget(self.status_label)
        
    def open_camera(self):
        if self.vid is not None:
            self.vid.release()
            self.vid = None
            self.start_cam_btn.setText("Open Camera")
            self.btn_record.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.status_label.setText("Camera closed.")
            self.video_label.clear()
            self.video_label.setStyleSheet("background-color: black;")
            return
            
        cam_idx = 0
        try:
            cam_idx = int(self.cam_entry.text())
        except ValueError:
            pass
            
        if os.name == 'nt':
            self.vid = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        else:
            self.vid = cv2.VideoCapture(cam_idx)
            
        if not self.vid.isOpened():
            self.vid = cv2.VideoCapture(cam_idx)
            
        if self.vid.isOpened():
            self.start_cam_btn.setText("Close Camera")
            self.btn_record.setEnabled(True)
            self.status_label.setText(f"Camera {cam_idx} opened.")
        else:
            self.status_label.setText(f"Failed to open camera {cam_idx}")
            self.vid = None

    def start_recording(self):
        if not self.vid or not self.vid.isOpened():
            QMessageBox.critical(self, "Error", "Camera is not opened.")
            return
            
        self.word = self.word_entry.text().strip()
        if not self.word:
            QMessageBox.critical(self, "Error", "Please enter a word.")
            return
            
        try:
            self.num_samples = int(self.samples_entry.text())
            self.frames_per_sample = int(self.frames_entry.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Samples and Frames must be integers.")
            return

        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.is_recording = True
        self.current_sample = 0
        self.countdown = 3
        self.wait_timer = 0
        self.last_time = time.time()

    def stop_recording(self):
        self.is_recording = False
        self.btn_record.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_label.setText("Recording stopped.")

    def save_sample_task(self, frames, seq, sample_idx):
        sample_name = f"sample_{int(time.time())}"
        
        WEBCAM_DATA_DIR = "webcam_data"
        WEBCAM_OUT_DIR = "processed_webcam_data"
        
        raw_root = Path(WEBCAM_DATA_DIR) / self.word
        land_root = Path(WEBCAM_OUT_DIR) / "landmarks" / self.word
        prev_root = Path(WEBCAM_OUT_DIR) / "preview" / self.word
        
        raw_root.mkdir(parents=True, exist_ok=True)
        land_root.mkdir(parents=True, exist_ok=True)
        prev_root.mkdir(parents=True, exist_ok=True)
        
        raw_vid_path = raw_root / f"{sample_name}.mp4"
        writer = cv2.VideoWriter(str(raw_vid_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
        for f in frames: writer.write(f)
        writer.release()
        
        seq_np = np.array(seq)
        seq_np = pad(seq_np)
        
        npy_path = land_root / f"{sample_name}.npy"
        np.save(npy_path, seq_np)
        
        prev_path = prev_root / f"{sample_name}.mp4"
        reconstruct(seq_np, prev_path)
        
        print(f"Saved {sample_name} for word '{self.word}'")

    def save_sample(self):
        frames_copy = list(self.frames)
        seq_copy = list(self.seq)
        
        threading.Thread(target=self.save_sample_task, args=(frames_copy, seq_copy, self.current_sample)).start()
        
        self.current_sample += 1
        if self.current_sample < self.num_samples:
            self.wait_timer = 1
            self.last_time = time.time()
            self.status_label.setText("Waiting before next sample...")
        else:
            self.is_recording = False
            self.btn_record.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.status_label.setText("Collection complete!")

    def set_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def update_frame(self):
        if self.vid is not None and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                
                if self.is_recording:
                    if self.countdown > 0:
                        if time.time() - self.last_time >= 1.0:
                            self.countdown -= 1
                            self.last_time = time.time()
                        
                        display_frame = frame.copy()
                        cv2.putText(display_frame, f"Starting {self.current_sample+1} in {self.countdown}...", 
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                        self.set_image(display_frame)
                        
                        if self.countdown == 0:
                            self.status_label.setText(f"Recording: {self.word} (Sample {self.current_sample+1}/{self.num_samples})")
                            self.frames = []
                            self.seq = []
                    
                    elif self.wait_timer > 0:
                        if time.time() - self.last_time >= 1.0:
                            self.wait_timer -= 1
                            self.last_time = time.time()
                            if self.wait_timer == 0:
                                self.countdown = 3
                        
                        display_frame = frame.copy()
                        cv2.putText(display_frame, f"Wait...", 
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                        
                        self.set_image(display_frame)
                        
                    else:
                        self.frames.append(frame)
                        rgb_for_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = self.holo.process(rgb_for_mp)
                        self.seq.append(extract(res))
                        
                        display_frame = frame.copy()
                        cv2.putText(display_frame, f"Recording [{len(self.frames)}/{self.frames_per_sample}]", 
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    
                        self.set_image(display_frame)
                        
                        if len(self.frames) >= self.frames_per_sample:
                            self.save_sample()
                else:
                    self.set_image(frame)

    def closeEvent(self, event):
        if self.vid is not None:
            self.vid.release()
        self.holo.close()
        event.accept()

def main():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    window = App()
    window.show()
    app.exec()

if __name__ == '__main__':
    main()
