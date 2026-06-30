# --------------------------------------------------------------
#  MODERN AI PROCTORING SYSTEM - IMPROVED DETECTION + FULL STACK
#  (Single-file version with screen/mic/ffmpeg/input/browser)
# --------------------------------------------------------------
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import threading
from collections import deque
import csv
from datetime import datetime, timedelta
import json
import os
import wave
import pyaudio
import warnings
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tempfile

# NEW IMPORTS FOR FULL PROCTORING STACK
import mss
import pyautogui
from pynput import keyboard, mouse
import ffmpeg
from selenium import webdriver

# ================================
# CONFIGURATION MANAGEMENT
# ================================
class Config:
    def __init__(self, config_file="proctor_config.json"):
        self.config_file = config_file
        self.default_config = {
            "fullscreen": True,
            "show_preview": True,
            "preview_width": 320,
            "preview_height": 240,
            "camera_width": 1280,
            "camera_height": 720,
            "process_every": 2,
            "history_max": 15,
            "gaze_warning_cooldown": 2.5,
            "multi_person_cooldown": 3.0,
            "no_face_cooldown": 5.0,
            "phone_detection_interval": 10,
            "audio_threshold": 500,
            "gaze_thresholds": {
                "left": 0.38,
                "right": 0.62,
                "up": 0.35,
                "down": 0.65
            },
            "ear_threshold": 0.20,
            "voice_rate": 150,
            "voice_volume": 0.9,
            "reports_folder": "Reports",
            "log_retention_days": 30,
            # Additional config for system recording
            "screenshot_interval_sec": 30,
            "screen_record_fps": 10,
            "allowed_exam_domains": ["example-exam.com"]  # replace with real exam domains
        }
        self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                self.default_config.update(user_config)
                print(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        else:
            self.save_config()
        
        os.makedirs(self.default_config["reports_folder"], exist_ok=True)
    
    def save_config(self):
        """Save current configuration to JSON file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.default_config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def __getitem__(self, key):
        return self.default_config[key]
    
    def __setitem__(self, key, value):
        self.default_config[key] = value

# ================================
# COMPACT UI THEME
# ================================
class CompactUITheme:
    def __init__(self):
        # Professional dark color scheme
        self.colors = {
            'background': (13, 17, 23),
            'surface': (22, 27, 34),
            'surface_light': (33, 38, 45),
            'primary': (47, 129, 247),
            'success': (46, 160, 67),
            'warning': (217, 123, 0),
            'danger': (218, 54, 51),
            'text_primary': (248, 250, 252),
            'text_secondary': (173, 186, 199),
            'text_muted': (108, 122, 137),
            'border': (48, 54, 61)
        }
        
    def draw_panel(self, img, x, y, width, height, title=None):
        """Draw a clean panel with border"""
        # Fill panel
        cv2.rectangle(img, (x, y), (x + width, y + height), self.colors['surface'], -1)
        # Border
        cv2.rectangle(img, (x, y), (x + width, y + height), self.colors['border'], 2)
        
        # Title if provided
        if title:
            cv2.putText(img, title, (x + 10, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text_primary'], 2)

    def draw_status_indicator(self, img, x, y, status, size=8):
        """Draw small status indicator"""
        color_map = {
            'normal': self.colors['success'],
            'warning': self.colors['warning'], 
            'danger': self.colors['danger'],
            'inactive': self.colors['text_muted']
        }
        color = color_map.get(status, self.colors['text_muted'])
        cv2.circle(img, (x, y), size, color, -1)

# ================================
# AUDIO MONITORING (MIC CAPTURE + NOISE DETECTION)
# ================================
class AudioMonitor:
    def __init__(self, threshold=500, output_wav_path=None):
        self.threshold = threshold
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_monitoring = False
        self.noise_detected = False
        self.last_noise_time = 0
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.output_wav_path = output_wav_path
        self.wave_file = None
        
    def start_monitoring(self):
        """Start audio monitoring in a separate thread and record to WAV"""
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            if self.output_wav_path:
                self.wave_file = wave.open(self.output_wav_path, 'wb')
                self.wave_file.setnchannels(self.channels)
                self.wave_file.setsampwidth(self.audio.get_sample_size(self.format))
                self.wave_file.setframerate(self.rate)

            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_audio, daemon=True)
            self.monitor_thread.start()
        except Exception as e:
            print(f"Audio monitoring not available: {e}")
    
    def _monitor_audio(self):
        """Monitor audio levels for background noise and record raw audio"""
        while self.is_monitoring:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                # Record to WAV
                if self.wave_file:
                    self.wave_file.writeframes(data)

                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data**2))
                
                if rms > self.threshold:
                    self.noise_detected = True
                    self.last_noise_time = time.time()
                else:
                    self.noise_detected = False
                    
            except Exception as e:
                print(f"Audio monitoring error: {e}")
                break
            time.sleep(0.1)
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        self.is_monitoring = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.wave_file:
            self.wave_file.close()
        self.audio.terminate()

# ================================
# SCREEN RECORDER (MSS + OpenCV)
# ================================
class ScreenRecorder(threading.Thread):
    def __init__(self, output_path, fps=10):
        super().__init__(daemon=True)
        self.output_path = output_path
        self.fps = fps
        self.running = False
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.width = self.monitor["width"]
        self.height = self.monitor["height"]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

    def run(self):
        self.running = True
        frame_interval = 1.0 / self.fps
        while self.running:
            start = time.time()
            img = self.sct.grab(self.monitor)
            frame = np.array(img)[:, :, :3]  # BGRA -> BGR
            self.writer.write(frame)
            elapsed = time.time() - start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    def stop(self):
        self.running = False
        time.sleep(0.1)
        try:
            self.writer.release()
        except:
            pass
        try:
            self.sct.close()
        except:
            pass

# ================================
# PERIODIC SCREENSHOTS (pyautogui)
# ================================
class ScreenshotCapturer(threading.Thread):
    def __init__(self, folder, interval_sec=30, event_callback=None):
        super().__init__(daemon=True)
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.interval = interval_sec
        self.running = False
        self.event_callback = event_callback

    def run(self):
        self.running = True
        while self.running:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self.folder, f"screenshot_{ts}.png")
            try:
                img = pyautogui.screenshot()
                img.save(path)
                if self.event_callback:
                    self.event_callback("Screenshot", f"Saved {os.path.basename(path)}")
            except Exception as e:
                if self.event_callback:
                    self.event_callback("ScreenshotError", str(e))
            # Sleep in small steps to allow fast stop
            total_sleep = self.interval
            step = 0.2
            while self.running and total_sleep > 0:
                time.sleep(step)
                total_sleep -= step

    def stop(self):
        self.running = False

# ================================
# INPUT MONITOR (pynput)
# ================================
class InputMonitor:
    def __init__(self, event_callback, suspicious_threshold=30, window_sec=5):
        """
        event_callback: callable(event_type:str, details:str)
        suspicious_threshold: number of events in window_sec to flag pattern
        """
        self.event_callback = event_callback
        self.suspicious_threshold = suspicious_threshold
        self.window_sec = window_sec
        self.events = deque(maxlen=1000)
        self.lock = threading.Lock()
        self.k_listener = None
        self.m_listener = None

    def start(self):
        self.k_listener = keyboard.Listener(on_press=self.on_key)
        self.m_listener = mouse.Listener(on_click=self.on_click)
        self.k_listener.start()
        self.m_listener.start()

    def stop(self):
        try:
            if self.k_listener:
                self.k_listener.stop()
            if self.m_listener:
                self.m_listener.stop()
        except:
            pass

    def _register_event(self, kind, detail):
        now = time.time()
        with self.lock:
            self.events.append(now)
            # Drop old
            while self.events and now - self.events[0] > self.window_sec:
                self.events.popleft()
            # Suspicious pattern
            if len(self.events) >= self.suspicious_threshold:
                self.event_callback("Suspicious Input Pattern", f"{len(self.events)} events in {self.window_sec}s")
                self.events.clear()
        self.event_callback(kind, detail)

    def on_key(self, key):
        try:
            k_str = key.char if hasattr(key, 'char') else str(key)
        except:
            k_str = str(key)
        self._register_event("Keyboard", k_str)

    def on_click(self, x, y, button, pressed):
        if pressed:
            self._register_event("Mouse", f"{button}@{x},{y}")

# ================================
# BROWSER MONITOR (selenium)
# ================================
class BrowserMonitor(threading.Thread):
    def __init__(self, event_callback, allowed_domains, start_url=None):
        """
        allowed_domains: list of domain substrings considered allowed (e.g. ["exam.com"])
        """
        super().__init__(daemon=True)
        self.event_callback = event_callback
        self.allowed_domains = allowed_domains
        self.running = False

        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        # You can add more restrictions here if needed (disable extensions, etc.)
        try:
            self.driver = webdriver.Chrome(options=options)
            if start_url:
                self.driver.get(start_url)
        except Exception as e:
            self.driver = None
            self.event_callback("BrowserMonitorInitError", str(e))

    def is_allowed(self, url: str) -> bool:
        if not url:
            return True
        return any(dom in url for dom in self.allowed_domains)

    def run(self):
        if self.driver is None:
            return
        self.running = True
        while self.running:
            try:
                url = self.driver.current_url
                if not self.is_allowed(url):
                    self.event_callback("Browser Navigation Blocked", url)
                    try:
                        self.driver.back()
                    except:
                        pass
            except Exception as e:
                self.event_callback("BrowserMonitorError", str(e))
            time.sleep(1.0)

    def stop(self):
        self.running = False
        time.sleep(0.5)
        try:
            if self.driver:
                self.driver.quit()
        except:
            pass

# ================================
# REPORT GENERATOR
# ================================
class ReportGenerator:
    def __init__(self, reports_folder="Reports"):
        self.reports_folder = reports_folder
        os.makedirs(reports_folder, exist_ok=True)
    
    def generate_pdf_report(self, student_id, session_data, violation_counts, event_history, duration):
        """Generate a comprehensive PDF report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.reports_folder, f"proctor_report_{student_id}_{timestamp}.pdf")
        
        with PdfPages(filename) as pdf:
            # Title page
            plt.figure(figsize=(11, 8.5))
            plt.axis('off')
            plt.text(0.5, 0.7, "EXAM PROCTORING REPORT", 
                    ha='center', va='center', fontsize=24, weight='bold')
            plt.text(0.5, 0.6, f"Student: {student_id}", 
                    ha='center', va='center', fontsize=16)
            plt.text(0.5, 0.5, f"Session: {session_data['start_time'].strftime('%Y-%m-%d %H:%M:%S')}", 
                    ha='center', va='center', fontsize=12)
            plt.text(0.5, 0.4, f"Duration: {duration}", 
                    ha='center', va='center', fontsize=12)
            pdf.savefig()
            plt.close()
            
            # Violation summary
            plt.figure(figsize=(11, 8.5))
            plt.subplot(2, 1, 1)
            
            violations = list(violation_counts.keys())
            counts = list(violation_counts.values())
            
            colors = ['red' if count > 0 else 'green' for count in counts]
            bars = plt.bar(violations, counts, color=colors, alpha=0.7)
            
            plt.title('Violation Summary', fontsize=16, weight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Count')
            
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
            
            plt.subplot(2, 1, 2)
            total_violations = sum(counts)
            if total_violations > 0:
                labels = ['Normal', 'Violations']
                sizes = [max(1, 100 - total_violations), total_violations]
                colors = ['green', 'red']
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                plt.axis('equal')
            else:
                plt.text(0.5, 0.5, 'No Violations\nDetected', 
                        ha='center', va='center', fontsize=16, weight='bold')
                plt.axis('off')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # Event timeline
            plt.figure(figsize=(11, 8.5))
            if event_history:
                events = list(event_history)[-10:]
                y_pos = range(len(events))
                
                plt.barh(y_pos, [1] * len(events), color='lightblue', alpha=0.7)
                plt.yticks(y_pos, [event.split('→')[-1].strip() for event in events])
                plt.xlabel('Recent Events Timeline')
                plt.title('Recent Proctoring Events', fontsize=16, weight='bold')
            else:
                plt.text(0.5, 0.5, 'No Events Recorded', 
                        ha='center', va='center', fontsize=16)
                plt.axis('off')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        return filename

# ================================
# IMPROVED PROCTORING CLASS
# ================================
class ImprovedProctorMonitor:
    def __init__(self):
        self.config = Config()
        self.theme = CompactUITheme()
        self.student_id = input("Enter Student ID (or press Enter for 'Guest'): ").strip() or "Guest"

        # Recording paths (used by webcam, screen, audio, screenshots)
        base_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recordings_folder = os.path.join(self.config["reports_folder"], "Recordings")
        os.makedirs(self.recordings_folder, exist_ok=True)
        self.webcam_record_path = os.path.join(self.recordings_folder, f"{self.student_id}_{base_ts}_webcam.avi")
        self.screen_record_path = os.path.join(self.recordings_folder, f"{self.student_id}_{base_ts}_screen.avi")
        self.audio_record_path = os.path.join(self.recordings_folder, f"{self.student_id}_{base_ts}_audio.wav")
        self.screenshot_folder = os.path.join(self.recordings_folder, f"{self.student_id}_{base_ts}_screenshots")

        # Initialize components
        self.setup_mediapipe()
        self.setup_audio()
        self.setup_tts()
        self.setup_camera()
        
        # Session data
        self.session_start = time.time()
        self.frame_counter = 0
        self.fps_history = deque(maxlen=30)
        self.attention_history = deque(maxlen=100)
        self.prev_gaze_dir = "Center"
        
        # Enhanced violation tracking
        self.violation_counts = {
            "Look Left": 0, "Look Right": 0, "Look Up": 0, "Look Down": 0,
            "Multiple People": 0, "No Face": 0, "Phone Detected": 0, "Background Noise": 0
        }
        
        # Detection state tracking
        self.phone_detection_count = 0
        self.multiple_people_count = 0
        self.current_face_count = 0
        self.current_phone_detected = False
        self.current_multiple_people = False
        
        # Cooldown timers
        self.last_gaze_warning = 0
        self.last_multi_warning = 0
        self.last_no_face_warning = 0
        self.last_phone_check = 0
        
        # Event history
        self.event_history = deque(maxlen=self.config["history_max"])
        
        # Log file
        timestamp = int(time.time())
        self.log_file = f"proctor_log_{self.student_id}_{timestamp}.csv"
        self.setup_logging()

        # System-level recorders / monitors (screen, screenshots, input, browser)
        self.setup_system_monitors()
        
        print(f"\n🚀 Improved Proctoring Monitor STARTED for Student: {self.student_id}")
        print(f"📊 Log: {self.log_file}")
        print("🎮 Controls: ESC to end session, 'p' to toggle preview\n")
    
    def setup_mediapipe(self):
        """Initialize MediaPipe models"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_objectron = mp.solutions.objectron
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=5,
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Eye landmarks
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
    
    def setup_audio(self):
        """Initialize audio monitoring + audio recording"""
        self.audio_monitor = AudioMonitor(
            threshold=self.config["audio_threshold"],
            output_wav_path=self.audio_record_path
        )
        try:
            self.audio_monitor.start_monitoring()
            self.audio_available = True
        except:
            self.audio_available = False
            print("Audio monitoring not available")
    
    def setup_tts(self):
        """Initialize text-to-speech"""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', self.config["voice_rate"])
        self.engine.setProperty('volume', self.config["voice_volume"])
        
        self.speech_queue = deque()
        self.speech_lock = threading.Lock()
        self.speech_thread = None
        self.stop_speech = threading.Event()
    
    def setup_camera(self):
        """Initialize camera and webcam recording"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera_height"])

        # Webcam recording writer (raw AVI; compressed later with ffmpeg)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.webcam_writer = cv2.VideoWriter(
            self.webcam_record_path,
            fourcc,
            20.0,
            (int(self.config["camera_width"]), int(self.config["camera_height"]))
        )
        
        # Create window (UI unchanged)
        cv2.namedWindow("AI Proctoring Monitor", cv2.WND_PROP_FULLSCREEN)
        if self.config["fullscreen"]:
            cv2.setWindowProperty("AI Proctoring Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def setup_logging(self):
        """Initialize CSV logging"""
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "StudentID", "Event Type", "Details"])
            writer.writerow([datetime.now().isoformat(), self.student_id, "SESSION_START", "Proctoring started"])
    
    def setup_system_monitors(self):
        """Initialize screen recording, screenshots, input monitoring, and browser monitoring"""
        # Screen recorder
        self.screen_recorder = ScreenRecorder(
            self.screen_record_path,
            fps=self.config["screen_record_fps"]
        )
        # Periodic screenshots
        self.screenshot_capturer = ScreenshotCapturer(
            self.screenshot_folder,
            interval_sec=self.config["screenshot_interval_sec"],
            event_callback=lambda et, d: self.log_event(et, d)
        )
        # Input monitor
        self.input_monitor = InputMonitor(
            event_callback=lambda et, d: self.log_event(et, d),
            suspicious_threshold=30,
            window_sec=5
        )
        # Browser monitor (optional if Chrome driver available)
        self.browser_monitor = None
        try:
            self.browser_monitor = BrowserMonitor(
                event_callback=lambda et, d: self.log_event(et, d),
                allowed_domains=self.config["allowed_exam_domains"],
                start_url=None  # you can pass exam URL here
            )
        except Exception as e:
            print(f"Browser monitor not started: {e}")
            self.browser_monitor = None

    def log_event(self, event_type: str, details: str = ""):
        """Log event to CSV and history"""
        ts = datetime.now()
        ts_str = ts.strftime("%H:%M:%S")
        full = f"{ts_str} → {event_type}"
        if details:
            full += f" ({details})"
        self.event_history.append(full)

        if event_type in self.violation_counts:
            self.violation_counts[event_type] += 1

        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts.isoformat(), self.student_id, event_type, details])
    
    def speak(self, text: str):
        """Non-blocking text-to-speech"""
        def _speech_worker():
            while not self.stop_speech.is_set():
                if self.speech_queue:
                    with self.speech_lock:
                        if self.speech_queue and self.speech_queue[0] == "__STOP__":
                            self.speech_queue.popleft()
                            break
                        txt = self.speech_queue.popleft()
                    self.engine.say(txt)
                    self.engine.runAndWait()
                else:
                    time.sleep(0.1)
        
        with self.speech_lock:
            self.speech_queue.append(text)
        if self.speech_thread is None or not self.speech_thread.is_alive():
            self.speech_thread = threading.Thread(target=_speech_worker, daemon=True)
            self.speech_thread.start()
    
    # ================================
    # IMPROVED DETECTION METHODS
    # ================================
    def eye_coords(self, landmarks, idx, w, h):
        return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idx], dtype=np.float32)

    def iris_center(self, landmarks, idx, w, h):
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx]
        return np.mean(pts, axis=0)

    def eye_bbox(self, eye_pts):
        hull = cv2.convexHull(eye_pts.astype(np.int32))
        x, y, bw, bh = cv2.boundingRect(hull)
        return x, y, x + bw, y + bh

    def gaze_ratio(self, iris, bbox):
        left, top, right, bottom = bbox
        cx, cy = iris
        w = max(right - left, 5)
        h = max(bottom - top, 5)
        return (cx - left) / w, (cy - top) / h

    def direction_from_ratio(self, xr, yr):
        thresholds = self.config["gaze_thresholds"]
        if xr < thresholds["left"]: return "Look Left"
        if xr > thresholds["right"]: return "Look Right"
        if yr < thresholds["up"]: return "Look Up"
        if yr > thresholds["down"]: return "Look Down"
        return "Center"

    def eye_aspect_ratio(self, eye_pts):
        A = np.linalg.norm(eye_pts[1] - eye_pts[5])
        B = np.linalg.norm(eye_pts[2] - eye_pts[4])
        C = np.linalg.norm(eye_pts[0] - eye_pts[3])
        return (A + B) / (2.0 * (C + 1e-6))
    
    def detect_phone_usage(self, frame, holistic_results):
        """Improved phone detection using multiple methods"""
        phone_detected = False
        h, w, _ = frame.shape
        
        # Method 1: Hand near face detection
        if holistic_results.left_hand_landmarks or holistic_results.right_hand_landmarks:
            face_landmarks = holistic_results.face_landmarks
            
            if face_landmarks:
                # Get face bounding box with larger margin
                face_x = [lm.x for lm in face_landmarks.landmark]
                face_y = [lm.y for lm in face_landmarks.landmark]
                face_left = min(face_x) - 0.2  # Increased margin
                face_right = max(face_x) + 0.2
                face_top = min(face_y) - 0.2
                face_bottom = max(face_y) + 0.2
                
                # Check if hands are near face
                for hand_landmarks in [holistic_results.left_hand_landmarks, holistic_results.right_hand_landmarks]:
                    if hand_landmarks:
                        hand_x = np.mean([lm.x for lm in hand_landmarks.landmark])
                        hand_y = np.mean([lm.y for lm in hand_landmarks.landmark])
                        
                        if (face_left < hand_x < face_right and 
                            face_top < hand_y < face_bottom):
                            phone_detected = True
                            break
        
        # Method 2: Hand shape analysis for phone grip
        if not phone_detected:
            phone_detected = self.detect_phone_by_hand_shape(frame)
        
        # Method 3: Object detection for rectangular shapes near face
        if not phone_detected:
            phone_detected = self.detect_phone_by_shape(frame)
        
        return phone_detected
    
    def detect_phone_by_hand_shape(self, frame):
        """Detect phone usage by analyzing hand shapes that typically hold phones"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb)
        
        if not hand_results.multi_hand_landmarks:
            return False
        
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Get key hand landmarks
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape
            
            # Calculate hand features that might indicate phone holding
            thumb_tip = np.array([landmarks[4].x * w, landmarks[4].y * h])
            index_tip = np.array([landmarks[8].x * w, landmarks[8].y * h])
            middle_tip = np.array([landmarks[12].x * w, landmarks[12].y * h])
            
            # Check for "phone grip" - thumb and fingers close together
            thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
            thumb_middle_dist = np.linalg.norm(thumb_tip - middle_tip)
            
            # Typical phone grip has thumb and fingers close (20-100 pixels at 720p)
            avg_dist = (thumb_index_dist + thumb_middle_dist) / 2
            if 20 < avg_dist < 150:  # Adjusted range for phone grip
                return True
        
        return False
    
    def detect_phone_by_shape(self, frame):
        """Detect phone-like rectangular shapes in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 1000:  # Minimum phone size
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 sides)
            if len(approx) == 4:
                # Check aspect ratio (typical phone aspect ratios)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                if 0.4 < aspect_ratio < 0.7:  # Typical phone aspect ratios
                    return True
        
        return False
    
    def calculate_attention_score(self):
        """Calculate attention score based on gaze stability"""
        if not self.attention_history:
            return 100
        
        center_count = sum(1 for gaze in self.attention_history if gaze == "Center")
        return (center_count / len(self.attention_history)) * 100
    
    def calculate_fps(self):
        """Calculate current FPS"""
        if len(self.fps_history) < 2:
            return 0
        time_diffs = np.diff(list(self.fps_history))
        return 1.0 / np.mean(time_diffs) if len(time_diffs) > 0 else 0

    # ================================
    # IMPROVED UI METHODS (UNCHANGED LAYOUT)
    # ================================
    def draw_top_bar(self, disp, w, h):
        """Draw the top bar with title, timer, and student ID"""
        # Top bar background
        cv2.rectangle(disp, (0, 0), (w, 50), self.theme.colors['surface'], -1)
        cv2.rectangle(disp, (0, 0), (w, 50), self.theme.colors['border'], 1)
        
        # Title (left)
        cv2.putText(disp, "AI PROCTORING SYSTEM", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.theme.colors['text_primary'], 2)
        
        # Timer (center)
        elapsed = timedelta(seconds=int(time.time() - self.session_start))
        timer_text = f"TIME: {str(elapsed)}"
        timer_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(disp, timer_text, (w//2 - timer_size[0]//2, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.colors['primary'], 2)
        
        # Student ID (right)
        student_text = f"STUDENT: {self.student_id}"
        student_size = cv2.getTextSize(student_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(disp, student_text, (w - student_size[0] - 20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.colors['text_secondary'], 2)

    def draw_status_panel(self, disp, w, h):
        """Draw improved left status panel with multiple people count and phone detection"""
        panel_w = w // 4
        panel_h = h - 100
        panel_x = 10
        panel_y = 60
        
        self.theme.draw_panel(disp, panel_x, panel_y, panel_w, panel_h, "STATUS PANEL")
        
        # Status items
        y_offset = panel_y + 50
        
        # Face Count with Multiple People Detection
        face_count = getattr(self, 'current_face_count', 0)
        if face_count == 0:
            status = 'danger'
            face_text = f"Faces: {face_count}"
        elif face_count == 1:
            status = 'normal'
            face_text = f"Faces: {face_count}"
        else:
            status = 'danger'
            face_text = f"Faces: {face_count} (Multiple!)"
        
        self.theme.draw_status_indicator(disp, panel_x + 15, y_offset, status)
        cv2.putText(disp, face_text, (panel_x + 30, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme.colors['text_primary'], 1)
        
        # Multiple People Count
        y_offset += 30
        multi_count = getattr(self, 'multiple_people_count', 0)
        self.theme.draw_status_indicator(disp, panel_x + 15, y_offset, 'danger' if multi_count > 0 else 'inactive')
        cv2.putText(disp, f"Multi Events: {multi_count}", (panel_x + 30, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme.colors['text_primary'], 1)
        
        # Gaze Direction
        y_offset += 30
        gaze_dir = getattr(self, 'current_gaze_dir', 'Center')
        status = 'normal' if gaze_dir == 'Center' else 'warning'
        self.theme.draw_status_indicator(disp, panel_x + 15, y_offset, status)
        cv2.putText(disp, f"Gaze Dir: {gaze_dir}", (panel_x + 30, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme.colors['text_primary'], 1)
        
        # Eyes Status
        y_offset += 30
        eyes_status = "Open" if not getattr(self, 'current_blink_detected', False) else "Closed"
        status = 'normal' if eyes_status == "Open" else 'warning'
        self.theme.draw_status_indicator(disp, panel_x + 15, y_offset, status)
        cv2.putText(disp, f"Eyes: {eyes_status}", (panel_x + 30, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme.colors['text_primary'], 1)
        
        # Phone Detection with Count
        y_offset += 30
        phone_detected = getattr(self, 'current_phone_detected', False)
        phone_count = getattr(self, 'phone_detection_count', 0)
        status = 'danger' if phone_detected else 'normal'
        self.theme.draw_status_indicator(disp, panel_x + 15, y_offset, status)
        status_text = "Detected!" if phone_detected else "Not Detected"
        cv2.putText(disp, f"Phone: {status_text}", (panel_x + 30, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme.colors['text_primary'], 1)
        
        # Phone Count
        y_offset += 20
        cv2.putText(disp, f"Phone Events: {phone_count}", (panel_x + 30, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.theme.colors['text_secondary'], 1)

    def draw_center_panel(self, disp, w, h, frame):
        """Draw center panel with gaze compass and camera feed"""
        panel_w = w // 2
        panel_h = h - 100
        panel_x = w // 4 + 5
        panel_y = 60
        
        self.theme.draw_panel(disp, panel_x, panel_y, panel_w, panel_h, "GAZE COMPASS + CAMERA FEED")
        
        # Gaze compass position
        compass_x = panel_x + panel_w // 2
        compass_y = panel_y + 80
        
        # Draw simple compass
        cv2.circle(disp, (compass_x, compass_y), 40, self.theme.colors['surface_light'], -1)
        cv2.circle(disp, (compass_x, compass_y), 40, self.theme.colors['border'], 2)
        
        # Current gaze direction indicator
        gaze_dir = getattr(self, 'current_gaze_dir', 'Center')
        directions = {
            "Look Left": (-30, 0),
            "Look Right": (30, 0),
            "Look Up": (0, -30),
            "Look Down": (0, 30),
            "Center": (0, 0)
        }
        
        dx, dy = directions.get(gaze_dir, (0, 0))
        color = self.theme.colors['success'] if gaze_dir == "Center" else self.theme.colors['warning']
        cv2.circle(disp, (compass_x + dx, compass_y + dy), 12, color, -1)
        
        # Center dot
        cv2.circle(disp, (compass_x, compass_y), 5, self.theme.colors['primary'], -1)
        
        # Camera feed below compass
        feed_y = compass_y + 80
        feed_h = panel_h - 160
        
        if hasattr(self, 'current_frame'):
            # Resize frame to fit
            feed_w = min(panel_w - 40, int(self.current_frame.shape[1] * (feed_h / self.current_frame.shape[0])))
            resized_frame = cv2.resize(self.current_frame, (feed_w, feed_h))
            
            # Center the feed
            feed_x = panel_x + (panel_w - feed_w) // 2
            
            # Add border to camera feed
            cv2.rectangle(disp, (feed_x, feed_y), (feed_x + feed_w, feed_y + feed_h), 
                         self.theme.colors['primary'], 2)
            
            # Place camera feed
            disp[feed_y:feed_y + feed_h, feed_x:feed_x + feed_w] = resized_frame

    def draw_summary_panel(self, disp, w, h):
        """Draw right summary panel"""
        panel_w = w // 4 - 20
        panel_h = h - 100
        panel_x = w - panel_w - 10
        panel_y = 60
        
        self.theme.draw_panel(disp, panel_x, panel_y, panel_w, panel_h, "SUMMARY")
        
        # Attention score
        attention_score = self.calculate_attention_score()
        y_offset = panel_y + 50
        
        # Simple attention bar
        bar_width = panel_w - 40
        cv2.rectangle(disp, (panel_x + 20, y_offset), 
                     (panel_x + 20 + bar_width, y_offset + 20), 
                     self.theme.colors['surface_light'], -1)
        
        fill_width = int((attention_score / 100) * bar_width)
        if fill_width > 0:
            color = self.theme.colors['success'] if attention_score > 70 else self.theme.colors['warning'] if attention_score > 40 else self.theme.colors['danger']
            cv2.rectangle(disp, (panel_x + 20, y_offset), 
                         (panel_x + 20 + fill_width, y_offset + 20), 
                         color, -1)
        
        cv2.putText(disp, f"Attention: {attention_score:.1f}%", 
                   (panel_x + 20, y_offset + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme.colors['text_primary'], 1)
        
        # Violations section
        y_offset += 60
        cv2.putText(disp, "Violations:", (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.colors['text_primary'], 1)
        
        y_offset += 25
        total_violations = sum(self.violation_counts.values())
        cv2.putText(disp, f"Total: {total_violations}", (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme.colors['text_secondary'], 1)
        
        # Recent events
        y_offset += 50
        cv2.putText(disp, "Recent Events:", (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.theme.colors['text_primary'], 1)
        
        y_offset += 25
        for i, event in enumerate(list(self.event_history)[-4:]):
            color = self.theme.colors['danger'] if any(x in event for x in ['Look', 'Multiple', 'Phone', 'Noise', 'Suspicious', 'Browser']) else self.theme.colors['text_secondary']
            cv2.putText(disp, event, (panel_x + 20, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def draw_bottom_bar(self, disp, w, h):
        """Draw bottom status bar"""
        bar_height = 40
        bar_y = h - bar_height
        
        # Background
        cv2.rectangle(disp, (0, bar_y), (w, h), self.theme.colors['surface'], -1)
        cv2.rectangle(disp, (0, bar_y), (w, h), self.theme.colors['border'], 1)
        
        # Camera status
        cam_status = "✓" if self.cap.isOpened() else "✗"
        cam_color = self.theme.colors['success'] if self.cap.isOpened() else self.theme.colors['danger']
        cv2.putText(disp, f"CAMERA: {cam_status}", (20, bar_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, cam_color, 1)
        
        # Audio status
        audio_status = "✓" if self.audio_available else "✗"
        audio_color = self.theme.colors['success'] if self.audio_available else self.theme.colors['danger']
        cv2.putText(disp, f"AUDIO: {audio_status}", (150, bar_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, audio_color, 1)
        
        # FPS
        fps = self.calculate_fps()
        cv2.putText(disp, f"FPS: {fps:.1f}", (280, bar_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme.colors['text_secondary'], 1)
        
        # Controls
        controls_text = "[ESC] Exit  [P] Preview"
        controls_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        cv2.putText(disp, controls_text, (w - controls_size[0] - 20, bar_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.theme.colors['text_muted'], 1)

    def process_frame(self, frame):
        """Improved frame processing with better phone detection"""
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_count = 0
        direction = "Center"
        blink_detected = False
        phone_detected = False
        
        now = time.time()
        
        # Process with MediaPipe every N frames
        if self.frame_counter % self.config["process_every"] == 0:
            face_results = self.face_mesh.process(rgb)
            face_count = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
            
            # Update current face count for UI
            self.current_face_count = face_count
            
            # Multiple people detection and counting
            if face_count > 1:
                if now - self.last_multi_warning > self.config["multi_person_cooldown"]:
                    self.multiple_people_count += 1
                    self.speak("Multiple people detected")
                    self.log_event("Multiple People", f"{face_count} faces")
                    self.last_multi_warning = now
            
            # Holistic processing for phone detection (less frequent but more thorough)
            if self.frame_counter % (self.config["process_every"] * 2) == 0:
                holistic_results = self.holistic.process(rgb)
                phone_detected = self.detect_phone_usage(frame, holistic_results)
                
                # Update phone detection state
                self.current_phone_detected = phone_detected
                
                # Phone detection counting
                if phone_detected and now - self.last_phone_check > self.config["phone_detection_interval"]:
                    self.phone_detection_count += 1
                    self.speak("Phone usage detected")
                    self.log_event("Phone Detected", "Potential phone usage")
                    self.last_phone_check = now
            
            # No face detection
            if face_count == 0:
                if now - self.last_no_face_warning > self.config["no_face_cooldown"]:
                    self.speak("No face detected")
                    self.log_event("No Face", "Student not visible")
                    self.last_no_face_warning = now
            
            # Single face processing
            elif face_count == 1:
                lm = face_results.multi_face_landmarks[0].landmark
                l_pts = self.eye_coords(lm, self.LEFT_EYE, w, h)
                r_pts = self.eye_coords(lm, self.RIGHT_EYE, w, h)
                l_iris = self.iris_center(lm, self.LEFT_IRIS, w, h)
                r_iris = self.iris_center(lm, self.RIGHT_IRIS, w, h)
                
                # Blink detection
                ear = (self.eye_aspect_ratio(l_pts) + self.eye_aspect_ratio(r_pts)) / 2.0
                if ear < self.config["ear_threshold"]:
                    blink_detected = True
                
                # Gaze detection
                else:
                    l_xr, l_yr = self.gaze_ratio(l_iris, self.eye_bbox(l_pts))
                    r_xr, r_yr = self.gaze_ratio(r_iris, self.eye_bbox(r_pts))
                    xr = (l_xr + r_xr) * 0.5
                    yr = (l_yr + r_yr) * 0.5
                    cur_dir = self.direction_from_ratio(xr, yr)
                    
                    # Smooth direction transitions
                    if cur_dir != self.prev_gaze_dir:
                        if self.prev_gaze_dir == "Center" or cur_dir == "Center":
                            direction = cur_dir
                        else:
                            direction = self.prev_gaze_dir
                    else:
                        direction = cur_dir
                    self.prev_gaze_dir = cur_dir
                    
                    # Gaze violation
                    if direction != "Center":
                        if now - self.last_gaze_warning > self.config["gaze_warning_cooldown"]:
                            self.speak(f"Looking {direction.lower()[5:]}")
                            self.log_event(direction, f"x:{xr:.2f} y:{yr:.2f}")
                            self.last_gaze_warning = now
            
            # Background noise detection
            if (self.audio_available and self.audio_monitor.noise_detected and 
                now - self.last_no_face_warning > self.config["no_face_cooldown"]):
                self.speak("Background noise detected")
                self.log_event("Background Noise", "Audio threshold exceeded")
                self.last_no_face_warning = now
        
        # Update attention history
        self.attention_history.append("Center" if direction == "Center" else "Away")
        
        # Store current state for UI
        self.current_gaze_dir = direction
        self.current_blink_detected = blink_detected
        self.current_frame = frame
        
        return face_count, direction, blink_detected, phone_detected

    def run(self):
        """Main processing loop"""
        # Start background recorders/monitors
        try:
            if self.screen_recorder and not self.screen_recorder.is_alive():
                self.screen_recorder.start()
        except:
            print("Screen recorder failed to start.")
        try:
            if self.screenshot_capturer and not self.screenshot_capturer.is_alive():
                self.screenshot_capturer.start()
        except:
            print("Screenshot capturer failed to start.")
        try:
            if self.input_monitor:
                self.input_monitor.start()
        except:
            print("Input monitor failed to start.")
        try:
            if self.browser_monitor and not self.browser_monitor.is_alive():
                self.browser_monitor.start()
        except:
            print("Browser monitor failed to start.")

        try:
            while self.cap.isOpened():
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)

                # Write webcam frame to recording file
                if hasattr(self, "webcam_writer") and self.webcam_writer.isOpened():
                    try:
                        self.webcam_writer.write(frame)
                    except Exception as e:
                        print(f"Webcam writer error: {e}")
                
                h, w, _ = frame.shape
                
                # Create UI canvas
                disp = np.zeros((h, w, 3), dtype=np.uint8)
                disp[:] = self.theme.colors['background']
                
                # Process frame
                self.process_frame(frame)
                
                # Draw exact layout (UI unchanged)
                self.draw_top_bar(disp, w, h)
                self.draw_status_panel(disp, w, h)
                self.draw_center_panel(disp, w, h, frame)
                self.draw_summary_panel(disp, w, h)
                self.draw_bottom_bar(disp, w, h)
                
                # Display
                cv2.imshow("AI Proctoring Monitor", disp)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('p'):
                    # Toggle preview mode if needed (placeholder)
                    pass
                
                # Update FPS history
                self.fps_history.append(time.time())
                self.frame_counter += 1
                
        except KeyboardInterrupt:
            print("\nSession interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources and run ffmpeg post-processing"""
        print("\n" + "="*70)
        print(f" 📊 SESSION SUMMARY – {self.student_id}")
        print("="*70)
        
        duration = timedelta(seconds=int(time.time() - self.session_start))
        
        # Print enhanced summary
        print("\n🔍 DETECTION COUNTS:")
        print("-" * 40)
        print(f"  Multiple People Events: {self.multiple_people_count}")
        print(f"  Phone Detection Events: {self.phone_detection_count}")
        print(f"  Total Face Frames: {self.current_face_count}")
        
        print("\n🚨 VIOLATION SUMMARY:")
        print("-" * 40)
        for k, v in self.violation_counts.items():
            print(f"  {k:20}: {v}")
        
        total = sum(self.violation_counts.values())
        print("-" * 40)
        print(f"  {'Total Violations':20}: {total}")
        print(f"  ⏱️  Duration: {duration}")
        print(f"  📁 Log: {self.log_file}")
        print(f"  💾 Recordings folder: {self.recordings_folder}")
        print("="*70)
        
        # Final log entry
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), self.student_id, 
                        "SESSION_END", f"Duration: {duration}"])
            writer.writerow(["", self.student_id, "DETECTION_SUMMARY", f"Multiple People: {self.multiple_people_count}"])
            writer.writerow(["", self.student_id, "DETECTION_SUMMARY", f"Phone Detections: {self.phone_detection_count}"])
        
        # Generate PDF report
        try:
            report_gen = ReportGenerator(self.config["reports_folder"])
            session_data = {
                "start_time": datetime.fromtimestamp(self.session_start),
                "student_id": self.student_id
            }
            
            # Add detection counts to violation counts for report
            enhanced_counts = self.violation_counts.copy()
            enhanced_counts["Multiple People Events"] = self.multiple_people_count
            enhanced_counts["Phone Detection Events"] = self.phone_detection_count
            
            report_file = report_gen.generate_pdf_report(
                self.student_id, session_data, enhanced_counts, 
                self.event_history, str(duration)
            )
            print(f"  📄 Report: {report_file}")
        except Exception as e:
            print(f"  ❌ Report generation failed: {e}")
        
        # Stop screen/screenshot/browser monitors
        try:
            if hasattr(self, "screen_recorder") and self.screen_recorder:
                self.screen_recorder.stop()
        except Exception as e:
            print(f"  Screen recorder stop error: {e}")
        try:
            if hasattr(self, "screenshot_capturer") and self.screenshot_capturer:
                self.screenshot_capturer.stop()
        except Exception as e:
            print(f"  Screenshot capturer stop error: {e}")
        try:
            if hasattr(self, "browser_monitor") and self.browser_monitor:
                self.browser_monitor.stop()
        except Exception as e:
            print(f"  Browser monitor stop error: {e}")
        try:
            if hasattr(self, "input_monitor") and self.input_monitor:
                self.input_monitor.stop()
        except Exception as e:
            print(f"  Input monitor stop error: {e}")
        
        # Release webcam writer and camera
        try:
            if hasattr(self, "webcam_writer") and self.webcam_writer:
                self.webcam_writer.release()
        except:
            pass
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()
        self.holistic.close()
        self.hands.close()
        
        if self.audio_available:
            self.audio_monitor.stop_monitoring()
        
        # Stop TTS
        with self.speech_lock:
            self.speech_queue.append("__STOP__")
        self.stop_speech.set()
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=2.0)
        
        try:
            self.engine.stop()
        except:
            pass

        # Post-process recordings with ffmpeg (efficient storage)
        try:
            # Webcam video
            if os.path.exists(self.webcam_record_path):
                webcam_out = self.webcam_record_path.replace(".avi", ".mp4")
                ffmpeg.input(self.webcam_record_path).output(
                    webcam_out,
                    vcodec="libx264",
                    preset="fast",
                    pix_fmt="yuv420p"
                ).overwrite_output().run(quiet=True)
                print(f"  🎥 Webcam recording: {webcam_out}")
            # Screen video
            if os.path.exists(self.screen_record_path):
                screen_out = self.screen_record_path.replace(".avi", ".mp4")
                ffmpeg.input(self.screen_record_path).output(
                    screen_out,
                    vcodec="libx264",
                    preset="fast",
                    pix_fmt="yuv420p"
                ).overwrite_output().run(quiet=True)
                print(f"  🖥️ Screen recording: {screen_out}")
            # Audio
            if os.path.exists(self.audio_record_path):
                audio_out = self.audio_record_path.replace(".wav", ".m4a")
                ffmpeg.input(self.audio_record_path).output(
                    audio_out,
                    acodec="aac",
                    audio_bitrate="128k"
                ).overwrite_output().run(quiet=True)
                print(f"  🎧 Audio recording: {audio_out}")
        except Exception as e:
            print(f"  ❌ ffmpeg post-processing failed: {e}")
        
        print("✅ Proctoring monitor stopped successfully.")

# ================================
# MAIN EXECUTION
# ================================
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    print("🚀 Starting Improved AI Proctoring System...")
    print("📷 Initializing camera and enhanced detection models...")
    
    try:
        proctor = ImprovedProctorMonitor()
        print("✅ System initialized successfully!")
        print("🎯 Starting proctoring session...\n")
        proctor.run()
    except Exception as e:
        print(f"❌ Error starting proctoring monitor: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Ensure camera is connected and not in use by other applications")
        print("   2. Install all dependencies:")
        print("      pip install opencv-python mediapipe pyttsx3 numpy pyaudio matplotlib mss pyautogui pynput ffmpeg-python selenium")
        print("   3. On macOS, grant camera/microphone/screen recording permissions to Terminal or your IDE")
        print("   4. Ensure Chrome and the correct ChromeDriver are installed for selenium.")