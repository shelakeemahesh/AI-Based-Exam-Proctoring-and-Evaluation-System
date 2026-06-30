# 🎓 AI-Based Exam Proctoring and Evaluation System

An intelligent, AI-powered exam proctoring system built with Python that monitors students in real-time during online examinations using computer vision, audio analysis, and behavioral tracking.

> **4th Year College Project** — Designed and developed as a final-year engineering project.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 👁️ **Gaze Tracking** | Detects if the student is looking away from the screen using MediaPipe face mesh landmarks |
| 👥 **Multi-Person Detection** | Alerts when more than one person is detected in the camera frame |
| 📱 **Phone Detection** | Identifies if a mobile phone is visible in the webcam feed |
| 🎤 **Audio Monitoring** | Captures ambient audio and flags suspicious noise levels |
| 🖥️ **Screen Recording** | Records the student's screen throughout the exam session |
| ⌨️ **Input Logging** | Monitors keyboard and mouse activity for unusual patterns |
| 🌐 **Browser Lockdown** | Restricts navigation to allowed exam domains only |
| 📊 **PDF Report Generation** | Generates comprehensive post-exam proctoring reports with charts and violation logs |
| 🔊 **Voice Alerts** | Real-time text-to-speech warnings for detected violations |
| ⚙️ **Configurable** | All thresholds and settings are customizable via `proctor_config.json` |

---

## 🛠️ Tech Stack

- **Language:** Python
- **Computer Vision:** OpenCV, MediaPipe
- **Audio:** PyAudio
- **Screen Capture:** MSS, PyAutoGUI
- **Input Monitoring:** Pynput
- **Browser Automation:** Selenium
- **Video Processing:** FFmpeg
- **Reporting:** Matplotlib, PDF Generation
- **TTS Engine:** Pyttsx3

---

## 📁 Project Structure

```
AI-Based-Exam-Proctoring-and-Evaluation-System/
├── ok.py                          # Main proctoring system (full-stack, single-file)
├── proctor_config.json            # Configuration file (thresholds, settings)
├── presentation/
│   └── AI Presentation.pdf        # Project presentation slides
├── paper/
│   └── Research Paper publish.pdf  # Published research paper
├── output screen shot/            # System output screenshots
│   └── *.png                      # 6 screenshots demonstrating system in action
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

All system parameters can be tuned via [`proctor_config.json`](proctor_config.json):

```json
{
    "gaze_warning_cooldown": 2.5,
    "multi_person_cooldown": 3.0,
    "no_face_cooldown": 5.0,
    "audio_threshold": 500,
    "ear_threshold": 0.2,
    "screenshot_interval_sec": 30,
    "screen_record_fps": 10
}
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Webcam
- Microphone
- FFmpeg installed on your system

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shelakeemahesh/AI-Based-Exam-Proctoring-and-Evaluation-System.git
   cd AI-Based-Exam-Proctoring-and-Evaluation-System
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python mediapipe numpy pyttsx3 pyaudio matplotlib mss pyautogui pynput ffmpeg-python selenium
   ```

3. **Run the proctoring system:**
   ```bash
   python ok.py
   ```

---

## 📄 Documentation

- 📑 [**Project Presentation**](presentation/AI%20Presentation.pdf) — Overview slides covering architecture, features, and demo
- 📝 [**Research Paper**](paper/Research%20Paper%20publish.pdf) — Published research paper with methodology, results, and analysis

---

## 📸 Output Screenshots

| Screenshot | Preview |
|---|---|
| Proctoring System - View 1 | ![Screenshot 1](output%20screen%20shot/Screenshot%202026-06-24%20at%2011.16.12%E2%80%AFPM.png) |
| Proctoring System - View 2 | ![Screenshot 2](output%20screen%20shot/Screenshot%202026-06-24%20at%2011.16.26%E2%80%AFPM.png) |
| Proctoring System - View 3 | ![Screenshot 3](output%20screen%20shot/Screenshot%202026-06-24%20at%2011.16.32%E2%80%AFPM.png) |
| Proctoring System - View 4 | ![Screenshot 4](output%20screen%20shot/Screenshot%202026-06-24%20at%2011.16.42%E2%80%AFPM.png) |
| Proctoring System - View 5 | ![Screenshot 5](output%20screen%20shot/Screenshot%202026-06-24%20at%2011.16.57%E2%80%AFPM.png) |
| Proctoring System - View 6 | ![Screenshot 6](output%20screen%20shot/Screenshot%202026-06-24%20at%2011.19.43%E2%80%AFPM.png) |

---

## 📊 How It Works

1. **Initialization** — System loads configuration, initializes camera, microphone, and screen recorder
2. **Real-Time Monitoring** — Continuously analyzes video feed for gaze direction, face count, and phone presence
3. **Audio Analysis** — Monitors ambient sound levels and flags anomalies
4. **Screen & Input Tracking** — Records screen activity and monitors keyboard/mouse inputs
5. **Violation Logging** — All violations are timestamped and categorized
6. **Report Generation** — At the end of the exam, a detailed PDF report is generated with charts and logs

---

## 👤 Author

**Mahesh Shelake**

- GitHub: [@shelakeemahesh](https://github.com/shelakeemahesh)

---

## 📜 License

This project is developed as part of an academic requirement. Please contact the author for reuse or collaboration.
