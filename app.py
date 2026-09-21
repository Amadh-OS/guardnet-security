import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from ultralytics import YOLO
import av

# Page Configuration
st.set_page_config(page_title="GuardNet Security", layout="wide")

st.title("🛡️ GuardNet: AI Perimeter & Airspace Defense")
st.caption("Software-Native Edge-AI | 0 SAR Setup")

# 1. Load Model (YOLOv8 Nano - optimized for speed)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt") 

model = load_model()

# 2. Google STUN server configuration for smooth streaming on public Wi-Fi
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 3. Threat detection class IDs (0=person, 4=airplane/drone proxy, 67=cell phone for testing)
THREAT_CLASSES = [0, 4, 67] 

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Run YOLO detection
    results = model(img, conf=0.40)
    
    threat_detected = False
    detected_labels = []

    # Analyze detected objects
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            # Check if detected object is a monitored threat
            if cls_id in THREAT_CLASSES:
                threat_detected = True
                if label not in detected_labels:
                    detected_labels.append(label)

    # Plot detection bounding boxes
    annotated_frame = results[0].plot()

    # 4. Draw Live HUD Security Overlay on Video
    height, width, _ = annotated_frame.shape

    if threat_detected:
        # Draw RED Alert Banner at the top of the video
        cv2.rectangle(annotated_frame, (0, 0), (width, 60), (0, 0, 220), -1)
        alert_text = f"THREAT DETECTED: {', '.join(detected_labels).upper()}"
        cv2.putText(
            annotated_frame, alert_text, (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
        )
    else:
        # Draw GREEN Secure Banner at the top of the video
        cv2.rectangle(annotated_frame, (0, 0), (width, 50), (0, 150, 0), -1)
        cv2.putText(
            annotated_frame, "SYSTEM SECURE - PERIMETER MONITORED", (20, 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 5. Live Camera Streamer
webrtc_streamer(
    key="guardnet-hud",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
