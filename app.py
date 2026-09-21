import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Page Configuration
st.set_page_config(page_title="GuardNet Security", layout="wide")

st.title("🛡️ GuardNet: AI Perimeter & Airspace Defense")
st.caption("Software-Native Edge-AI | 0 SAR Setup")

# 1. Load Model (YOLOv8 Nano)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt") 

model = load_model()

# Threat detection class IDs (0=person, 4=airplane/drone proxy, 67=cell phone for testing)
THREAT_CLASSES = [0, 4, 67] 

# 2. Camera Input Widget (Native, zero-config mobile support)
camera_file = st.camera_input("Capture Perimeter Feed")

if camera_file is not None:
    # Convert uploaded image file to OpenCV array
    bytes_data = camera_file.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    # Run YOLO detection
    results = model(img, conf=0.40)
    
    threat_detected = False
    detected_labels = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if cls_id in THREAT_CLASSES:
                threat_detected = True
                if label not in detected_labels:
                    detected_labels.append(label)

    # Plot bounding boxes
    annotated_frame = results[0].plot()
    
    # Convert BGR to RGB for Streamlit display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    height, width, _ = annotated_frame.shape

    # 3. Draw Live HUD Security Banner Overlay
    if threat_detected:
        alert_text = f"THREAT DETECTED: {', '.join(detected_labels).upper()}"
        st.error(f"🚨 {alert_text}")
    else:
        st.success("✅ SYSTEM SECURE - PERIMETER MONITORED")

    # Display processed frame
    st.image(annotated_frame, channels="RGB", use_container_width=True)
