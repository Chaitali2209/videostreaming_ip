import cv2
import subprocess
import time
from pymavlink import mavutil
from datetime import datetime
import torch
import sys
import numpy as np

# === Import YOLOv8 ===
sys.path.insert(0, "./yolov5")
from ultralytics import YOLO
from utils.plots import Annotator, colors  # still using YOLOv5's visualizer for easy box drawing

# === Load YOLOv8 Model ===
model = YOLO('yolo11n.pt')  # use yolov8n/yolov8s etc.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

# === Connect to MAVLink ===
mav = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)
mav.wait_heartbeat()
print("✅ MAVLink: Connected to FCU")

# === Set up Video Capture ===
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# === Set up FFmpeg for RTSP Streaming ===
ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", "1280x720",
    "-r", "30",
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    "rtsp://192.168.104.160:8554/stream1"
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# === Initialize telemetry variables ===
last_alt = last_lat = last_lon = last_yaw = "N/A"

# === Main Loop ===
while True:
    # MAVLink telemetry (non-blocking)
    while True:
        msg = mav.recv_match(blocking=False)
        if not msg:
            break
        if msg.get_type() == "GLOBAL_POSITION_INT":
            last_alt = f"{msg.alt / 1000.0:.2f}m"
            last_lat = f"{msg.lat / 1e7:.6f}"
            last_lon = f"{msg.lon / 1e7:.6f}"
        elif msg.get_type() == "ATTITUDE":
            last_yaw = f"{msg.yaw * (180.0 / 3.14159):.2f}"

    # === Capture frame ===
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        continue

    # === YOLOv8 Inference ===
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=rgb_frame, device=device, verbose=False)

    # === Annotate Detections ===
    annotator = Annotator(frame, line_width=2, example=str(model.names))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            label = f"{model.names[cls]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(cls, True))
    frame = annotator.result()

    # === Overlay Telemetry ===
    timestamp = datetime.now().strftime("%H:%M:%S")
    overlay = f"[{timestamp}] Alt: {last_alt} | Yaw: {last_yaw} | Lat: {last_lat} | Lon: {last_lon}"
    cv2.putText(frame, overlay, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # === Send frame to FFmpeg ===
    ffmpeg_proc.stdin.write(frame.tobytes())

    # === Optional debug display ===
    # cv2.imshow("Debug", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
