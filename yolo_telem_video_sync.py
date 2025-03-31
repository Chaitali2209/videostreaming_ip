import cv2
import subprocess
import time
from pymavlink import mavutil
from datetime import datetime
import torch
import sys
import numpy as np

# === Add YOLOv5 to path and import necessary modules ===
sys.path.insert(0, "./yolov5")  # Adjust if yolov5 folder is elsewhere
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# === Load YOLOv5 Model ===
device = select_device('')
model = DetectMultiBackend('yolov5s.pt', device=device)  # Replace with your model path
stride, names = model.stride, model.names
imgsz = (640, 640)  # Inference size

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
    "rtsp://192.168.0.130:8554/stream1"  # Adjust your RTSP server address
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

    # === YOLOv5 Preprocessing ===
    img = letterbox(frame, imgsz, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # === Inference and NMS ===
    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # === Annotate Frame ===
    annotator = Annotator(frame, line_width=2, example=str(names))
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{names[int(cls)]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
    frame = annotator.result()

    # === Overlay Telemetry ===
    timestamp = datetime.now().strftime("%H:%M:%S")
    overlay = f"[{timestamp}] Alt: {last_alt} | Yaw: {last_yaw} | Lat: {last_lat} | Lon: {last_lon}"
    cv2.putText(frame, overlay, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # === Send to FFmpeg ===
    ffmpeg_proc.stdin.write(frame.tobytes())

    # === Optional Debug Preview ===
    # cv2.imshow("Debug", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
