
import cv2
import subprocess
import time
from pymavlink import mavutil
from datetime import datetime
import torch
import sys
import numpy as np
import os
import csv
import math
from pyproj import Geod

# === YOLOv8 Setup ===
sys.path.insert(0, "./yolov5")  # for Annotator from YOLOv5 if used
from ultralytics import YOLO
from utils.plots import Annotator, colors

model = YOLO('yolo11n.pt')  # Replace with your trained YOLOv8 model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

# === Connect to MAVLink ===
mav = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)
mav.wait_heartbeat()
print("✅ MAVLink: Connected to FCU")

# === Camera Setup ===
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# === RTSP Streaming Setup ===
ffmpeg_cmd = [
    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24", "-s", "1280x720", "-r", "30", "-i", "-",
    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
    "-f", "rtsp", "-rtsp_transport", "tcp", "rtsp://192.168.146.160:8554/stream3"
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# === Logging Setup ===
os.makedirs("logged_telemetry", exist_ok=True)
telemetry_log_file = os.path.join("logged_telemetry", f"geo_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with open(telemetry_log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Object", "Latitude", "Longitude"])

# === Geodetic Projection ===
geod = Geod(ellps='WGS84')

# === Camera Parameters ===
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
HFOV = 81.0
VFOV = 53.0
cx = IMAGE_WIDTH / 2
cy = IMAGE_HEIGHT / 2

def get_angle_offsets(u, v):
    dx = (u - cx) / cx
    dy = (v - cy) / cy
    angle_x = dx * (HFOV / 2)
    angle_y = dy * (VFOV / 2)
    return math.radians(angle_x), math.radians(angle_y)

def compute_ground_offset(angle_x, angle_y, altitude):
    x = altitude * math.tan(angle_x)
    y = altitude * math.tan(angle_y)
    return x, y  # Camera frame

def rotate_by_yaw(x, y, yaw_deg):
    yaw_rad = math.radians(yaw_deg)
    x_enu = x * math.cos(yaw_rad) - y * math.sin(yaw_rad)
    y_enu = x * math.sin(yaw_rad) + y * math.cos(yaw_rad)
    return x_enu, y_enu

def pixel_to_geo(x, y, lat_c, lon_c, alt, yaw_deg):
    angle_x, angle_y = get_angle_offsets(x, y)
    x_offset, y_offset = compute_ground_offset(angle_x, angle_y, alt)
    x_enu, y_enu = rotate_by_yaw(x_offset, y_offset, yaw_deg)
    azimuth = math.degrees(math.atan2(x_enu, y_enu))
    distance = math.hypot(x_enu, y_enu)
    lon_target, lat_target, _ = geod.fwd(lon_c, lat_c, azimuth, distance)
    return lat_target, lon_target

# === Telemetry Initialization ===
last_alt = last_lat = last_lon = last_yaw = "N/A"
frame_count = 0
log_interval = 10

# === Main Loop ===
while True:
    # Update MAVLink telemetry
    while True:
        msg = mav.recv_match(blocking=False)
        if not msg:
            break
        if msg.get_type() == "GLOBAL_POSITION_INT":
            last_alt = msg.alt / 1000.0  # in meters
            last_lat = msg.lat / 1e7
            last_lon = msg.lon / 1e7
        elif msg.get_type() == "ATTITUDE":
            last_yaw = msg.yaw * (180.0 / math.pi)  # in degrees

    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        continue

    # YOLOv8 Inference
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=rgb_frame, device=device, verbose=False)

    annotator = Annotator(frame, line_width=2, example=str(model.names))
    objects_in_frame = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            label = f"{model.names[cls]} {conf:.2f}"

            x_center = int((xyxy[0] + xyxy[2]) / 2)
            y_center = int((xyxy[1] + xyxy[3]) / 2)

            try:
                lat_obj, lon_obj = pixel_to_geo(
                    x_center, y_center,
                    float(last_lat), float(last_lon),
                    float(last_alt), float(last_yaw)
                )
                label += f"\n({lat_obj:.6f}, {lon_obj:.6f})"
                objects_in_frame.append((model.names[cls], lat_obj, lon_obj))
            except Exception as e:
                print(f"❌ Geolocation error: {e}")

            annotator.box_label(xyxy, label, color=colors(cls, True))

    # Overlay telemetry
    timestamp = datetime.now().strftime("%H:%M:%S")
    overlay = f"[{timestamp}] Alt: {last_alt:.2f}m | Yaw: {last_yaw:.2f}° | Lat: {last_lat:.6f} | Lon: {last_lon:.6f}"
    cv2.putText(frame, overlay, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Stream frame
    ffmpeg_proc.stdin.write(frame.tobytes())

    # Logging
    frame_count += 1
    if frame_count % log_interval == 0:
        log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(telemetry_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for obj_name, obj_lat, obj_lon in objects_in_frame:
                writer.writerow([log_time, obj_name, obj_lat, obj_lon])

    # Optional: Uncomment to preview locally
    # cv2.imshow("Frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()
