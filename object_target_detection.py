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

# === YOLOv8 Setup ===
sys.path.insert(0, "./yolov5")
from ultralytics import YOLO
from utils.plots import Annotator, colors

model = YOLO('yolo11n.pt')  # Replace with your model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

classes_to_detect = [0]

# === Connect to MAVLink ===
mav = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)
mav.wait_heartbeat()
print("✅ MAVLink: Connected to FCU")

# === Camera Setup ===
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# === RTSP Streaming ===
ffmpeg_cmd = [
    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo","-probesize", "32",
    "-analyzeduration", "1000000",
    "-pix_fmt", "bgr24", "-s", "1280x720", "-r", "30", "-i", "-",
    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
    "-f", "rtsp", "-rtsp_transport", "tcp", "rtsp://192.168.0.130:8554/stream3"
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# === Logging Setup ===
os.makedirs("logged_frames", exist_ok=True)
# base_dir = "logged_frames/"
# date_str = datetime.now().strftime("%Y%m%d")
# mission_prefix = f"captured_frames_"

# mission_dir = os.path.join(base_dir, f"{mission_prefix}_{date_str}")

# os.makedirs(f"{mission_prefix}_{date_str}")
# print(f"✅ Saving frames to: {mission_dir}")

os.makedirs("logged_telemetry", exist_ok=True)
telemetry_log_file = os.path.join("logged_telemetry", f"target_object_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with open(telemetry_log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Object", "Target_Latitude", "Target_Longitude", "Global_Latitude", "Global_Longitude", "Altitude", "Yaw", "Pixel_X", "Pixel_Y"])

# === Pixel to GPS Conversion Function ===
def pixel_to_geo(x, y, lat_c, lon_c, h):
    # IFOV (instantaneous field of view) in radians/pixel
    hfov_rad = math.radians(81)
    vfov_rad = math.radians(57)
    image_width = 1280
    image_height = 720
    IFOV_h = hfov_rad / image_width
    IFOV_v = vfov_rad / image_height

    R = 6371000  # Earth radius (meters)
    center_x = image_width / 2
    center_y = image_height / 2

    delta_theta_h = (x - center_x) * IFOV_h
    delta_theta_v = (y - center_y) * IFOV_v

    d_x = h * math.tan(delta_theta_h)
    d_y = h * math.tan(delta_theta_v)

    delta_lat = (d_y / R) * (180 / math.pi)
    delta_lon = (d_x / (R * math.cos(math.radians(lat_c)))) * (180 / math.pi)

    lat_target = lat_c + delta_lat
    lon_target = lon_c + delta_lon

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
            last_alt = f"{msg.relative_alt / 1000.0:.2f}m"
            last_lat = f"{msg.lat / 1e7:.6f}"
            last_lon = f"{msg.lon / 1e7:.6f}"
        elif msg.get_type() == "ATTITUDE":
            last_yaw = f"{msg.yaw * (180.0 / math.pi):.2f}"

    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        continue

    # YOLOv8 Inference
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=rgb_frame, device=device, classes=classes_to_detect, verbose=False)
    
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
                alt = float(last_alt.replace('m', ''))
                lat_c = float(last_lat)
                lon_c = float(last_lon)

                lat_obj, lon_obj = pixel_to_geo(x_center, y_center, lat_c, lon_c, alt)
                label += f"\n({lat_obj:.6f}, {lon_obj:.6f})"
                objects_in_frame.append((model.names[cls], lat_obj, lon_obj))

            except Exception as e:
                print(f"❌ Geolocation error: {e}")

            annotator.box_label(xyxy, label, color=colors(cls, True))

    # Overlay telemetry
    timestamp = datetime.now().strftime("%H:%M:%S")
    overlay = f"[{timestamp}] Alt: {last_alt} | Yaw: {last_yaw} | Lat: {last_lat} | Lon: {last_lon}"
    cv2.putText(frame, overlay, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Stream frame
    ffmpeg_proc.stdin.write(frame.tobytes())

    # Logging
    frame_count += 1
    if frame_count % log_interval == 0:
        log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save frame
        cv2.imwrite(f"logged_frames/frame_{log_time}.jpg", frame)

        # Save telemetry
        with open(telemetry_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for obj_name, obj_lat, obj_lon in objects_in_frame:
                writer.writerow([log_time, obj_name, obj_lat, obj_lon, last_lat, last_lon, last_alt, last_yaw, x_center, y_center])

    # Optional local display (uncomment to view)
    # cv2.imshow("Frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
