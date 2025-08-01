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
import socket
import json
import redis

from ultralytics import YOLO
from utils.plots import Annotator, colors

telem_data = redis.Redis()

# === UDP Setup to GCS ===
GCS_IP = "192.168.0.130"  # 🔁 Replace with your actual GCS IP
GCS_PORT = 5005         # 🔁 Port your Qt GCS is listening on
udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

model = YOLO('yolo11n-seg.pt')  # Replace with your model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

# === Connect to MAVLink ===
# mav = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)
# mav.wait_heartbeat()
# print("✅ MAVLink: Connected to FCU")

# === Camera Setup ===
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === RTSP Streaming ===
ffmpeg_cmd = [
    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo","-probesize", "32",
    "-analyzeduration", "1000000",
    "-pix_fmt", "bgr24", "-s", "640x480", "-r", "30", "-i", "-",
    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
    "-f", "rtsp", "-rtsp_transport", "tcp", "rtsp://192.168.0.130:8554/stream4"
]

ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# === Logging Setup ===
# os.makedirs("logged_frames", exist_ok=True)
# base_dir = "logged_frames/"
# date_str = datetime.now().strftime("%Y%m%d")
# mission_prefix = f"captured_frames_"

# mission_dir = os.path.join(base_dir, f"{mission_prefix}_{date_str}")

# os.makedirs(f"{mission_prefix}_{date_str}")
# print(f"✅ Saving frames to: {mission_dir}")

# os.makedirs("logged_telemetry", exist_ok=True)
# telemetry_log_file = os.path.join("logged_telemetry", f"target_object_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
# with open(telemetry_log_file, 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["Timestamp", "Object", "Target_Latitude", "Target_Longitude", "Global_Latitude", "Global_Longitude", "Altitude", "Yaw", "Pixel_X", "Pixel_Y"])

# === Pixel to GPS Conversion Function ===
def pixel_to_geo(x, y, lat_c, lon_c, h):
    # IFOV (instantaneous field of view) in radians/pixel
    hfov_rad = math.radians(81)
    vfov_rad = math.radians(57)
    image_width = 640
    image_height = 480
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
last_alt = last_lat = last_lon = last_yaw = last_satellites =  "N/A"
frame_count = 0
log_interval = 10
self_drone_id = 0

# === Main Loop ===
while True:
    # Update MAVLink telemetry
    
    # === Retrieve telemetry ===
        
    # while True:
        
    try:
        self_drone_id = int(telem_data.get('self_drone_id'))
        last_satellites = int(telem_data.get("sat_count"))

        msg_att = float(telem_data.get("self_yaw"))
        if msg_att:
            last_yaw = f"{float(msg_att) * (180.0 / 3.14159):.2f}"
        
        gps_data = json.loads(telem_data.get('self_gps')) 
    
        if gps_data:
            last_alt = f"{gps_data['alt'] / 1000.0:.2f}m"
            last_lat = f"{gps_data['lat'] / 1e7:.6f}"
            last_lon = f"{gps_data['lon'] / 1e7:.6f}"
            
        # print(f"{self_drone_id} Telemetry: Alt={last_alt}, Yaw={last_yaw}, Lat={last_lat}, Lon={last_lon}, Sats={last_satellites}")
    
    except Exception as e:
        print("Error in telemetry retrieval:", e)


    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame capture failed.")
        continue

     # Process detections and annotations
    objects_in_frame = []

    # Create an annotator to draw bounding boxes and masks
    annotator = Annotator(frame, line_width=2)

    results = model.track(frame, conf=0.35, imgsz=640, half=True, stream=True, persist=True, tracker="bytetrack.yaml", device=device, classes=[0],
                          verbose=False)

    for i, r in enumerate(results):
        annotated_frame = r.plot()

            try:
                alt = float(last_alt.replace('m', ''))
                lat_c = float(last_lat)
                lon_c = float(last_lon)
                sat = float(last_satellites)
                drone_id = float(self_drone_id)

                lat_obj, lon_obj = pixel_to_geo(x_center, y_center, lat_c, lon_c, alt)
                label += f"\n({lat_obj:.6f}, {lon_obj:.6f})"
                objects_in_frame.append((model.names[cls], lat_obj, lon_obj))

                # === Send JSON telemetry to GCS ===
                json_payload = {
                    "drone_id": drone_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "object": model.names[cls],
                    "target_lat": lat_obj,
                    "target_lon": lon_obj,
                    "drone_lat": lat_c,
                    "drone_lon": lon_c,
                    "altitude": alt,
                    "yaw": float(last_yaw),
                    "pixel_x": x_center,
                    "pixel_y": y_center,
                    "sat": sat
                }

                try:
                    udp_sock.sendto(json.dumps(json_payload).encode(), (GCS_IP, GCS_PORT))
                except Exception as e:
                    print(f"❌ UDP send error: {e}")

            except Exception as e:
                print(f"❌ Geolocation error: {e}")

    # Overlay telemetry
    timestamp = datetime.now().strftime("%H:%M:%S")
    overlay = f"[{timestamp}] Alt: {last_alt} | Yaw: {last_yaw} | Lat: {last_lat} | Lon: {last_lon} | Sats: {last_satellites}"
    cv2.putText(frame, overlay, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Stream frame
    ffmpeg_proc.stdin.write(frame.tobytes())

    # # Logging
    # frame_count += 1
    # if frame_count % log_interval == 0:
    #     log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #     # Save frame
    #     cv2.imwrite(f"logged_frames/frame_{log_time}.jpg", frame)

    #     # Save telemetry
    #     with open(telemetry_log_file, 'a', newline='') as f:
    #         writer = csv.writer(f)
    #         for obj_name, obj_lat, obj_lon in objects_in_frame:
    #             writer.writerow([log_time, obj_name, obj_lat, obj_lon, last_lat, last_lon, last_alt, last_yaw, x_center, y_center])


# Release resources
cap.release()
ffmpeg_proc.stdin.close()
ffmpeg_proc.terminate()
