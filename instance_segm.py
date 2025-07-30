import cv2
import subprocess
import time
from pymavlink import mavutil
from datetime import datetime
import torch
import numpy as np
import os
import csv
import math
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors 

# === YOLOv8 Setup ===
model = YOLO('yolo11n-seg.pt')  # Segmentation model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

# Define the class ID for "person" (COCO dataset class ID for person is 0)
PERSON_CLASS_ID = 0

# === MAVLink Connection ===
mav = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)
mav.wait_heartbeat()
print("✅ MAVLink: Connected to FCU")

# === Camera Setup ===
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# === RTSP Streaming ===
ffmpeg_cmd = [
    "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
    "-s", "640x480", "-r", "20", "-i", "-",
    "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
    "-f", "rtsp", "-rtsp_transport", "udp", "rtsp://192.168.0.130:8554/stream1"
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# === Logging Setup ===
os.makedirs("logged_frames", exist_ok=True)
os.makedirs("logged_telemetry", exist_ok=True)
telemetry_log_file = os.path.join("logged_telemetry", f"target_object_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
with open(telemetry_log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Object", "Target_Latitude", "Target_Longitude", "Global_Latitude", "Global_Longitude", "Altitude", "Yaw", "Pixel_X", "Pixel_Y"])

# === Pixel to GPS Conversion Function ===
def pixel_to_geo(x, y, lat_c, lon_c, h):
    hfov_rad = math.radians(81)
    vfov_rad = math.radians(57)
    image_width = 640  # Match camera resolution
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

# === Runtime State ===
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

    # Process detections and annotations
    objects_in_frame = []

     # Create an annotator to draw bounding boxes and masks
    annotator = Annotator(frame, line_width=2)

    results = model.track(frame, persist=True, tracker="botsort.yaml", device=device, classes=[PERSON_CLASS_ID], verbose=False)

    if results and results[0].masks is not None:
        person_mask = results[0].boxes.cls == PERSON_CLASS_ID
        boxes = results[0].boxes[person_mask]
        masks = results[0].masks[person_mask]
        confs = results[0].boxes.conf[person_mask]
        track_ids = results[0].boxes.id[person_mask] if results[0].boxes.id is not None else None

        im_gpu = torch.from_numpy(frame).to(device).permute(2, 0, 1).float() / 255.0

        mask_colors = [colors(int(PERSON_CLASS_ID), bgr=True) for _ in range(len(masks))]

        # Draw segmentation masks
        annotator.masks(
            masks=masks.data,
            colors=mask_colors,
            im_gpu=im_gpu,
            alpha=0.5,
            retina_masks=False
        )
        
        for i, (box, conf, mask) in enumerate(zip(boxes.xyxy, boxes.conf, masks.xy)):
            xyxy = box.cpu().numpy().astype(int)
            x_center = int((xyxy[0] + xyxy[2]) / 2)
            y_center = int((xyxy[1] + xyxy[3]) / 2)
            track_id = int(track_ids[i]) if track_ids is not None and len(track_ids) > i else -1
            try:
                alt = float(last_alt.replace('m', ''))
                lat_c = float(last_lat)
                lon_c = float(last_lon)
                lat_obj, lon_obj = pixel_to_geo(x_center, y_center, lat_c, lon_c, alt)
                label = f"Person ID {track_id} {conf:.2f} ({lat_obj:.6f}, {lon_obj:.6f})"
                objects_in_frame.append(("Person", lat_obj, lon_obj, x_center, y_center))
            except Exception as e:
                print(f"❌ Geolocation error: {e}")
                label = f"Person {conf:.2f}"

            # Get color for person class
            color = colors(int(PERSON_CLASS_ID), True)
            txt_color = annotator.get_txt_color(color)

            # Annotate with segmentation mask and bounding box
            annotator.box_label(
                box=xyxy,
                label=label,
                color=color,
                txt_color=txt_color
            )

    # Get annotated frame
    annotated_frame = annotator.result()

    # Overlay telemetry
    timestamp = datetime.now().strftime("%H:%M:%S")
    overlay = f"[{timestamp}] Alt: {last_alt} | Yaw: {last_yaw} | Lat: {last_lat} | Lon: {last_lon}"
    cv2.putText(annotated_frame, overlay, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

    # Stream frame
    ffmpeg_proc.stdin.write(annotated_frame.tobytes())

    # Logging
    frame_count += 1
    if frame_count % log_interval == 0:
        log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cv2.imwrite(f"logged_frames/frame_{log_time}.jpg", annotated_frame)
        with open(telemetry_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for obj_name, obj_lat, obj_lon, x_center, y_center in objects_in_frame:
                writer.writerow([log_time, obj_name, obj_lat, obj_lon, last_lat, last_lon, last_alt, last_yaw, x_center, y_center])

# Release resources
cap.release()
ffmpeg_proc.stdin.close()
ffmpeg_proc.terminate()