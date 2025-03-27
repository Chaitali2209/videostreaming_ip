import cv2
import subprocess
import time
from pymavlink import mavutil
import torch

# Load YOLOv5 nano model (lightweight for Jetson)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# MAVLink telemetry (Cube Orange via serial)
mav = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)

# Initialize camera (USB or CSI)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Define FFmpeg command (GPU encoder for Jetson)
ffmpeg_cmd = [
    'ffmpeg', '-y',
    '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '640x480', '-r', '30', '-i', '-',
    '-c:v', 'h264_nvenc', '-preset', 'llhp', '-tune', 'zerolatency',
    '-g', '30', '-keyint_min', '30', '-sc_threshold', '0',
    '-b:v', '800k', '-maxrate', '800k', '-bufsize', '1000k',
    '-fflags', 'nobuffer', '-flags', 'low_delay',
    '-f', 'rtsp', '-rtsp_transport', 'tcp',
    'rtsp://192.168.0.130:8554/stream1'  # Replace with your RTSP server address
]

# Start FFmpeg subprocess
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Telemetry defaults
last_alt, last_yaw, last_lat, last_lon = 0.0, 0.0, 0.0, 0.0

print("🔄 Streaming started... Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Telemetry update
        msg_alt = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        msg_att = mav.recv_match(type="ATTITUDE", blocking=False)
        if msg_alt:
            last_alt = msg_alt.alt / 1000.0
            last_lat = msg_alt.lat / 1e7
            last_lon = msg_alt.lon / 1e7
        if msg_att:
            last_yaw = msg_att.yaw * (180.0 / 3.14159)

        # Run object detection
        results = model(frame)
        for *box, conf, cls in results.xyxy[0]:
            label = model.names[int(cls)]
            if label == 'person':
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add telemetry overlay
        overlay_text = f"Alt: {last_alt:.1f}m | Yaw: {last_yaw:.1f}° | Lat: {last_lat:.5f} | Lon: {last_lon:.5f}"
        cv2.putText(frame, overlay_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Pipe frame to FFmpeg
        try:
            ffmpeg_proc.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"❌ FFmpeg pipe error: {e}")
            break

        time.sleep(0.03)  # ~30 FPS

except KeyboardInterrupt:
    print("🛑 Stopping streaming...")

finally:
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.terminate()
    cap.release()

