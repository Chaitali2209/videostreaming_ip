import cv2
import subprocess
import time
from pymavlink import mavutil
from datetime import datetime

# Connect to MAVLink
mav = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)
mav.wait_heartbeat()
print("Connected to FCU")

# Video capture setup
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FFmpeg RTSP stream setup
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
    "rtsp://192.168.0.130:8554/stream1"
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Initialize telemetry values
last_alt = "N/A"
last_lat = "N/A"
last_lon = "N/A"
last_yaw = "N/A"

while True:
    # Grab latest MAVLink messages (non-blocking)
    while True:
        msg = mav.recv_match(blocking=False)
        if not msg:
            break

        if msg.get_type() == "GLOBAL_POSITION_INT":
            try:
                last_alt = f"{msg.alt / 1000.0:.2f}m"
                last_lat = f"{msg.lat / 1e7:.6f}"
                last_lon = f"{msg.lon / 1e7:.6f}"
            except:
                last_alt = last_lat = last_lon = "N/A"

        elif msg.get_type() == "ATTITUDE":
            try:
                last_yaw = f"{msg.yaw * (180.0 / 3.14159):.2f}°"
            except:
                last_yaw = "N/A"

    # Grab video frame
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed.")
        continue

    # Compose telemetry text
    timestamp = datetime.now().strftime("%H:%M:%S")
    overlay = f"[{timestamp}] Alt: {last_alt} | Yaw: {last_yaw} | Lat: {last_lat} | Lon: {last_lon}"

    # Put text overlay on frame
    cv2.putText(frame, overlay, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Send frame to FFmpeg for RTSP
    ffmpeg_proc.stdin.write(frame.tobytes())

    # Optional preview (for debugging)
    # cv2.imshow("Debug Preview", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

