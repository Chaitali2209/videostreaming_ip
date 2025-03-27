import subprocess
import time
from pymavlink import mavutil

# Connect to Cube Orange telemetry port
mav = mavutil.mavlink_connection("/dev/ttyACM0", baud=115200)

# Wait for heartbeat to confirm connection
mav.wait_heartbeat()
print("Heartbeat received. Connected to flight controller.")

def set_gimbal_pitch(pitch_angle):
    """ Set gimbal pitch angle using MAVLink command """
    mav.mav.command_long_send(
        mav.target_system,  # System ID
        mav.target_component,  # Component ID
        mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,  # Command to control gimbal
        0,  # Confirmation
        pitch_angle,  # Pitch angle (degrees)
        0,  # Roll (0 to maintain level)
        0,  # Yaw (0 to keep centered)
        0,  # Altitude (not used)
        0,  # Latitude (not used)
        0,  # Longitude (not used)
        mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING,  # Gimbal mode
    )
    print(f"Gimbal set to pitch {pitch_angle}°")

# Set gimbal to -90 degrees (perpendicular to ground)
set_gimbal_pitch(-90)

# Start FFmpeg with dynamic drawtext input
ffmpeg_cmd = [
    "ffmpeg",
    "-i", "/dev/video0",
    "-vf", "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:textfile=/tmp/telem_data.txt:reload=1:x=10:y=50:fontsize=24:fontcolor=white",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-fflags", "nobuffer",
    "-flags", "low_delay",
    "-g", "30",
    "-keyint_min", "30",
    "-an",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    "rtsp://192.168.0.130:8554/stream1"
]

# Start FFmpeg
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd)

# Continuously update altitude and yaw text
while True:
    try:
        # Try to get telemetry messages (non-blocking)
        msg_alt = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        msg_att = mav.recv_match(type="ATTITUDE", blocking=False)

        # Only update if data is available
        if msg_alt:
            altitude = msg_alt.alt / 1000.0  # Convert mm to meters
        else:
            altitude = "N/A"

        if msg_att:
            yaw = msg_att.yaw * (180.0 / 3.14159)  # Convert radians to degrees
        else:
            yaw = "N/A"

        # Write to file for FFmpeg to read
        with open("/tmp/telem_data.txt", "w") as f:
            f.write(f"Altitude: {altitude}m | Yaw: {yaw}°\n")

    except Exception as e:
        print(f"Error: {e}")

    time.sleep(1)  # Update every second

    # Check if FFmpeg is still running, restart if necessary
    if ffmpeg_proc.poll() is not None:
        print("FFmpeg crashed! Restarting...")
        ffmpeg_proc = subprocess.Popen(ffmpeg_cmd)