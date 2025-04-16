# commands

## creating a virtual environment
python3 -m venv objectdetect

## set up pytorch and torchvision on jetpack6 with cuda12.6
1. use this link to download - https://pypi.jetson-ai-lab.dev/jp6/cu126
2. pip install /path/to/torch, /path/to/torchvision
3. run python, check if cuda is available and get cuda device name ID

## debugging
1. For serial error - pip install serial, pip install pyserial
2. pip install ultralytics

## if receiving frame capture failed or broken pipe

1. Check for IP address and stream
2. Replace the ffmpeg command with this - 
ffmpeg_cmd = [
    "ffmpeg",
    "-y",
    "-f", "rawvideo",
    "-vcodec", "rawvideo",
    "-probesize", "32",
    "-analyzeduration", "1000000",
    "-pix_fmt", "bgr24",
    "-s", "1280x720",
    "-r", "30",
    "-i", "-",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-f", "rtsp",
    "-rtsp_transport", "tcp",
    "rtsp://192.168.104.160:8554/stream4"
]


