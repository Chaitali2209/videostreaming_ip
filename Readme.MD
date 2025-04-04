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