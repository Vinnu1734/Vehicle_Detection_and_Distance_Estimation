# Object_Detection_and_Distance_Estimation
# Object Detection and Distance Estimation

## Cloning the Repository
To clone this repository, run the following command:
```sh
git clone https://github.com/Vinnu1734/Object_Detection_and_Distance_Estimation.git
cd Lane_Detection
```

## Installing Prerequisites
Before running the code, install the required dependencies:
```sh
pip install numpy opencv-python ultralytics
```

## Running the Code
You can run the following scripts based on your output preference:

- **For frames output:**
  ```sh
  python frames.py
  ```

- **For video output:**
  ```sh
  python video.py
  ```

## Modifications
Make sure to update the location of `weight.pt` and the input video file in both `video.py` and `frames.py` before running the scripts.

## Description
- `frames.py`: Extracts frames from the video and performs object detection.
- `video.py`: Processes the video and performs object detection in real-time.
