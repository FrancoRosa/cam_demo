# rknn object detection demo
## requirements
rknn-toolkit2 virtual environment is required

## activate conda
Put this snipped on a `.sh` file to activate the environment
```bash
#! /bin/bash
cd
source miniforge3/bin/activate
conda activate RKNN-Toolkit2
```

## run on camera or video
You can use the script with the following commands
```bash
python cam_classes.py --camera_id 
python cam_classes.py --camera_id 50 # camera id 50, run v4l2-ctl --device-list
python cam_classes.py --camera_id "ppvideo.mp4" # add the route of a video
```