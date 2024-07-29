# MVSLab-IndoorCooperativePerception
Official Repo of ITSC2024: "Enhancing Indoor Mobility with Connected Sensor Nodes: A Real-Time, Delay-Aware Cooperative Perception Approach" Authored by, Minghao Ning, Yaodong Cui, Yufeng Yang, Shucheng Huang, Zhenan Liu, Ahmad Reza Alghooneh, Ehsan Hashemi and Amir Khajepour. 

For questions please contact Minghao Ning, minghao.ning@uwaterloo.ca.

### Overview of the project
[![Watch the video](https://img.youtube.com/vi/YsNX8Ubo7pk/maxresdefault.jpg)](https://youtu.be/YsNX8Ubo7pk)

## Dataset
The dataset is stored in ROS bag format. The required camera parameters are saved in camera_info, sensor transformations are saved in tf, so you can easily visualize each dataset via Rviz. The dataset is available at [Google Drive](https://drive.google.com/drive/folders/1_2UOGiY8OORMtG4qc-np3rAN6A3O275M?usp=sharing).

The dataset is labeled with CVAT, the labels are oriented bounding boxes in the BEV (Bird's Eye View) format. An example is shown in:
[![Watch the video](https://img.youtube.com/vi/UVUcsEo8Olw/maxresdefault.jpg)](https://youtu.be/UVUcsEo8Olw)

## Results
### Local Perception for varying density case
[![Watch the video](https://img.youtube.com/vi/yXRH-UHOn6I/maxresdefault.jpg)](https://youtu.be/yXRH-UHOn6I)

### Local Perception for crowded case
[![Watch the video](https://img.youtube.com/vi/bZyYKFu86tI/maxresdefault.jpg)](https://youtu.be/bZyYKFu86tI)

### Global Perception for 3People1Bed
[![Watch the video](https://img.youtube.com/vi/ESNUPyDQLdw/maxresdefault.jpg)](https://youtu.be/ESNUPyDQLdw)

### Global Perception for 9People
[![Watch the video](https://img.youtube.com/vi/y_sCzz1o6HM/maxresdefault.jpg)](https://youtu.be/y_sCzz1o6HM)
