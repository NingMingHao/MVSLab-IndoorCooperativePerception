# MVSLab-IndoorCooperativePerception
Official Repo of ITSC2024: Enhancing Indoor Mobility with Connected Sensor Nodes: A Real-Time, Delay-Aware Cooperative Perception Approach

For questions please contact Minghao Ning, minghao.ning@uwaterloo.ca.

### Overview of the project
<iframe width="560" height="315" src="https://www.youtube.com/embed/YsNX8Ubo7pk?si=ab3iLMfKaPM2Ztsu" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Dataset
The dataset is stored in ROS bag format. The required camera parameters are saved in camera_info, sensor transformations are saved in tf, so you can easily visualize each dataset via Rviz. The dataset is available at [Google Drive](https://drive.google.com/drive/folders/1_2UOGiY8OORMtG4qc-np3rAN6A3O275M?usp=sharing).

The dataset is labeled with CVAT, the labels are oriented bounding boxes in the BEV (Bird's Eye View) format. An example is shown in
<iframe width="560" height="315" src="https://www.youtube.com/embed/UVUcsEo8Olw?si=lR1AxJ_omYrJEtYZ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Results
### Local Perception for varying density case
<iframe width="560" height="315" src="https://www.youtube.com/embed/yXRH-UHOn6I?si=Cf6x946FOw7mfgfk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Local Perception for crowded case
<iframe width="560" height="315" src="https://www.youtube.com/embed/bZyYKFu86tI?si=YEx1epUDQK8qd3yc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Global Perception for 3People1Bed
<iframe width="560" height="315" src="https://www.youtube.com/embed/ESNUPyDQLdw?si=4yiqX2l6RYHPSye1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### Global Perception for 9People
<iframe width="560" height="315" src="https://www.youtube.com/embed/y_sCzz1o6HM?si=1mjJ9oofWXOuXlpd" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>