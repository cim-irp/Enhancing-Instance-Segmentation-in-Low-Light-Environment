# Enhancing-Instance-Segmentation-in-Low-Light-Environment

This repository is the implementation of Enhancing Instance Segmentation in Low-light Environment. Our code is based on detectron2.

**Features**
- 


Object detection and segmentation are the critical component in the automotive industry, especially for applications like autonomous driving and advanced driver-assistance systems (ADAS).
This process involves identifying and locating objects in the environment, such as vehicles,
pedestrians, and traffic signs, which is essential for understanding the surroundings and segmenting them. Camera-based systems are the most commonly used sensors for this purpose
due to their ability to provide detailed visual information. Numerous architectures have been
developed to enhance object detection, enabling vehicles to make informed decisions. Beyond
detection, instance segmentation plays a vital role by not only detecting objects but also delineating each object’s precise boundaries. This detailed identification allows autonomous systems to differentiate between objects, even when they overlap, and to analyze their shapes, positions, and movements accurately. This capability is crucial for tasks requiring detailed environmental understanding, such as obstacle avoidance and path planning. It is also important that the segmentation takes place without any difficulty even in low-light conditions, thus ensuring a safer and more efficient driving experience. However, for this purpose it is manually impossible to label each and every data. So it is important that the model itself gets generalized to the various environmental conditions. This can be done using the domain adaptation techniques.

