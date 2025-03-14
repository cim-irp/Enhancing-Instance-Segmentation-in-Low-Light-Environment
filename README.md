# Enhancing-Instance-Segmentation-in-Low-Light-Environment

This repository is the implementation of Enhancing Instance Segmentation in Low-light Environment. Our code is based on detectron2.

**About the work**
Object detection and segmentation are the critical component in the automotive industry, especially for applications like autonomous driving and advanced driver-assistance systems (ADAS).This process involves identifying and locating objects in the environment, such as vehicles,pedestrians, and traffic signs, which is essential for understanding the surroundings and segmenting them. Camera-based systems are the most commonly used sensors for this purpose
due to their ability to provide detailed visual information. Numerous architectures have been
developed to enhance object detection, enabling vehicles to make informed decisions. Beyond
detection, instance segmentation plays a vital role by not only detecting objects but also delineating each object’s precise boundaries. This detailed identification allows autonomous systems to differentiate between objects, even when they overlap, and to analyze their shapes, positions, and movements accurately. This capability is crucial for tasks requiring detailed environmental understanding, such as obstacle avoidance and path planning. It is also important that the segmentation takes place without any difficulty even in low-light conditions, thus ensuring a safer and more efficient driving experience. However, for this purpose it is manually impossible to label each and every data. So it is important that the model itself gets generalized to the various environmental conditions. This can be done using the domain adaptation techniques.

**Features**

- An enhanced model for the instance segmentation in real-time environmental conditions.
- Mainly focused on the low-light environment.
- Generalized for real-time application using the domain adaptation techniques.

**Methodology**

The type of Domain Adaptation used is semi-supervised. The techniques used are Adversarial training and Image-to-image translation.
The fine-tuning techniques adapted are:
- Data Augmentation
- Pixel Normalization
- Backbone Tuning
- Increasing the Learning Rate
- Reducing the Weight Decay

**Results**

The inference of the model in ideal case is shown below.
![Screenshot 2025-03-12 233213](https://github.com/user-attachments/assets/6e79bca5-7d70-4f44-b538-27359d43cfe1)

The segmentation of the various objects in the ideal case is accurate and the further evaluation on the low-light condition is conducted.

The inference of the model at different stages of fine-tuning is as shown below.
![Screenshot 2025-03-14 150432](https://github.com/user-attachments/assets/8966e1e0-4459-401c-93ba-25b1cfaeebac)

The model shows greater improvement after fine-tuning. The number of false positives is significantly reduced and the overall detection ccuracy and robustnedd of the model is improved.
