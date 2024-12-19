# Human Pose estimation Using CNN and Keypoint R-CNN

## Introduction
Human pose estimation is a crucial task in computer vision, involving the prediction of the spatial arrangement of a person's body from images or videos. The accurate estimation of human poses has numerous applications, including activity recognition, human-computer interaction, and augmented reality. However, pose estimation is challenging due to variations in human body shapes, poses, and environmental conditions.

In this project, we aim to develop a human pose estimation model using deep learning techniques. Our goal is to accurately predict keypoint coordinates corresponding to anatomical landmarks on the human body, such as joints and limbs. By addressing this problem, we aim to contribute to advancements in computer vision and enable applications that require precise understanding of human movements and interactions

## Methodology - CNN
* <strong>Data Preprocessing: </strong>
  * Image Resizing and Normalization: All images in the MPII Human Pose dataset are resized to 220x220 pixels to ensure consistent input dimensions for the model. The pixel values are normalized to a range between 0 and 1 to facilitate convergence during training.
  * Data Augmentation: Data augmentation techniques such as random rotations, 
translations, flips, and adjustments in brightness or contrast are applied to 
increase the variability of the training data. This helps in improving model 
generalization and robustness.

* <strong>Model Architecture Design: </strong>
  * We design a deep convolutional neural network (CNN) architecture tailored for human pose estimation.
  * The model architecture consists of multiple convolutional layers followed by max-pooling layers to learn hierarchical features from input images.
  * Skip connections or residual connections are incorporated to facilitate information flow across different layers and capture both local and global spatial relationships in the input images.

* <strong>Training Procedure: </strong>
  * Dataset Splitting: The preprocessed dataset is split into training and validation sets to monitor model performance and prevent overfitting.
  * Optimizer: The Adam optimizer is selected for its adaptive learning rate capabilities, which help in faster and more stable convergence.
  * Loss Function: Mean Squared Error (MSE) is used as the loss function to minimize the difference between predicted and ground truth keypoints.
  * Learning Rate Scheduling: The learning rate is adjusted during training. For example, it can be reduced after a certain number of epochs or based on validation performance to prevent overshooting the optimal solution.
  * Training Iterations: Batches of training data are iteratively fed to the model. The model's weights are updated based on gradients computed from backpropagation.
  * Regularization Techniques: Early stopping and learning rate scheduling are employed to prevent overfitting and ensure stable convergence during training.

## Results - CNN
* <strong>Performance Metrics: </strong>
  * Learning Curves: Plots of training and validation loss over epochs show the model's learning progress and help in diagnosing potential overfitting or underfitting.
  * Accuracy Improvements: Regularization techniques and learning rate adjustments lead to significant improvements in model accuracy over the training period.

* <strong>Training and Validation Performance: </strong>
  * Training Progress: The learning rate adjustments and regularization techniques applied during training helped in achieving stable convergence and avoiding overfitting.
  * Validation Results: The model demonstrated high accuracy in predicting keypoint coordinates on the validation set, as indicated by low MAE and effective visualizations of predicted poses.

![lmao](https://github.com/Harish-Balaji-B/Human-Pose-estimation-Using-CNN-and-Keypoint-R-CNN/blob/main/Results/cnn.png)<br>


## Methodology - Key Point R-CNN
* <strong>Data Preprocessing: </strong>
  * Image Reading: Input images are read using OpenCV.
  * Transformation: Images are transformed to tensors using the torchvision.transforms module, which normalizes and prepares the images for the model.

* <strong>Model: </strong>
  * The model used is the keypointrcnn_resnet50_fpn, a pre-trained model designed for object detection and keypoint detection. This model leverages a ResNet-50 backbone with a Feature Pyramid Network (FPN) for robust feature extraction at multiple scales.
  
* <strong>Image Processing: </strong>
  * Images are read and transformed into a format suitable for the model.
  * The model processes the image, outputting the detected keypoints and their 
corresponding scores.

* <strong>Keypoint and Skeleton Drawing: </strong>
  * Keypoints are drawn on the images using OpenCV, with confidence scores dictating their visibility.
  * Skeletons are drawn by connecting relevant keypoints based on predefined limb connections.
 
## Results - Key Point R-CNN
![lmao](https://github.com/Harish-Balaji-B/Human-Pose-estimation-Using-CNN-and-Keypoint-R-CNN/blob/main/Results/rcnn.png)<br>

## Extra Work
An web application using flask has also been implemented using key point RCNN where we can upload an image or do live time pose estimation.

<strong>Results: </strong>
![lmao](https://github.com/Harish-Balaji-B/Human-Pose-estimation-Using-CNN-and-Keypoint-R-CNN/blob/main/Results/layout.png)<br>

![lmao](https://github.com/Harish-Balaji-B/Human-Pose-estimation-Using-CNN-and-Keypoint-R-CNN/blob/main/Results/result.png)<br>

![lmao](https://github.com/Harish-Balaji-B/Human-Pose-estimation-Using-CNN-and-Keypoint-R-CNN/blob/main/Results/live.png)<br>

## Future Scope
* Fine-tuning for Specific Applications: Fine-tuning the pre-trained R-CNN model on specific datasets or for specific applications (e.g., sports, medical analysis) could enhance its performance in those areas.
* Semi-supervised Learning: Utilizing semi-supervised or unsupervised learning techniques could help in leveraging large amounts of unlabeled data to further improve the model's performance.
* Enhanced Data Augmentation: Implementing more sophisticated data augmentation techniques could help improve the model's robustness to variations in lighting, occlusions, and different poses.

## Note
Download the saved-model_MPIIy1.keras from the link below: <br>
https://drive.google.com/file/d/1t9OG8_kjbfmAIF06BTk28A4ygDafKFHP/view?usp=drive_link

`Create a folder "uploads" inside client_server to store the images for the flask application` <br>
Also: <br>
`Create a folder "testresults" inside CNN folder to store the result images for the CNN method`
