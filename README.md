# ImageML : Classification Detection Segmentation

## Classification

- Finding the accuracy on MNIST dataset using `CNN and Autoencoder` (DL).
     - `CNN` : I used the LeNet architecture as a CNN model 
     - `ViT` : In the comparision, I replaced the CNN model with the vision transformer.
     - The convergence rates for CNN are higher than that of Transformer but over a large run it is generally observed that transformer performs better that CNN model.This is generally because CNN well captures the local features but it also has inductive biases, which helps it to fit on the data and not generalize well.
     
- Finding the accuracy on MNIST dataset using `SIFT+KMeans+BoVW+SVMs` (ML).
     - `SIFT` (Scale invariant feature transform) was used to extract the descriptors from the image.
     - `KMeans` clustering was used to cluster all the extracted descriptors from all the images into cluster groups. Note that each image can have more tha 1 or even larger number of descriptors.
     - `BoVW` (Bag of Visual words) can generate the histogram for each image where the visual word constituting the histogram will be the cluster center.
     - `SVM` (Support Vector Machine) will be finally used to fit to the train data and predict on the test data.

## Detection 

### Face Detection and Association-based Tracking

#### 1. Data Preparation
- Downloaded video clip from [Forrest Gump scene](https://www.youtube.com/watch?v=bSMxl1V8FSg) and converted the first 30 seconds into ~720 frames.

#### 2. Face Detection (1.5 points)
- Utilized OpenCV’s Viola-Jones Haar cascades-based face detector.
- Average time taken to process each frame was noted.
- Key factors affecting processing time include:
  - Size of the image.
  - Complexity of the face detector (e.g., number of stages in cascade).
  - Scale factor and minimum neighbors set in the Haar classifier.

#### 3. Face Detection Visualization
- Visualized face detection by drawing bounding boxes on detected faces in each frame.
- **Conclusions:**
  - Works well in well-lit, frontal face scenarios.
  - Fails when faces are partially occluded or in profile view.
  - Struggles with detecting small, distant faces due to scaling issues.

#### 4. Association-based Tracking
- Implemented IoU-based tracking across consecutive frames (IoU > 0.5).
- New tracks created for unseen faces, and tracks ended when faces were no longer detected.
- Visualized the tracking by assigning and displaying unique track identifiers. Results and challenges in the jupyter notebook.

---

### YOLO Object Detection

#### 1. Data Preparation
- Downloaded Open Images Dataset v7 (ducks) from [Kaggle](https://www.kaggle.com/datasets/haziqasajid5122/yolov8-finetuning-dataset-ducks).

#### 2. Understanding YOLO Object Detector 
YOLO is a single-shot detector, processing images in one go, unlike R-CNN which generates region proposals first.


#### 3. Hands-on with Ultralytics
- Created a `yolov8n` (nano) model
- Compared with `yolov8m` (medium)

#### 4. Training YOLO Variants
- Created two training datasets (`train1` with 100 images, `train2` with 400 images).
- Trained 3 variants of YOLOv8 models on both datasets:
  - `yolov8n` from scratch.
  - `yolov8n` with pretrained weights.
  - `yolov8m` with pretrained weights.
- Reported AP50 scores across models and datasets.

## Segmentation
- Used `UNet` architecture for the segmentation of the image.
- Analyzed the effects of having skip-connections.
- Trained `UNet-with Skip` and `UNet-without Skip` and tested on sample images to visually check differences.
- Found that without skip-connections we get a `smoother boundary` (curvature) between two segmented regions as compared to without Skip. Also, in terms of IoU, the Skip connections perform better than the model not having them.

## Further exploring the SOTA models in the above fields of Computer Vision :) 