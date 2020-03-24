# Object Detection
- Object detection plays a crucial role in Computer Vision. Because detecting a particular object is the starting point of many other tasks, for instance;
    - object tracking
    - object recognition/matching (face detection --> face recognition/matching)
    - CBIR - content based image retrieval
    - activity recognition
    - person/human re-identificaton

## Brief overview
<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/practical-ml-implementations/blob/master/imgs/classification_vs_object_detection.png" width="400"></p>

- Object detection do two things either sequentially or simultaneously:
    - localization: where is the object?
    - classification: what is this object?
    
- Some object detection algorithms:
    - Two stage (region proposal + classification) algorithms: [R-CNN](https://arxiv.org/abs/1311.2524), [Fast R-CNN](https://arxiv.org/abs/1504.08083), [Faster R-CNN](https://arxiv.org/abs/1506.01497)
    - Single stage algorithms:
        - [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
        - [YOLO: You Only Look Once](https://arxiv.org/abs/1506.02640)
        - [RefineDet: Single-Shot Refinement Neural Network for Object Detection](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single-Shot_Refinement_Neural_CVPR_2018_paper.pdf)
        - [RetinaNet: Focal Loss for Dense Object Detection](http://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
        - [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070): 


## Evaluation metrics
- Great [repository](https://github.com/rafaelpadilla/Object-Detection-Metrics) for object detection metrics by [Rafael Padilla](https://www.linkedin.com/in/rafael-padilla/)

Building blocks for object detection evaluation metrics:
- **IoU**: Intersection over Union. IoU measures how well two bounding boxes overlapped. IoU threshold defines if the detected bounding box detection is valid (`True Positive`) or not (`False Positive`). Usually, in most object detection challenges, by default `IoU threshold=0.5`. But this may change by custom projects. 

<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/practical-ml-implementations/blob/master/imgs/iou.png" width="400"></p>


<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/practical-ml-implementations/blob/master/imgs/iou_examples.png" width="400"></p>


- **Confidence threshold**: is the threshold to define the acceptance score of classification score for a particular object. As you may already realized, there are two thresholds, one is for accepting the localization as a valid or not (`IoU threshold`) and the second one is for accepting the classification as true positive or falso positive (`confidence threshold`).


- **Precision**: specifies how precise the detection algorithm has detected the relevant objects. In other words, what ratio of detections are, indeed, correct.   

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?Precision%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FP%7D%3D%5Cfrac%7BTP%7D%7B%5Ctext%7Ball%20detections%7D%7D">
</p>

- **Recall**: specifies the hit rate of the detection algorithm. In other words, what ratio of ground truth objects could be catched by the detection algorithm.   

<img src="http://latex.codecogs.com/gif.latex?Recall%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FN%7D%3D%5Cfrac%7BTP%7D%7B%5Ctext%7Ball%20ground%20truths%7D%7D">
</p>

Object detection evaluation metrics:
- **Precision Recall Curve**: is a great assessment tool to measure the performance of object detection model or algorithm. 
The Precision x Recall curve is a good way to evaluate the performance of an object detector as the confidence is changed by plotting a curve for each object class. An object detector of a particular class is considered good if its precision stays high as recall increases, which means that if you vary the confidence threshold, the precision and recall will still be high. Another way to identify a good object detector is to look for a detector that can identify only relevant objects (0 False Positives = high precision), finding all ground truth objects (0 False Negatives = high recall).

A poor object detector needs to increase the number of detected objects (increasing False Positives = lower precision) in order to retrieve all ground truth objects (high recall). That's why the Precision x Recall curve usually starts with high precision values, decreasing as recall increases. You can see an example of the Prevision x Recall curve in the next topic (Average Precision). This kind of curve is used by the PASCAL VOC 2012 challenge and is available in our implementation.

- [sklearn.metrics.precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
<p align="center"> 
    


## Benchmark datasets/comptetitions:
- [PASCAL VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)
- [COCO Detection Challenge](https://competitions.codalab.org/competitions/5181)
- [Google Open Images Dataset V4 Competition](https://storage.googleapis.com/openimages/web/challenge.html)
- [ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-detection-challenge)

## Practical hints:
- you can play with IoU threshold according to your needs. For instance, if the localization is more important for a particular project, then it may be better to increase the IoU threshold (`IoU threshold = 0.8`), Likewise, if the localization is not that much crucial for the implementation, then it is possible to decrease the thresholds (`IoU threshold = 0.3`).

# Object Detection implementations:
## Face detection from scratch


