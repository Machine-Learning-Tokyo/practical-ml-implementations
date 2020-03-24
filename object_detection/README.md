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

<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/practical-ml-implementations/blob/master/imgs/iou.png" width="400"> <em>[https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/] </em> </p>


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
The precision recall curve is the curve plotting the precision and recall values by changing the confidence scores (remember that precision and recall values change by confidence score). 
    - the performance of detector (for particular class) is considered good if its precision does not descrease much as recall increases. This means, independent of a confidence threshold (whether it is `0.9`, `0.7` or `0.3`) the precision and recall will stay at high values. It is worth to mention that this is not the case in a real object detection tasks... 
    - the another indicator of good detector is the number of False Positives (smaller is better, i.e. no false alarm), which defines the **Precision**. While performing good in terms of the precision, if a detector can find all ground truth objects, which defines the **Recall**, the detector can be considered as "good".
    - The term **"good"** is a subjective metric. There is a trade-off between high precision and high recall. If our implementation has to perform better in terms of precision, then possibly we have to keep the **confidence score** higher (to keep the precision at higher values as much as possible). If our implementation has to perform better in terms of the recall, then it is ok to decrease the **confidence score** lower (to increase the recall value). How can be sure which **confidence score** we have to choose given the precision/recall requirements. The **Precision Recall Curve** helps us to decide on **confidence score**.  

Here is some sample precision-recall-curves and we will which confidence score would be more meaningful after analysing the curve:




A poor object detector needs to increase the number of detected objects (increasing False Positives = lower precision) in order to retrieve all ground truth objects (high recall). That's why the Precision x Recall curve usually starts with high precision values, decreasing as recall increases. You can see an example of the Prevision x Recall curve in the next topic (Average Precision). This kind of curve is used by the PASCAL VOC 2012 challenge and is available in our implementation.

- [sklearn.metrics.precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
<p align="center"> 
    
- **Average Precision (AP)**: a numerical metric, can also help us compare different detectors. In practice AP is the precision averaged across all recall values between 0 and 1. In other words, averaged precision values across all **confidence score**

- **mean Average Precision (mAP)**: mean of AP for all classes in our dataset. 

## Benchmark datasets/comptetitions:
- [PASCAL VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)
- [COCO Detection Challenge](https://competitions.codalab.org/competitions/5181)
- [Google Open Images Dataset V4 Competition](https://storage.googleapis.com/openimages/web/challenge.html)
- [ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-detection-challenge)

## Practical hints:
- you can play with IoU threshold according to your needs. For instance, if the localization is more important for a particular project, then it may be better to increase the IoU threshold (`IoU threshold = 0.8`), Likewise, if the localization is not that much crucial for the implementation, then it is possible to decrease the thresholds (`IoU threshold = 0.3`).
- if your team leader/project manager/client asks you to implement an object detection algorithm, please be sure to ask them:
    - whether the **precision** or **recall** is more important for them. Let me guess, you would get the answer: "Oh, both of them are important for us!" almost in 90% cases :laughing:. If you get this answer, then go ahead with your own trading offs, i.e. choose the appropriate **confidence score** according to the **precision recall curve**.   
    - how precise should be the detected bounding boxes. Again, almost everytime you would get the answer "Oh, it is always better to fit 100%" :laughing:. If you are lucky, your client/PM/team leader knows what he/she needs, it would be much easier for you to define the **IoU threshold**. Otherwise, ho ahead with the default value: `IoU thresohld=0.5`.
    
# Object Detection implementations:
## Face detection from scratch


