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
- Great [repository](https://github.com/rafaelpadilla/Object-Detection-Metrics) for object detection metrics
- IoU: Intersection over Union. IoU measures how well two bounding boxes overlapped. IoU threshold defines if the detected bounding box detection is valid (`True Positive`) or not (`False Positive`). Usually, in most object detection challenges, by default `IoU threshold=0.5`. But this may change by custom projects. 

## Benchmark datasets/comptetitions:
- [PASCAL VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)
- [COCO Detection Challenge](https://competitions.codalab.org/competitions/5181)
- [Google Open Images Dataset V4 Competition](https://storage.googleapis.com/openimages/web/challenge.html)
- [ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-detection-challenge)

## Practical hints:
- you can play with IoU threshold according to your needs. For instance, if the localization is more important for a particular project, then it may be better to increase the IoU threshold (`IoU threshold = 0.8`), Likewise, if the localization is not that much crucial for the implementation, then it is possible to decrease the thresholds (`IoU threshold = 0.3`).

# Object Detection implementations:
## Face detection from scratch


