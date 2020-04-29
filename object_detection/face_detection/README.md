# Face Detection implementation from scratch
This is a face detection model training pipeline with no annotated data. The purpose of this implementation is to show how we can leverage the open source libraries to get a real product (face detector) by only having appropriate (can be used for any purpose) images. 


Pipeline:
- Prepare your data: put the folder of face images under the `./data` folder. No need to have the annotations.
- Prepare training data: images and corresponding annotations. Since we assume that we don't have an annotated data we are going to get the annotations using OpenCV's face detector.
    - Run OpenCV's face detector over all images, and store the detection results.
- Train MTCNN networks:
    - PNet (proposal network)
    - RNet (refine network)
    - ONet (output network)


## OpenCV's face detector:
- download [trained classifier XML file](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and put it under `./models` (md5sum: `a03f92a797e309e76e6a034ab9e02616`)
- 



