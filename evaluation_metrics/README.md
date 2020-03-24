# Evaluation metrics for ML algorithms
A proper evaluation metric for a machine learning algorithm is as important as data preparation and model training. 
Because, knowing how to evaluate the trained model would help us to assess the trained model, hence choose proper model.

There is no single magic performance metric for all the tasks!


## Accuracy
- one of the well known evaluation/performance metric.

<img src="https://render.githubusercontent.com/render/math?math=Accuracy = \frac{TP %2B TN}{TP %2B TN %2B FP %2B FN} = \frac{all \enspace correct \enspace predictions}{total \enspace number \enspace of \enspace samples or predictions}"/>

- the ratio of correctly classified samples to the total number of samples.
- (# of correct predictions) / (total # of predictions)
- mostly used in classification problems.
- we may need additional evaluation metrics (precision, recall, f1-score) depending on the problem.


## TP, TN, FP, FN
- most of the evaluation metrics can be defined using these four metrics
 
- TP (True Positive, hit rate): 
    - the sample was predicted as `positive` and the prediction itself was `true`/`correct`
    - it is a success
- TN (True Negative, correct rejection)
    - the sample was predicted as `negative` and the prediction itself was `true`/`correct`
    - it is a success
- FP (False Positive, false alarm, Type-I error)
    - the sample was predicted as `positive` but the prediction itself was `false`/`not correct`
    - it is an error (aka, Type-I). Should have been predicted as `negative`
- FN (False Negative, miss, Type-II error)
    - the sample was predicted as `negative` but the prediction itself was `false`/`not correct`
    - it is an error (aka, Type-II). Should have been predicted as `positive`

## Precision
- can be seen as a measure of exactness or quality
- in classification task: the ratio of true-positive (TP) predictions over all positive predictions (TP + FP)
- precision = TP / (TP + FP)

## Recall
- can be seen as a measure of completeness or quantity 
- in classification task: the ratio of true-positive (TP) predictions over all true values (TP + FN)
- recall = TP / (TP + FN)

## Confusion Matrix (error matrix)

<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/practical-ml-implementations/blob/master/imgs/binary_confusion_matrix.png" width="400"></p>

- **Binary classification case**. Spam vs not-spam classification.
```
[[3, 0],
 [0, 7]]
TP = 7
FP = 0
TN = 3
FN = 0
accuracy = 100%
precision = 1
recall = 1
```

```
[[2, 1],
 [2, 5]]
TP = ?
FP = ?
TN = ?
FN = ?
accuracy = ?
precision = ?
recall = ?
```


- **Multi-class classification case.** Gender detection: Female, Male, Both (meaning both female and male exist in the image).
```
[[3753  164   79]
 [ 171 1495   94]
 [ 173  140  567]]

Accuracy = 0.8762808921036769

              precision    recall  f1-score   support

      Female       0.92      0.94      0.93      3996
        Male       0.83      0.85      0.84      1760
        Both       0.77      0.64      0.70       880
```


```
[[3864   90   42]
 [  87 1781   32]
 [  59   39  782]]

Accuracy = ?
              precision    recall  f1-score   support

      Female                         
        Male                         
        Both                          
```

## Classification scenarios: 
- binary classification tasks (spam vs not-spam classification). 
    - we want to evaluate the classification model performance on validation set (with 10 examples). positive class (`1`) indicates the spam whereas negative class (`0`) indicates the not-spam. 
    - scenario # 1:
       - ground_truth = `[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]`
       - predictions  = `[0, 0, 0, 0, 1, 0, 0, 0, 1, 0]`
       - accuracy = 80%
       - precision:
       - recall:
    - scenario # 2:
       - ground_truth = `[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]`
       - predictions  = `[0, 0, 1, 0, 1, 0, 0, 0, 1, 0]`
       - accuracy:
       - precision:
       - recall:
    - scenario # 3:
       - ground_truth = `[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]`
       - predictions  = `[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]`
       - accuracy:
       - precision:
       - recall:
    - scenario # 4:
       - ground_truth = `[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]`
       - predictions   = `[1, 1, 1, 0, 1, 0, 0, 0, 1, 0]`
       - accuracy:
       - precision:
       - recall:
    - scenario # 5:
       - ground_truth = `[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]`
       - predictions   = `[0, 0, 0, 1, 0, 0, 0, 0, 1, 1]`
       - accuracy: 
       - precision:
       - recall:
    - scenario # 6:
       - ground_truth = `[0, 0, 0, 1, 0, 0, 0, 0, 1, 0]`
       - predictions  = `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`
       - accuracy:
       - precision:
       - recall:

