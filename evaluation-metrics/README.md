# Evaluation metrics for ML algorithms
A proper evaluation metric for a machine learning algorithm is as important as data preparation and model training. 
Because, knowing how to evaluate the trained model would help us to know our the model will perform as expected in production.

 
## TP, TN, FP, FN
- most of the rest metrics use these as a building block
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
 
## Accuracy
- the ratio of correctly classified samples to the total number of samples.
- (# of correct predictions) / (total # of predictions)
- 
## Precision and Recall
## mAP
## Confusion Matrix
## f1-score



