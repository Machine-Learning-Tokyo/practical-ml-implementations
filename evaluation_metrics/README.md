# Evaluation metrics for ML algorithms
A proper evaluation metric for a machine learning algorithm is as important as data preparation and model training. 
Because, knowing how to evaluate the trained model would help us to assess the trained model, hence choose proper model.

 
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
 
## Accuracy
- <img src="https://render.githubusercontent.com/render/math?math=Accuracy = \frac{TP %2B TN}{TP %2B TN %2B FP %2B FN} = \frac{all \enspace correct \enspace predictions}{total \enspace number \enspace of \enspace samples or predictions}"/>
- the ratio of correctly classified samples to the total number of samples.
- (# of correct predictions) / (total # of predictions)
- mostly used in classification problems. However, one may need additional evaluation metrics (precision, recall, f1-score) depending on the problem.


## Precision and Recall
## mAP
## Confusion Matrix
## f1-score



