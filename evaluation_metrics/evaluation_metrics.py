import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, precision_recall_curve
import matplotlib.pyplot as plt


class NumpyEvaluationMetrics:
    def __init__(self):
        pass

    def convert_to_numpy_array(self, y_true, y_pred):
        """ convert list arrays to numpy arrays """
        return np.array(y_true), np.array(y_pred)

    def class_specific_categorical_to_binary(self, y_true, y_pred, label):
        y_true_ = [int(i) for i in y_true == label]
        y_pred_ = [int(i) for i in y_pred == label]
        return y_true_, y_pred_

    def tp_fp_tn_fn(self, y_true, y_pred):
        """
        True for only binary labels:
            TP (True Positive):  we predict a label of 1 (positive), and the true label is 1.
            FP (False Positive): we predict a label of 1 (positive), but the true label is 0.
            TN (True Negative):  we predict a label of 0 (negative), and the true label is 0.
            FN (False Negative): we predict a label of 0 (negative), but the true label is 1.
        """
        y_true, y_pred = self.convert_to_numpy_array(y_true, y_pred)

        TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
        FP = np.sum(np.logical_and(y_true == 0, y_pred == 1))
        TN = np.sum(np.logical_and(y_true == 0, y_pred == 0))
        FN = np.sum(np.logical_and(y_true == 1, y_pred == 0))

        return TP, FP, TN, FN

    def accuracy(self, y_true, y_pred):
        """
        measured in percentage:
            accuracy_for_class_0 = (TP + TN) / (P + N)
            OR
            accuracy_for_class_0 = (TP + TN) / (TP + FP + TN + FN)
            OR
            percentage of correctly predicted classes
        """
        if len(y_true) != len(y_pred):
            print("y_true and y_pred should have the same length, returning None")
            return None
        number_of_instances = len(y_true)
        y_true, y_pred = self.convert_to_numpy_array(y_true, y_pred)
        acc = np.sum(y_true == y_pred) / number_of_instances
        return acc

    def confusion_matrix(self, y_true, y_pred, labels=None):
        y_true, y_pred = self.convert_to_numpy_array(y_true, y_pred)
        if labels is None:
            number_of_classes = len(list(np.unique(y_true)))
        else:
            number_of_classes = len(labels)

        cm = np.zeros((number_of_classes, number_of_classes))
        for a, p in zip(y_true, y_pred):
            cm[a][p] += 1

        return cm

    def precision(self, y_true, y_pred, labels=None):
        """
        for each class:
            precision = tp / (tp + fp)
            OR
            precision = tp / (total number of positives in y_pred)
        """
        if labels is None:
            labels = list(np.unique(y_true))
        y_true, y_pred = self.convert_to_numpy_array(y_true, y_pred)
        precision_scores = np.zeros((len(labels), ))
        for ind, label in enumerate(labels):
            y_true_, y_pred_ = self.class_specific_categorical_to_binary(y_true, y_pred, label)
            tp, fp, tn, fn = self.tp_fp_tn_fn(y_true_, y_pred_)
            precision_scores[ind] = tp / (tp + fp)
        return precision_scores

    def recall(self, y_true, y_pred, labels=None):
        """
        for each class:
            recall = tp / (tp + fn)
            OR
            recall = tp / (total number of positives in y_true)
        """
        if labels is None:
            labels = list(np.unique(y_true))
        y_true, y_pred = self.convert_to_numpy_array(y_true, y_pred)
        recall_scores = np.zeros((len(labels), ))
        for ind, label in enumerate(labels):
            y_true_, y_pred_ = self.class_specific_categorical_to_binary(y_true, y_pred, label)
            tp, fp, tn, fn = self.tp_fp_tn_fn(y_true_, y_pred_)
            recall_scores[ind] = tp / (tp + fn)
        return recall_scores

    def iou(self, bbox1, bbox2):
        """
            Source: PyImageSearch
            reference: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        """
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        boxBArea = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou


class SklearnMetrics(NumpyEvaluationMetrics):
    def __init__(self):
        super(SklearnMetrics, self).__init__()

    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true=y_true, y_pred=y_pred)

    def confusion_matrix(self, y_true, y_pred, labels=None):
        return confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    def precision(self, y_true, y_pred, labels=None):
        if labels is None:
            labels = list(np.unique(y_true))
        return precision_score(y_true=y_true, y_pred=y_pred, labels=labels, average=None)

    def recall(self, y_true, y_pred, labels=None):
        return recall_score(y_true=y_true, y_pred=y_pred, labels=labels, average=None)

    def precision_recall_curve(self, y_true, probas_pred):
        return precision_recall_curve(y_true, probas_pred)

    def plot_precision_recall_vs_thresholds(self, precisions, recalls, thresholds, title='', output_file=None):
        plt.gcf().clear()

        balance_point = np.argmin(np.abs(precisions - recalls))
        balance_y_axis = precisions[balance_point]
        balance_x_axis = thresholds[balance_point]

        plt.plot(thresholds, precisions[:-1], 'g--', label='precision')
        plt.plot(thresholds, recalls[:-1], 'b--', label='recall')
        fig = plt.gcf()
        ax = fig.gca()

        balance_circle = plt.Circle((balance_x_axis, balance_y_axis), 0.01, color='r', fill=True, label='balance point')
        ax.add_artist(balance_circle)

        major_ticks = np.arange(0, 1, 0.1)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)

        plt.xlabel('Threshold')
        plt.ylabel('Precision / Recall')
        plt.legend(loc='lower center')
        plt.ylim([0, 1])
        plt.xlim([0, 1])

        title = "Balance point threshold: {} \n ".format(np.round(balance_x_axis, 2)) + title

        plt.title(title)
        plt.grid(True)

        if output_file is not None:
            plt.savefig(output_file, dpi=100)
            plt.gcf().clear()
        else:
            return plt


y_pred = [1, 1, 1, 0, 1, 0, 0]
y_true = [1, 0, 1, 0, 0, 1, 0]

# y_true = [0, 2, 1, 1, 0, 1, 2]
# y_pred = [1, 2, 2, 1, 0, 1, 2]


numpy_em = NumpyEvaluationMetrics()
sklearn_em = SklearnMetrics()

print(numpy_em.tp_fp_tn_fn(y_true=y_true, y_pred=y_pred))
print(sklearn_em.tp_fp_tn_fn(y_true=y_true, y_pred=y_pred))
print(numpy_em.recall(y_true=y_true, y_pred=y_pred))
print(sklearn_em.recall(y_true=y_true, y_pred=y_pred))
