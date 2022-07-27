import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

def calc_threshold_at_95(y_true, y_pred, tpr_thresh=0.95, verbose=False):
    """Calculate the TNR at TPR of 0.95"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    if verbose:
        print(f"True positive rate {tpr[np.argmax(tpr >= tpr_thresh)]}")

    threshold = np.argmax(tpr >= tpr_thresh)
    if round(tpr[threshold], 2) != tpr_thresh:
        print(f"Lowest TPR >= 95 % was {round(tpr[np.argmax(tpr >= tpr_thresh)], 2)}")
    return thresholds[threshold]

def sensitivity(true, pred):
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    return tp / (tp + fn)


def specificity(true, pred):
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    return tn / (tn + fp)

def calculate_aucs(y_true, y_pred, labels):
    aucs = [roc_auc_score(y_true[:,i], y_pred[:,i])
            for i in range(y_true.shape[1])]
    mean = round(sum(aucs) / len(aucs), 3)
    result = {label: f"{round(auc, 3)}"
            for label, auc in zip(labels, aucs)}
    result["Mean"] = mean
    return result
