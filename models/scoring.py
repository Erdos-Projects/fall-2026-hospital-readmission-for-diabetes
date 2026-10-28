from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score, 
                             f1_score, fbeta_score, confusion_matrix, log_loss, 
                             classification_report, average_precision_score, precision_recall_curve)



def score(yv, y_pred=None, proba=None, threshold=None):
    """
    yv is the validation set's target values
    y_pred are the predicted values
    proba are the predicted probabilities
    threshold is user inputed threshold
    """
    # if y_pred is not provided but thresholds and the probabilities are, define predictions
    if threshold is not None and proba is not None and y_pred is None:
        y_pred = (proba >= threshold).astype(int)

    # obtain various metrics
    if y_pred is not None:
        print(f"Accuracy: {accuracy_score(yv, y_pred):.3f}")
        print("Confusion matrix:\n", confusion_matrix(yv, y_pred))
        print(classification_report(yv, y_pred))
        print(f"F1 @{threshold}:", round(f1_score(yv, y_pred), 3))
        f2 = fbeta_score(yv, y_pred, beta=2)
        print(f"F2 @{threshold}:", round(f2, 3))

    # if threshold and probabilities are provided
    if threshold is not None and proba is not None:
        y_pred = (proba >= threshold).astype(int)
        print("ROC-AUC:", round(roc_auc_score(yv, proba), 4))
        print("PR-AUC: ", round(average_precision_score(yv, proba), 4))
        p, r, thr = precision_recall_curve(yv, proba)
        f1s = 2 * p * r / (p + r + 1e-12)
        best = f1s[:-1].argmax()
        print(f"\nBest-threshold F1: {f1s[best]:.5f} at threshold {thr[best]:.5f}.")

        f2s = 5 * p * r / (4 * p + r + 1e-12)
        best2 = f2s[:-1].argmax()
        print(f"\nBest-threshold F2: {f2s[best2]:.5f} at threshold {thr[best2]:.5f}.")
        

    
    

    

    