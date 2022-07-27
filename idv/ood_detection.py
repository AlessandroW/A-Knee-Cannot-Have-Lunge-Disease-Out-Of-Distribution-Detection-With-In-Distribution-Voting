from pprint import pprint
from pathlib import Path

import torch
import numpy as np

from idv import metrics


def filter_no_finding(predictions):
    """Remove ID samples with the label no finding."""
    mask = predictions["labels"].sum(axis=1).bool()
    predictions["predictions"] = predictions["predictions"][mask]
    predictions["labels"] = predictions["labels"][mask]
    return predictions


def report_sensitivity_and_specificity(y_true, y_pred, title):
    sensitivity = metrics.sensitivity(y_true, y_pred)
    specificity = metrics.specificity(y_true, y_pred)
    print(f"\n{title}")
    print(f"ID classification sensitivity {round(sensitivity, 3)}")
    print(f"ID classification specificity {round(specificity, 3)}")


def report_end_to_end(id_test, ood_test, ood_pred, labels):
    end_to_end_pred = np.concatenate([id_test["predictions"], ood_test["predictions"]])
    end_to_end_true = np.concatenate([id_test["labels"], ood_test["labels"]])
    end_to_end_pred[ood_pred <= 0.5] = np.zeros(end_to_end_pred.shape[1])
    print("\nEnd-to-End AUC")
    pprint(metrics.calculate_aucs(end_to_end_true, end_to_end_pred, labels=labels))


def idv(cxr14_val, cxr14_test, ood_val, ood_test, labels):
    def _idv(id, ood, id_threshold=None):
        y_ood_pred = np.concatenate([id["predictions"],
                                     ood["predictions"]])
        if id_threshold is not None:
            ood_labels = np.zeros(len(ood["predictions"]))
            id_labels = np.ones(len(id["predictions"]))
            y_ood_true = np.concatenate([id_labels, ood_labels])
            y_ood_pred = 1 * np.any((y_ood_pred > id_threshold), axis=1)
            return y_ood_true, y_ood_pred
        else:
            ood_labels = np.zeros([ood["predictions"].shape[0],
                                   id["predictions"].shape[1]])
            y_ood_true = np.concatenate([id["labels"], ood_labels])
            return [metrics.calc_threshold_at_95(y_ood_true[:, i], y_ood_pred[:, i])
                    for i in range(id["labels"].shape[1])]

    return _output_based(
        cxr14_val,
        cxr14_test,
        ood_val,
        ood_test,
        labels,
        ood_detection_method=_idv,
    )


def no_ood(cxr14_val, cxr14_test, ood_test, labels):
    """Chexnet baseline without OOD detection."""
    cxr14_val = torch.load(cxr14_val)
    cxr14_test = torch.load(cxr14_test)

    y_val_true = cxr14_val["labels"]
    y_val_pred = cxr14_val["predictions"]

    # Calculates class thresholds on the ID validation set.
    class_thresholds = [
        metrics.calc_threshold_at_95(y_val_true[:, i], y_val_pred[:, i])
        for i in range(y_val_true.shape[1])
    ]

    # Calculate OOD Performance.
    ood_test = torch.load(ood_test)
    ood_test_labels = np.zeros(len(ood_test["predictions"]))
    id_test_labels = np.ones(len(cxr14_test["predictions"]))
    y_ood_true = np.concatenate([id_test_labels, ood_test_labels])
    y_ood_pred = np.concatenate([cxr14_test["predictions"], ood_test["predictions"]])
    y_ood_pred = 1 * np.any((y_ood_pred > class_thresholds), axis=1)
    report_sensitivity_and_specificity(y_ood_true, y_ood_pred, "With No Finding")

    # Calculate OOD detection performance without No Finding labels.
    cxr14_test_wo_no_finding = filter_no_finding(cxr14_test.copy())
    y_ood_true_wo_no_finding = np.concatenate(
        [
            np.ones(len(cxr14_test_wo_no_finding["predictions"])),
            np.zeros(len(ood_test["predictions"])),
        ]
    )
    y_ood_pred_wo_no_finding = np.concatenate(
        [cxr14_test_wo_no_finding["predictions"], ood_test["predictions"]]
    )
    y_ood_pred_wo_no_finding = 1 * np.any(
        (y_ood_pred_wo_no_finding > class_thresholds), axis=1
    )

    report_sensitivity_and_specificity(
        y_ood_true_wo_no_finding, y_ood_pred_wo_no_finding, "Without No Finding"
    )

    # Calculate end-to-end performance.
    report_end_to_end(cxr14_test, ood_test, y_ood_pred, labels)


def _output_based(
    cxr14_val, cxr14_test, ood_val, ood_test, labels, ood_detection_method
):
    """General OOD Detection for all related output based methods.

    The only difference is how the ID threshold is determined.
    - Max Prediction : Maximum prediction across all classes
    - Max Softmax : Maximum Softmax
    - Max Logit : Maximum Logit across all classes
    - Max Energy : Maximum Energy_Function(Logit) across all classes
    """
    # Calculates class thresholds on the ID and OOD validation set.
    cxr14_val = torch.load(cxr14_val)
    ood_val = torch.load(ood_test)
    id_threshold = ood_detection_method(cxr14_val, ood_val)

    # Calculate OOD Performance.
    cxr14_test = torch.load(cxr14_test)
    ood_test = torch.load(ood_test)
    y_ood_test_true, y_ood_test_pred = ood_detection_method(
        cxr14_test, ood_test, id_threshold
    )
    report_sensitivity_and_specificity(
        y_ood_test_true, y_ood_test_pred, "With No Finding"
    )

    # Calculate OOD detection performance without No Finding labels.
    cxr14_test_wo_no_finding = filter_no_finding(cxr14_test.copy())
    y_ood_test_true_wo_no_finding, y_ood_test_pred_wo_no_finding = ood_detection_method(
        cxr14_test_wo_no_finding, ood_test, id_threshold
    )
    report_sensitivity_and_specificity(
        y_ood_test_true_wo_no_finding,
        y_ood_test_pred_wo_no_finding,
        "Without No Finding",
    )
    # Calculate end-to-end performance.
    report_end_to_end(cxr14_test, ood_test, y_ood_test_pred, labels)


def max_prediction(cxr14_val, cxr14_test, ood_val, ood_test, labels):
    def _max_prediction(id, ood, id_threshold=None):
        ood = ood["predictions"]
        id = id["predictions"]
        ood_labels = np.zeros(len(ood))
        id_labels = np.ones(len(id))
        y_ood_true = np.concatenate([id_labels, ood_labels])
        y_ood_pred = np.concatenate([id.max(axis=1).values, ood.max(axis=1).values])
        if id_threshold is not None:
            y_ood_pred = 1 * (y_ood_pred > id_threshold)
            return y_ood_true, y_ood_pred
        else:
            return metrics.calc_threshold_at_95(y_ood_true, y_ood_pred)

    return _output_based(
        cxr14_val,
        cxr14_test,
        ood_val,
        ood_test,
        labels,
        ood_detection_method=_max_prediction,
    )


def max_softmax(cxr14_val, cxr14_test, ood_val, ood_test, labels):
    def _max_softmax(id, ood, id_threshold=None):
        ood = ood["predictions"]
        id = id["predictions"]
        ood_labels = np.zeros(len(ood))
        id_labels = np.ones(len(id))
        id_softmax = torch.nn.functional.softmax(id, dim=1).max(axis=1).values.numpy()
        ood_softmax = torch.nn.functional.softmax(ood, dim=1).max(axis=1).values.numpy()
        y_ood_true = np.concatenate([id_labels, ood_labels])
        y_ood_pred = np.concatenate([id_softmax, ood_softmax])
        if id_threshold is not None:
            y_ood_pred = 1 * (y_ood_pred > id_threshold)
            return y_ood_true, y_ood_pred
        else:
            return metrics.calc_threshold_at_95(y_ood_true, y_ood_pred)

    return _output_based(
        cxr14_val,
        cxr14_test,
        ood_val,
        ood_test,
        labels,
        ood_detection_method=_max_softmax,
    )


def max_logit(cxr14_val, cxr14_test, ood_val, ood_test, labels):
    def _max_logit(id, ood, id_threshold=None):
        ood = ood["logits"]
        id = id["logits"]
        ood_labels = np.zeros(len(ood))
        id_labels = np.ones(len(id))
        id_logit = id.max(axis=1).values.numpy()
        ood_logit = ood.max(axis=1).values.numpy()
        y_ood_true = np.concatenate([id_labels, ood_labels])
        y_ood_pred = np.concatenate([id_logit, ood_logit])
        if id_threshold is not None:
            y_ood_pred = 1 * (y_ood_pred > id_threshold)
            return y_ood_true, y_ood_pred
        else:
            return metrics.calc_threshold_at_95(y_ood_true, y_ood_pred)

    return _output_based(
        cxr14_val,
        cxr14_test,
        ood_val,
        ood_test,
        labels,
        ood_detection_method=_max_logit,
    )


def max_energy(cxr14_val, cxr14_test, ood_val, ood_test, labels):
    def _max_energy(id, ood, id_threshold=None):
        ood = ood["logits"]
        id = id["logits"]
        ood_labels = np.zeros(len(ood))
        id_labels = np.ones(len(id))

        id_energy = torch.log(1 + torch.exp(id)).max(axis=1).values.numpy()
        ood_energy = torch.log(1 + torch.exp(ood)).max(axis=1).values.numpy()
        y_ood_true = np.concatenate([id_labels, ood_labels])
        y_ood_pred = np.concatenate([id_energy, ood_energy])
        if id_threshold is not None:
            y_ood_pred = 1 * (y_ood_pred > id_threshold)
            return y_ood_true, y_ood_pred
        else:
            return metrics.calc_threshold_at_95(y_ood_true, y_ood_pred)

    return _output_based(
        cxr14_val,
        cxr14_test,
        ood_val,
        ood_test,
        labels,
        ood_detection_method=_max_energy,
    )


def mahalanobis(cxr14_val,
                cxr14_test,
                ood_val,
                ood_test,
                labels,
                ):

    # Load class means and precisions
    # Generated using:
    # mahalanobis.sample_estimator(model, num_classes=14, num_activations=1024)
    directory = Path(cxr14_val).parent
    filename = directory / "mahalanobis_mean_precision.pth"
    sample_class_mean, precision = torch.load(filename)

    # Load Mahalanobis Score
    # Generated using:
    # mahalanobis_score = mahalanobis.get_Mahalanobis_score(
    #     model, test / val_dataloader, num_classes=14, sample_class_mean, precision,
    #     layer_index=0, magnitude=0.0, device=torch.device("cuda:0"))
    ood_dataset = Path(ood_val).stem.split("_")[0]  # irma, mura, boneage
    y_ood_val_pred = torch.load(directory / f"mahalanobis_val_{ood_dataset}.pth")

    cxr14_val = torch.load(cxr14_val)
    ood_val = torch.load(ood_val)
    y_ood_val_true = np.concatenate([np.ones(len(cxr14_val["predictions"])),
                                 np.zeros(len(ood_val["predictions"]))])

    # Calculates class thresholds on the ID and OOD validation set.
    id_threshold = metrics.calc_threshold_at_95(y_ood_val_true, y_ood_val_pred)

    # Calculate OOD Performance.
    y_ood_test_pred = torch.load(directory / f"mahalanobis_test_{ood_dataset}.pth")
    y_ood_test_pred = 1 * (y_ood_test_pred > id_threshold)

    cxr14_test = torch.load(cxr14_test)
    ood_test = torch.load(ood_test)
    y_ood_test_true = np.concatenate([np.ones(len(cxr14_test["predictions"])),
                                      np.zeros(len(ood_test["predictions"]))])
    report_sensitivity_and_specificity(
        y_ood_test_true, y_ood_test_pred, "With No Finding"
    )


    # Calculate OOD detection performance without No Finding labels.
    y_ood_test_pred_wo_no_finding = torch.load(directory
                                               / f"mahalanobis_test_irma_exclude_no_finding.pth")
    y_ood_test_pred_wo_no_finding = 1 * (y_ood_test_pred_wo_no_finding
                                         > id_threshold)
    cxr14_test_wo_no_finding = filter_no_finding(cxr14_test.copy())
    y_ood_test_true_wo_no_finding = np.concatenate([np.ones(len(cxr14_test_wo_no_finding["predictions"])),
                                      np.zeros(len(ood_test["predictions"]))])

    report_sensitivity_and_specificity(
        y_ood_test_true_wo_no_finding,
        y_ood_test_pred_wo_no_finding,
        "Without No Finding",
    )

    # Calculate end-to-end performance.
    report_end_to_end(cxr14_test, ood_test, y_ood_test_pred, labels)
