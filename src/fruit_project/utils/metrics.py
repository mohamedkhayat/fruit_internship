# SPDX-FileCopyrightText: 2025 Mohamed Khayat
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Dict, List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.ops import box_iou
from transformers.image_transforms import center_to_corners_format
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class ConfusionMatrix:
    """
    Object Detection Confusion Matrix inspired by Ultralytics.

    Args:
        nc (int): Number of classes.
        conf (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for matching.
    """

    def __init__(self, nc: int, conf: float = 0.25, iou_thres: float = 0.45) -> None:
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        # Matrix size is (num_classes + 1, num_classes + 1) to account for background (FP/FN)
        self.matrix = torch.zeros((nc + 1, nc + 1), dtype=torch.int64)
        self.eps = 1e-6

    def process_batch(self, detections: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Update the confusion matrix with a batch of detections and ground truths.

        Args:
            detections (torch.Tensor): Tensor of detections, shape [N, 6] (x1, y1, x2, y2, conf, class).
            labels (torch.Tensor): Tensor of ground truths, shape [M, 5] (class, x1, y1, x2, y2).
        """
        # Filter detections by confidence threshold
        detections = detections[detections[:, 4] >= self.conf]

        # Handle cases with no detections or no labels
        if detections.shape[0] == 0:
            if labels.shape[0] > 0:
                for lb in labels:
                    self.matrix[int(lb[0]), self.nc] += (
                        1  # All labels are False Negatives
                    )
            return

        if labels.shape[0] == 0:
            for dt in detections:
                self.matrix[self.nc, int(dt[5])] += (
                    1  # All detections are False Positives
                )
            return

        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()

        # Calculate IoU between all pairs of detections and labels
        iou = box_iou(labels[:, 1:], detections[:, :4])

        # Find the best detection for each ground truth
        x = torch.where(iou > self.iou_thres)

        if x[0].shape[0]:
            # Create a combined tensor of [gt_idx, det_idx, iou]
            matches = (
                torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                .cpu()
                .numpy()
            )
            if x[0].shape[0] > 1:
                # Greedy matching: sort by IoU and remove duplicates
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n_matches = matches.shape[0]
        matched_gt = set()
        matched_det = set()

        if n_matches:
            matched_gt.update(matches[:, 0].astype(int))
            matched_det.update(matches[:, 1].astype(int))
            for gt_idx, det_idx, _ in matches:
                gt_cls = gt_classes[int(gt_idx)]
                det_cls = detection_classes[int(det_idx)]
                self.matrix[gt_cls, det_cls] += 1

        # Unmatched Ground Truths are False Negatives (FN)
        for i, _ in enumerate(labels):
            if i not in matched_gt:
                gt_cls = gt_classes[i]
                self.matrix[gt_cls, self.nc] += 1

        # Unmatched Detections are False Positives (FP)
        for i, _ in enumerate(detections):
            if i not in matched_det:
                det_cls = detection_classes[i]
                self.matrix[self.nc, det_cls] += 1

    def update(self, preds, targets_for_cm):
        for i in range(len(preds)):
            pred_item = preds[i]
            gt_item = targets_for_cm[i]

            detections = torch.cat(
                [
                    pred_item["boxes"],
                    pred_item["scores"].unsqueeze(1),
                    pred_item["labels"].unsqueeze(1).float(),
                ],
                dim=1,
            )

            gt_boxes = gt_item["boxes"]
            gt_labels = gt_item["labels"]
            if gt_boxes.numel() > 0:
                labels = torch.cat([gt_labels.unsqueeze(1).float(), gt_boxes], dim=1)
            else:
                labels = torch.zeros((0, 5))

            self.process_batch(detections, labels)

    def plot(self, class_names: List[str], normalize: bool = True) -> plt.Figure:
        """Generates and returns a matplotlib figure of the confusion matrix."""
        array = self.matrix.numpy().astype(float)
        if normalize:
            # Normalize by the number of true instances per class
            array /= array.sum(1).reshape(-1, 1) + self.eps

        # Add background class for plotting
        plot_names = class_names + ["background"]

        fig, ax = plt.subplots(figsize=(14, 12), tight_layout=True)
        sns.heatmap(
            array,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=plot_names,
            yticklabels=plot_names,
            ax=ax,
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Object Detection Confusion Matrix")

        return fig

    def get_matrix(self) -> torch.Tensor:
        """
        Returns the raw confusion matrix tensor.

        Returns:
            torch.Tensor: The (nc + 1) x (nc + 1) confusion matrix.
        """
        return self.matrix


class MAPEvaluator:
    """Mean Average Precision evaluator for RT-DETRv2 - adapted for fruit_project."""

    def __init__(
        self,
        image_processor,
        device,
        threshold: float = 0.0,
        id2label: Optional[Dict[int, str]] = None,
    ):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label
        self.map_metric = MeanAveragePrecision(
            box_format="xyxy", class_metrics=True
        ).to(device)
        self.map_metric.warn_on_many_detections = False
        self.map_50_metric = MeanAveragePrecision(
            box_format="xyxy",
            class_metrics=True,
            iou_thresholds=[0.5],
            extended_summary=True,
        ).to(device)
        self.map_50_metric.warn_on_many_detections = False
        self.device = device

    def collect_image_sizes(self, targets):
        """Collect image sizes from targets."""
        image_sizes = []

        batch_image_sizes = []
        for target in targets:
            try:
                if "size" in target:
                    size = target["size"]
                else:
                    size = [480, 480]
                    print("⚠️ Using fallback image size [480, 480]")

                if torch.is_tensor(size):
                    size = size.tolist()
                batch_image_sizes.append(size)
            except Exception as e:
                print(f"⚠️ Error extracting size: {e}")
                batch_image_sizes.append([480, 480])

        image_sizes.append(torch.tensor(batch_image_sizes))
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        """Process ground truth targets - now handles HF-processed format."""
        post_processed_targets = []

        sizes = image_sizes[0] if image_sizes else []

        for i, target in enumerate(targets):
            if i < len(sizes):
                height, width = sizes[i].tolist()
            else:
                height, width = 480, 480

            if "boxes" in target and "class_labels" in target:
                boxes = target["boxes"]
                labels = target["class_labels"]

                boxes = center_to_corners_format(boxes)

                if boxes.device != self.device:
                    boxes = boxes.to(self.device)

                if labels.device != self.device:
                    labels = labels.to(self.device)

                boxes[:, [0, 2]] *= width
                boxes[:, [1, 3]] *= height

                post_processed_targets.append({"boxes": boxes, "labels": labels})
                continue

            else:
                post_processed_targets.append(
                    {
                        "boxes": torch.empty((0, 4), dtype=torch.float32),
                        "labels": torch.empty((0,), dtype=torch.int64),
                    }
                )

        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        """Process model predictions using HuggingFace post-processing."""

        target_sizes = image_sizes[0] if image_sizes else torch.empty((0, 2))

        post_processed_predictions = self.image_processor.post_process_object_detection(
            predictions,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )

        return post_processed_predictions

    def get_per_class(self, map_50_metrics, metric):
        per_class_metric = []
        class_names = [v for v in self.id2label.values()]
        if "classes" in map_50_metrics and metric in map_50_metrics:
            class_metric_dict = {
                c.item(): m.item()
                for c, m in zip(map_50_metrics["classes"], map_50_metrics[metric])
            }
            for i in range(len(class_names)):
                per_class_metric.append(class_metric_dict.get(i, 0.0))
        else:
            per_class_metric = [0.0] * len(class_names)

        per_class_metric = torch.tensor(per_class_metric)

        return per_class_metric

    def get_optimal_f1_ultralytics_style(self, metrics_dict):
        prec = metrics_dict["precision"]  # T×R×K×A×M
        classes_present = metrics_dict["classes"].to(self.device).long()

        # Debugging
        K = len(self.id2label)

        valid_mask = (classes_present >= 0) & (classes_present < K)
        classes_present_filtered = classes_present[valid_mask]

        # --- slice the tensor ---
        iou_idx = self.map_50_metric.iou_thresholds.index(0.5)
        prec_curves = prec[iou_idx, :, :, 0, -1].to(self.device)  # R×K
        rec_vec = torch.tensor(
            self.map_50_metric.rec_thresholds,
            dtype=prec_curves.dtype,
            device=self.device,
        )
        # --- compute F1 and pick best threshold per class ---
        f1 = (
            2
            * prec_curves
            * rec_vec[:, None]
            / (prec_curves + rec_vec[:, None] + 1e-16)
        )

        best_thr = torch.argmax(f1, dim=0)

        if len(classes_present_filtered) > 0:
            opt_p = prec_curves[
                best_thr[classes_present_filtered], classes_present_filtered
            ]
            opt_r = rec_vec[best_thr[classes_present_filtered]]
        else:
            opt_p = torch.empty(0, dtype=prec_curves.dtype, device=self.device)
            opt_r = torch.empty(0, dtype=rec_vec.dtype, device=self.device)

        P = torch.zeros(K, dtype=prec_curves.dtype, device=self.device)
        R = torch.zeros(K, dtype=rec_vec.dtype, device=self.device)

        if len(classes_present_filtered) > 0:
            P[classes_present_filtered] = opt_p
            R[classes_present_filtered] = opt_r

        return P, R

    def get_averaged_precision_recall_ultralytics_style(
        self,
        optimal_precisions: torch.Tensor,
        optimal_recalls: torch.Tensor,
        present_classes: torch.Tensor,
    ):
        """Calculate overall precision and recall..."""

        if len(present_classes) == 0:
            print("No present classes, returning 0.0, 0.0")
            return 0.0, 0.0

        present_class_ids = present_classes.long()

        if present_class_ids.max() >= optimal_precisions.shape[0]:
            valid_mask = present_class_ids < optimal_precisions.shape[0]
            present_class_ids = present_class_ids[valid_mask]

        if len(present_class_ids) == 0:
            print("No valid present classes after filtering, returning 0.0, 0.0")
            return 0.0, 0.0

        present_precisions = optimal_precisions[present_class_ids]
        present_recalls = optimal_recalls[present_class_ids]

        overall_precision = present_precisions.mean().item()
        overall_recall = present_recalls.mean().item()

        return overall_precision, overall_recall
