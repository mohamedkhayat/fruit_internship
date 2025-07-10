from typing import Dict, List, Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.ops import box_iou
from transformers.image_transforms import center_to_corners_format
from dataclasses import dataclass
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


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:
    """Mean Average Precision evaluator for RT-DETRv2 - adapted for fruit_project."""

    def __init__(
        self,
        image_processor,
        device,
        threshold: float = 0.01,
        id2label: Optional[Dict[int, str]] = None,
    ):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label
        self.map_metric = MeanAveragePrecision(
            box_format="xyxy", class_metrics=True
        ).to(device)
        self.device = device

    def collect_image_sizes(self, targets):
        """Collect image sizes from targets."""
        image_sizes = []

        batch_image_sizes = []
        for target in targets:
            try:
                if "size" in target:
                    size = target["size"]
                elif "orig_size" in target:
                    size = target["orig_size"]
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

                boxes[:, [0, 2]] *= width
                boxes[:, [1, 3]] *= height

                post_processed_targets.append({"boxes": boxes, "labels": labels})
                continue

            elif "annotations" in target:
                annotations = target["annotations"]

                if not annotations:
                    post_processed_targets.append(
                        {
                            "boxes": torch.empty((0, 4), dtype=torch.float32),
                            "labels": torch.empty((0,), dtype=torch.int64),
                        }
                    )
                    continue

                boxes_xywh = np.array(
                    [ann["bbox"] for ann in annotations], dtype=np.float32
                )
                labels = np.array(
                    [ann["category_id"] for ann in annotations], dtype=np.int64
                )

                boxes_xyxy = boxes_xywh.copy()
                boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
                boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]

                boxes = torch.tensor(boxes_xyxy, device=self.device)
                labels = torch.tensor(labels, device=self.device)

                post_processed_targets.append({"boxes": boxes, "labels": labels})

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
        post_processed_predictions = []

        target_sizes = image_sizes[0] if image_sizes else torch.tensor([[480, 480]])

        for i, model_output in enumerate(predictions):
            if i < len(target_sizes):
                target_size = target_sizes[i : i + 1]
            else:
                target_size = torch.tensor([[480, 480]])

            try:
                post_processed_output = (
                    self.image_processor.post_process_object_detection(
                        model_output, threshold=self.threshold, target_sizes=target_size
                    )
                )
                post_processed_predictions.extend(post_processed_output)
            except Exception as e:
                print(f"⚠️ Post-processing failed for prediction {i}: {e}")
                post_processed_predictions.append(
                    {
                        "boxes": torch.empty((0, 4), dtype=torch.float32),
                        "scores": torch.empty((0,), dtype=torch.float32),
                        "labels": torch.empty((0,), dtype=torch.int64),
                    }
                )

        return post_processed_predictions
