import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.ops import box_iou

class ConfusionMatrix:
    """
    Object Detection Confusion Matrix inspired by Ultralytics.
    
    Args:
        nc (int): Number of classes.
        conf (float): Confidence threshold for detections.
        iou_thres (float): IoU threshold for matching.
    """
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        # Matrix size is (num_classes + 1, num_classes + 1) to account for background (FP/FN)
        self.matrix = torch.zeros((nc + 1, nc + 1), dtype=torch.int64)
        self.eps = 1e-6

    def process_batch(self, detections, labels):
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
                    self.matrix[int(lb[0]), self.nc] += 1  # All labels are False Negatives
            return
        
        if labels.shape[0] == 0:
            for dt in detections:
                self.matrix[self.nc, int(dt[5])] += 1  # All detections are False Positives
            return

        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        
        # Calculate IoU between all pairs of detections and labels
        iou = box_iou(labels[:, 1:], detections[:, :4])
        
        # Find the best detection for each ground truth
        x = torch.where(iou > self.iou_thres)
        
        if x[0].shape[0]:
            # Create a combined tensor of [gt_idx, det_idx, iou]
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
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
                
    def plot(self, class_names, normalize=True):
        """Generates and returns a matplotlib figure of the confusion matrix."""
        array = self.matrix.numpy().astype(float)
        if normalize:
            # Normalize by the number of true instances per class
            array /= (array.sum(1).reshape(-1, 1) + self.eps)
        
        # Add background class for plotting
        plot_names = class_names + ['background']
        
        fig, ax = plt.subplots(figsize=(14, 12), tight_layout=True)
        sns.heatmap(array, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                    xticklabels=plot_names, yticklabels=plot_names, ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Object Detection Confusion Matrix')
        
        return fig

    def get_matrix(self):
        return self.matrix