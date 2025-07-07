from typing import List, Dict, Optional, Union
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

    def __init__(self, nc: int, conf: Union[float, Dict[int, float]] = 0.25, iou_thres: float = 0.45):
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        # Matrix size is (num_classes + 1, num_classes + 1) to account for background (FP/FN)
        self.matrix = torch.zeros((nc + 1, nc + 1), dtype=torch.int64)
        self.eps = 1e-6
        
        # Initialize confidence tracking for analysis
        self.confidence_scores = []  # List of (confidence, predicted_class, true_class, is_match)
        self.class_stats = {i: {'tp': 0, 'fp': 0, 'fn': 0, 'total_detections': 0, 'total_gt': 0} for i in range(nc)}

    def _get_conf_threshold(self, class_id: int) -> float:
        """Get confidence threshold for a specific class."""
        if isinstance(self.conf, dict):
            return self.conf.get(class_id, 0.25)
        return self.conf

    def process_batch(self, detections: torch.Tensor, labels: torch.Tensor):
        """
        Update the confusion matrix with a batch of detections and ground truths.

        Args:
            detections (torch.Tensor): Tensor of detections, shape [N, 6] (x1, y1, x2, y2, conf, class).
            labels (torch.Tensor): Tensor of ground truths, shape [M, 5] (class, x1, y1, x2, y2).
        """
        # Store all detections for confidence analysis
        if detections.shape[0] > 0:
            for detection in detections:
                conf_score = detection[4].item()
                pred_class = int(detection[5].item())
                
                # Track all detections regardless of threshold for analysis
                self.confidence_scores.append({
                    'confidence': conf_score,
                    'predicted_class': pred_class,
                    'true_class': -1,  # Will be updated if matched
                    'is_match': False,
                    'above_threshold': conf_score >= self._get_conf_threshold(pred_class)
                })
                
                self.class_stats[pred_class]['total_detections'] += 1

        # Track ground truth counts
        if labels.shape[0] > 0:
            for label in labels:
                gt_class = int(label[0].item())
                self.class_stats[gt_class]['total_gt'] += 1

        # Apply class-specific confidence thresholds
        if detections.shape[0] > 0:
            filtered_detections = []
            for detection in detections:
                pred_class = int(detection[5].item())
                if detection[4] >= self._get_conf_threshold(pred_class):
                    filtered_detections.append(detection)
            
            if filtered_detections:
                detections = torch.stack(filtered_detections)
            else:
                detections = torch.empty((0, 6))

        # Handle cases with no detections or no labels
        if detections.shape[0] == 0:
            if labels.shape[0] > 0:
                for lb in labels:
                    gt_class = int(lb[0])
                    self.matrix[gt_class, self.nc] += 1  # All labels are False Negatives
                    self.class_stats[gt_class]['fn'] += 1
            return

        if labels.shape[0] == 0:
            for dt in detections:
                det_class = int(dt[5])
                self.matrix[self.nc, det_class] += 1  # All detections are False Positives
                self.class_stats[det_class]['fp'] += 1
                
                # Update confidence tracking for unmatched detections
                for conf_entry in reversed(self.confidence_scores):
                    if (conf_entry['predicted_class'] == det_class and 
                        conf_entry['true_class'] == -1):
                        conf_entry['true_class'] = -1  # Background/False Positive
                        break
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
                
                # Update class stats
                if gt_cls == det_cls:
                    self.class_stats[int(gt_cls)]['tp'] += 1
                else:
                    self.class_stats[int(det_cls)]['fp'] += 1
                    self.class_stats[int(gt_cls)]['fn'] += 1
                
                # Update confidence tracking for matched detections
                det_conf = detections[int(det_idx)][4].item()
                for conf_entry in reversed(self.confidence_scores):
                    if (conf_entry['predicted_class'] == int(det_cls) and 
                        conf_entry['confidence'] == det_conf and
                        conf_entry['true_class'] == -1):
                        conf_entry['true_class'] = int(gt_cls)
                        conf_entry['is_match'] = True
                        break

        # Unmatched Ground Truths are False Negatives (FN)
        for i, _ in enumerate(labels):
            if i not in matched_gt:
                gt_cls = gt_classes[i]
                self.matrix[gt_cls, self.nc] += 1
                self.class_stats[int(gt_cls)]['fn'] += 1

        # Unmatched Detections are False Positives (FP)
        for i, _ in enumerate(detections):
            if i not in matched_det:
                det_cls = detection_classes[i]
                self.matrix[self.nc, det_cls] += 1
                self.class_stats[int(det_cls)]['fp'] += 1
                
                # Update confidence tracking for unmatched detections
                det_conf = detections[i][4].item()
                for conf_entry in reversed(self.confidence_scores):
                    if (conf_entry['predicted_class'] == int(det_cls) and 
                        conf_entry['confidence'] == det_conf and
                        conf_entry['true_class'] == -1):
                        conf_entry['true_class'] = -1  # Background/False Positive
                        break

    def get_confidence_analysis(self, class_names: List[str]) -> Dict:
        """
        Analyze confidence score distributions for debugging.
        
        Returns:
            Dict containing confidence statistics for each class
        """
        analysis = {}
        
        # Overall statistics
        all_confidences = [entry['confidence'] for entry in self.confidence_scores]
        analysis['overall'] = {
            'total_detections': len(all_confidences),
            'mean_confidence': np.mean(all_confidences) if all_confidences else 0,
            'std_confidence': np.std(all_confidences) if all_confidences else 0,
            'min_confidence': np.min(all_confidences) if all_confidences else 0,
            'max_confidence': np.max(all_confidences) if all_confidences else 0
        }
        
        # Per-class analysis
        for class_id in range(self.nc):
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            # Get confidences for this predicted class
            class_confidences = [entry['confidence'] for entry in self.confidence_scores 
                               if entry['predicted_class'] == class_id]
            
            # Get confidences for correctly detected objects of this class
            correct_confidences = [entry['confidence'] for entry in self.confidence_scores 
                                 if entry['predicted_class'] == class_id and 
                                    entry['true_class'] == class_id and entry['is_match']]
            
            # Get confidences for incorrectly detected objects predicted as this class
            incorrect_confidences = [entry['confidence'] for entry in self.confidence_scores 
                                   if entry['predicted_class'] == class_id and 
                                      (entry['true_class'] != class_id or not entry['is_match'])]
            
            # Get confidences above current threshold
            above_threshold = [entry['confidence'] for entry in self.confidence_scores 
                             if entry['predicted_class'] == class_id and entry['above_threshold']]
            
            stats = self.class_stats[class_id]
            
            analysis[class_name] = {
                'class_id': class_id,
                'total_detections': len(class_confidences),
                'correct_detections': len(correct_confidences),
                'incorrect_detections': len(incorrect_confidences),
                'above_threshold_detections': len(above_threshold),
                'tp': stats['tp'],
                'fp': stats['fp'],
                'fn': stats['fn'],
                'total_gt': stats['total_gt'],
                'precision': stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0,
                'recall': stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0,
                'current_threshold': self._get_conf_threshold(class_id),
                'confidence_stats': {
                    'all_mean': np.mean(class_confidences) if class_confidences else 0,
                    'all_std': np.std(class_confidences) if class_confidences else 0,
                    'correct_mean': np.mean(correct_confidences) if correct_confidences else 0,
                    'correct_std': np.std(correct_confidences) if correct_confidences else 0,
                    'incorrect_mean': np.mean(incorrect_confidences) if incorrect_confidences else 0,
                    'incorrect_std': np.std(incorrect_confidences) if incorrect_confidences else 0,
                }
            }
            
            # Calculate potential thresholds and their impact
            if class_confidences:
                thresholds = np.percentile(class_confidences, [10, 25, 50, 75, 90])
                analysis[class_name]['suggested_thresholds'] = {
                    'p10': thresholds[0],
                    'p25': thresholds[1], 
                    'p50': thresholds[2],
                    'p75': thresholds[3],
                    'p90': thresholds[4]
                }
        
        return analysis

    def plot_confidence_distributions(self, class_names: List[str], save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of confidence score distributions by class and ground truth.
        """
        n_classes = min(self.nc, len(class_names))
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confidence Score Analysis for Detection Debugging', fontsize=16)
        
        # Plot 1: Confidence distribution by predicted class
        ax1 = axes[0, 0]
        all_data = []
        all_labels = []
        
        for class_id in range(n_classes):
            class_name = class_names[class_id]
            confidences = [entry['confidence'] for entry in self.confidence_scores 
                          if entry['predicted_class'] == class_id]
            if confidences:
                all_data.extend(confidences)
                all_labels.extend([class_name] * len(confidences))
        
        if all_data:
            ax1.hist([all_data], bins=30, alpha=0.7, label='All Detections')
            ax1.axvline(x=0.25, color='red', linestyle='--', label='Default Threshold (0.25)')
            ax1.set_xlabel('Confidence Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Overall Confidence Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Correct vs Incorrect detections by confidence
        ax2 = axes[0, 1]
        correct_confs = [entry['confidence'] for entry in self.confidence_scores 
                        if entry['is_match'] and entry['true_class'] == entry['predicted_class']]
        incorrect_confs = [entry['confidence'] for entry in self.confidence_scores 
                          if not entry['is_match'] or entry['true_class'] != entry['predicted_class']]
        
        if correct_confs or incorrect_confs:
            if correct_confs:
                ax2.hist(correct_confs, bins=20, alpha=0.7, label=f'Correct ({len(correct_confs)})', color='green')
            if incorrect_confs:
                ax2.hist(incorrect_confs, bins=20, alpha=0.7, label=f'Incorrect ({len(incorrect_confs)})', color='red')
            ax2.axvline(x=0.25, color='black', linestyle='--', label='Default Threshold')
            ax2.set_xlabel('Confidence Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Correct vs Incorrect Detection Confidence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Per-class confidence box plots
        ax3 = axes[1, 0]
        class_data = []
        class_labels = []
        
        for class_id in range(min(8, n_classes)):  # Limit to 8 classes for readability
            class_name = class_names[class_id]
            confidences = [entry['confidence'] for entry in self.confidence_scores 
                          if entry['predicted_class'] == class_id]
            if confidences:
                class_data.append(confidences)
                class_labels.append(f"{class_name}\n({len(confidences)})")
        
        if class_data:
            ax3.boxplot(class_data, labels=class_labels)
            ax3.axhline(y=0.25, color='red', linestyle='--', label='Default Threshold')
            ax3.set_ylabel('Confidence Score')
            ax3.set_title('Confidence Distribution by Predicted Class')
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Detection statistics per class
        ax4 = axes[1, 1]
        class_names_subset = class_names[:min(8, n_classes)]
        tp_counts = [self.class_stats[i]['tp'] for i in range(len(class_names_subset))]
        fp_counts = [self.class_stats[i]['fp'] for i in range(len(class_names_subset))]
        fn_counts = [self.class_stats[i]['fn'] for i in range(len(class_names_subset))]
        
        x = np.arange(len(class_names_subset))
        width = 0.25
        
        ax4.bar(x - width, tp_counts, width, label='True Positives', color='green', alpha=0.7)
        ax4.bar(x, fp_counts, width, label='False Positives', color='red', alpha=0.7)
        ax4.bar(x + width, fn_counts, width, label='False Negatives', color='orange', alpha=0.7)
        
        ax4.set_xlabel('Class')
        ax4.set_ylabel('Count')
        ax4.set_title('Detection Statistics by Class')
        ax4.set_xticks(x)
        ax4.set_xticklabels(class_names_subset, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot(self, class_names: List, normalize=True) -> plt.Figure:
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

    def get_matrix(self):
        return self.matrix
