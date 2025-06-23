fruit_project.utils.metrics
===========================

.. py:module:: fruit_project.utils.metrics


Classes
-------

.. autoapisummary::

   fruit_project.utils.metrics.ConfusionMatrix


Module Contents
---------------

.. py:class:: ConfusionMatrix(nc: int, conf: float = 0.25, iou_thres: float = 0.45)

   Object Detection Confusion Matrix inspired by Ultralytics.

   :param nc: Number of classes.
   :type nc: int
   :param conf: Confidence threshold for detections.
   :type conf: float
   :param iou_thres: IoU threshold for matching.
   :type iou_thres: float


   .. py:attribute:: nc


   .. py:attribute:: conf
      :value: 0.25



   .. py:attribute:: iou_thres
      :value: 0.45



   .. py:attribute:: matrix


   .. py:attribute:: eps
      :value: 1e-06



   .. py:method:: process_batch(detections: torch.Tensor, labels: torch.Tensor)

      Update the confusion matrix with a batch of detections and ground truths.

      :param detections: Tensor of detections, shape [N, 6] (x1, y1, x2, y2, conf, class).
      :type detections: torch.Tensor
      :param labels: Tensor of ground truths, shape [M, 5] (class, x1, y1, x2, y2).
      :type labels: torch.Tensor



   .. py:method:: plot(class_names: List, normalize=True) -> matplotlib.pyplot.Figure

      Generates and returns a matplotlib figure of the confusion matrix.



   .. py:method:: get_matrix()


