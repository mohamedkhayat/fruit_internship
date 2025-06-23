fruit_project.utils.trainer
===========================

.. py:module:: fruit_project.utils.trainer


Classes
-------

.. autoapisummary::

   fruit_project.utils.trainer.Trainer


Module Contents
---------------

.. py:class:: Trainer(model: torch.nn.Module, processor: transformers.AutoImageProcessor, device: torch.device, cfg: omegaconf.DictConfig, name: str, run: wandb.sdk.wandb_run.Run, train_dl: torch.utils.data.DataLoader, test_dl: torch.utils.data.DataLoader, val_dl: torch.utils.data.DataLoader)

   .. py:attribute:: model


   .. py:attribute:: device


   .. py:attribute:: scaler


   .. py:attribute:: cfg


   .. py:attribute:: optimizer


   .. py:attribute:: processor


   .. py:attribute:: name


   .. py:attribute:: early_stopping


   .. py:attribute:: scheduler


   .. py:attribute:: run


   .. py:attribute:: train_dl


   .. py:attribute:: test_dl


   .. py:attribute:: val_dl


   .. py:attribute:: start_epoch
      :value: 0



   .. py:attribute:: accum_steps


   .. py:method:: get_scheduler() -> torch.optim.lr_scheduler.SequentialLR

      Creates a learning rate scheduler with a warmup phase.

      :returns: The learning rate scheduler.
      :rtype: SequentialLR



   .. py:method:: get_optimizer() -> torch.optim.AdamW

      Creates an AdamW optimizer with different learning rates for backbone and other parameters.

      :returns: The optimizer.
      :rtype: AdamW



   .. py:method:: move_labels_to_device(batch: transformers.BatchEncoding) -> transformers.BatchEncoding

      Moves label tensors within a batch to the specified device.

      :param batch: The batch containing labels.
      :type batch: BatchEncoding

      :returns: The batch with labels moved to the device.
      :rtype: BatchEncoding



   .. py:method:: nested_to_cpu(obj: Any) -> Any

      Recursively moves tensors in a nested structure (dict, list, tuple) to CPU.

      :param obj: The object containing tensors to move.

      :returns: The object with all tensors moved to CPU.



   .. py:method:: format_targets_for_map(y: List) -> List

      Formats target annotations for MeanAveragePrecision metric calculation.

      :param y: A list of target dictionaries.
      :type y: List

      :returns: A list of formatted target dictionaries for the metric.
      :rtype: List



   .. py:method:: train(current_epoch: int) -> float

      Performs one epoch of training.

      :param current_epoch: The current epoch number.
      :type current_epoch: int

      :returns: The average training loss for the epoch.
      :rtype: float



   .. py:method:: eval(test_dl: torch.utils.data.DataLoader, current_epoch: int, calc_cm: bool = False) -> Tuple[float, float, float, torch.Tensor, Optional[fruit_project.utils.metrics.ConfusionMatrix]]

      Evaluates the model on a given dataloader.

      :param test_dl: The dataloader for evaluation.
      :type test_dl: DataLoader
      :param current_epoch: The current epoch number.
      :type current_epoch: int
      :param calc_cm: Whether to calculate and return a confusion matrix. Defaults to False.
      :type calc_cm: bool, optional

      :returns:

                A tuple containing:
                    - loss (float): The average evaluation loss.
                    - test_map (float): The mAP@.5-.95.
                    - test_map50 (float): The mAP@.50.
                    - test_map_50_per_class (torch.Tensor): The mAP@.50 for each class.
                    - cm (ConfusionMatrix | None): The confusion matrix if calc_cm is True, else None.
      :rtype: tuple



   .. py:method:: fit() -> None

      Runs the main training loop for the specified number of epochs.



   .. py:method:: _save_checkpoint(epoch: int) -> str

      Saves a checkpoint of the model, optimizer, scheduler, and scaler states.

      :param epoch: The current epoch number.
      :type epoch: int

      :returns: The path to the saved checkpoint file.
      :rtype: str



   .. py:method:: _load_checkpoint(path: str) -> None

      Loads a checkpoint and restores the state of the model, optimizer, scheduler, and scaler.

      :param path: The path to the checkpoint file.
      :type path: str



   .. py:method:: get_epoch_log_data(epoch: int, train_loss: float, test_map: float, test_map50: float, test_loss: float, test_map_per_class: torch.Tensor) -> Dict[str, Any]

      Constructs a dictionary of metrics for logging at the end of an epoch.

      :param epoch: The current epoch number.
      :type epoch: int
      :param train_loss: The training loss.
      :type train_loss: float
      :param test_map: The test mAP@.5-.95.
      :type test_map: float
      :param test_map50: The test mAP@.50.
      :type test_map50: float
      :param test_loss: The test loss.
      :type test_loss: float
      :param test_map_per_class: The test mAP@.50 for each class.
      :type test_map_per_class: torch.Tensor

      :returns: A dictionary of metrics for logging.
      :rtype: dict



   .. py:method:: get_val_log_data(epoch: int, best_test_map: float) -> Dict[str, Any]

      Performs final validation, logs metrics, and returns the log data.

      :param epoch: The final epoch number.
      :type epoch: int
      :param best_test_map: The best test mAP@.50 achieved during training.
      :type best_test_map: float

      :returns: A dictionary of validation metrics for logging.
      :rtype: dict



