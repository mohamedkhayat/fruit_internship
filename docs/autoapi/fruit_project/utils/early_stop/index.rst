fruit_project.utils.early_stop
==============================

.. py:module:: fruit_project.utils.early_stop


Classes
-------

.. autoapisummary::

   fruit_project.utils.early_stop.EarlyStopping


Module Contents
---------------

.. py:class:: EarlyStopping(patience: int, delta: float, path: str, name: str, cfg: omegaconf.DictConfig, run: wandb.sdk.wandb_run.Run)

   .. py:attribute:: patience


   .. py:attribute:: delta


   .. py:attribute:: path


   .. py:attribute:: name


   .. py:attribute:: cfg


   .. py:attribute:: run


   .. py:attribute:: best_metric
      :type:  Optional[float]
      :value: None



   .. py:attribute:: counter
      :value: 0



   .. py:attribute:: earlystop
      :value: False



   .. py:attribute:: saved_checkpoints
      :type:  List[Tuple[float, pathlib.Path]]
      :value: []



   .. py:method:: __call__(val_metric: float, model: torch.nn.Module) -> bool

      Checks if early stopping criteria are met and saves the model if the metric improves.

      :param val_metric: Validation metric to monitor.
      :type val_metric: float
      :param model: PyTorch model to save.
      :type model: nn.Module

      :returns: True if early stopping criteria are met, False otherwise.
      :rtype: bool



   .. py:method:: save_model(model: torch.nn.Module, val_metric: float)

      Saves the model checkpoint.

      :param model: PyTorch model to save.
      :type model: nn.Module
      :param val_metric: Validation metric value used for naming the checkpoint file.
      :type val_metric: float

      :returns: None



   .. py:method:: cleanup_checkpoints()

      Deletes all saved checkpoints except the best one.

      :returns: None



   .. py:method:: get_best_model(model: torch.nn.Module) -> torch.nn.Module

      Loads the best model checkpoint and sets the model to evaluation mode.

      :param model: PyTorch model to load the best checkpoint into.
      :type model: nn.Module

      :returns: The model with the best checkpoint loaded.
      :rtype: nn.Module



