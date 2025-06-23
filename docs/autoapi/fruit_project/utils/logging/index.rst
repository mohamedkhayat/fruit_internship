fruit_project.utils.logging
===========================

.. py:module:: fruit_project.utils.logging


Functions
---------

.. autoapisummary::

   fruit_project.utils.logging.initwandb
   fruit_project.utils.logging.get_run_name
   fruit_project.utils.logging.log_images
   fruit_project.utils.logging.log_transforms
   fruit_project.utils.logging.log_training_time
   fruit_project.utils.logging.log_model_params
   fruit_project.utils.logging.log_class_value_counts
   fruit_project.utils.logging.log_checkpoint_artifact
   fruit_project.utils.logging.log_detection_confusion_matrix


Module Contents
---------------

.. py:function:: initwandb(cfg: omegaconf.DictConfig) -> wandb.sdk.wandb_run.Run

   Initializes a wandb run.

   :param cfg: Configuration object.
   :type cfg: DictConfig

   :returns: The wandb run object.
   :rtype: Run


.. py:function:: get_run_name(cfg: omegaconf.DictConfig) -> str

   Generates a run name based on the configuration.

   :param cfg: Configuration object.
   :type cfg: DictConfig

   :returns: The generated run name.
   :rtype: str


.. py:function:: log_images(run: wandb.sdk.wandb_run.Run, batch: Tuple[Dict, List], id2lbl: Dict, grid_size: Tuple[int, int] = (3, 3), mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None) -> None

   Logs a grid of images with their bounding boxes to wandb.

   :param run: The wandb run object.
   :type run: Run
   :param batch: A single batch of data (processed_batch, targets).
   :type batch: Tuple[Dict, List]
   :param id2lbl: A dictionary mapping class IDs to labels.
   :type id2lbl: Dict
   :param grid_size: The grid size for displaying images. Defaults to (3, 3).
   :type grid_size: Tuple[int, int], optional
   :param mean: The mean used for normalization. Defaults to None.
   :type mean: Optional[torch.Tensor], optional
   :param std: The standard deviation used for normalization. Defaults to None.
   :type std: Optional[torch.Tensor], optional


.. py:function:: log_transforms(run: wandb.sdk.wandb_run.Run, batch: Tuple[Dict, List], grid_size: Tuple[int, int] = (3, 3), id2lbl: Optional[Dict] = None, transforms: Optional[Dict] = None, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None) -> None

   Logs a grid of transformed images with their bounding boxes to wandb.

   :param run: The wandb run object.
   :type run: Run
   :param batch: A single batch of data (processed_batch, targets).
   :type batch: Tuple[Dict, List]
   :param grid_size: The grid size for displaying images. Defaults to (3, 3).
   :type grid_size: Tuple[int, int], optional
   :param id2lbl: A dictionary mapping class IDs to labels. Defaults to None.
   :type id2lbl: Optional[Dict], optional
   :param transforms: The transforms applied. Defaults to None.
   :type transforms: Optional[Dict], optional
   :param mean: The mean used for normalization. Defaults to None.
   :type mean: Optional[torch.Tensor], optional
   :param std: The standard deviation used for normalization. Defaults to None.
   :type std: Optional[torch.Tensor], optional


.. py:function:: log_training_time(run: wandb.sdk.wandb_run.Run, start_time: float) -> None

   Logs the elapsed training time.

   :param run: The wandb run object.
   :type run: Run
   :param start_time: The start time of training.
   :type start_time: float


.. py:function:: log_model_params(run: wandb.sdk.wandb_run.Run, model: torch.nn.Module) -> None

   Logs the total and trainable parameters of a model.

   :param run: The wandb run object.
   :type run: Run
   :param model: The model.
   :type model: nn.Module


.. py:function:: log_class_value_counts(run: wandb.sdk.wandb_run.Run, samples: List[Tuple[str, str]], stage: str = 'Train') -> None

   Logs the class distribution of a dataset.

   :param run: The wandb run object.
   :type run: Run
   :param samples: A list of samples (e.g., [(image, label), ...]).
   :type samples: List[Tuple[Any, Any]]
   :param stage: The dataset stage (e.g., 'Train', 'Test'). Defaults to "Train".
   :type stage: str, optional


.. py:function:: log_checkpoint_artifact(run: wandb.sdk.wandb_run.Run, path: str, name: str, epoch: int, wait: bool = False) -> None

   Logs a model checkpoint as a wandb artifact.

   :param run: The wandb run object.
   :type run: Run
   :param path: The path to the checkpoint file.
   :type path: str
   :param name: The name of the artifact.
   :type name: str
   :param epoch: The epoch number.
   :type epoch: int
   :param wait: Whether to wait for the artifact to be uploaded. Defaults to False.
   :type wait: bool, optional


.. py:function:: log_detection_confusion_matrix(run: wandb.sdk.wandb_run.Run, cm_object: fruit_project.utils.metrics.ConfusionMatrix, class_names: List[str]) -> None

   Logs a detection confusion matrix plot to wandb.

   :param run: The wandb run object.
   :type run: Run
   :param cm_object: The confusion matrix object.
   :type cm_object: ConfusionMatrix
   :param class_names: The list of class names.
   :type class_names: List[str]


