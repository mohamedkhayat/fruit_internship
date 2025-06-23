fruit_project.models.model_factory
==================================

.. py:module:: fruit_project.models.model_factory


Attributes
----------

.. autoapisummary::

   fruit_project.models.model_factory.supported_models


Functions
---------

.. autoapisummary::

   fruit_project.models.model_factory.get_model
   fruit_project.models.model_factory.get_RTDETRv2


Module Contents
---------------

.. py:data:: supported_models

.. py:function:: get_model(cfg: omegaconf.DictConfig, device: torch.device, n_classes: int, id2lbl: Dict, lbl2id: Dict) -> Tuple[torch.nn.Module, albumentations.Compose, List, List, transformers.AutoImageProcessor]

   Retrieves and initializes a model based on the provided configuration.
   :param cfg: Configuration object containing model specifications.
   :type cfg: DictConfig
   :param device: The device on which the model will be loaded (e.g., 'cpu' or 'cuda').
   :type device: torch.device
   :param n_classes: Number of classes for the model's output.
   :type n_classes: int
   :param id2lbl: Mapping from class IDs to labels.
   :type id2lbl: dict
   :param lbl2id: Mapping from labels to class IDs.
   :type lbl2id: dict

   :returns: The initialized model.
   :rtype: torch.nn.Module

   :raises ValueError: If the specified model name in the configuration is not supported.


.. py:function:: get_RTDETRv2(device: torch.device, n_classes: int, id2label: dict, label2id: dict, cfg: omegaconf.DictConfig) -> Tuple[torch.nn.Module, albumentations.Compose, List, List, transformers.AutoImageProcessor]

   Loads the RT-DETRv2 model along with its configuration, processor, and transformations.

   :param device: The device to load the model onto (e.g., 'cpu', 'cuda').
   :type device: str
   :param n_classes: The number of classes for the object detection task.
   :type n_classes: int
   :param id2label: A dictionary mapping class IDs to class labels.
   :type id2label: dict
   :param label2id: A dictionary mapping class labels to class IDs.
   :type label2id: dict
   :param cfg: Configuration object containing model settings, including the model name.
   :type cfg: object

   :returns:

             A tuple containing:
                 - model (torch.nn.Module): The loaded RT-DETRv2 model moved to the specified device.
                 - transforms (callable): The transformation function for preprocessing input images.
                 - image_mean (list): The mean values used for image normalization.
                 - image_std (list): The standard deviation values used for image normalization.
                 - processor (AutoImageProcessor): The processor for handling image inputs.
   :rtype: tuple


