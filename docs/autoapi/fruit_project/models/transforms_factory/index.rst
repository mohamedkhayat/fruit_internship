fruit_project.models.transforms_factory
=======================================

.. py:module:: fruit_project.models.transforms_factory


Functions
---------

.. autoapisummary::

   fruit_project.models.transforms_factory.get_transforms


Module Contents
---------------

.. py:function:: get_transforms(cfg: omegaconf.DictConfig)

   Generates a dictionary of Albumentations transformations for training and testing.
   :param cfg: Configuration object containing the following attributes:
   :type cfg: DictConfig

   :returns: A dictionary with keys "train" and "test"
   :rtype: dict


