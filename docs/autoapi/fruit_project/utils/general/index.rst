fruit_project.utils.general
===========================

.. py:module:: fruit_project.utils.general


Functions
---------

.. autoapisummary::

   fruit_project.utils.general.set_seed
   fruit_project.utils.general.seed_worker
   fruit_project.utils.general.plot_img
   fruit_project.utils.general.unnormalize
   fruit_project.utils.general.is_hf_model


Module Contents
---------------

.. py:function:: set_seed(SEED: int) -> torch.Generator

   Sets the seed for reproducibility across various libraries.

   :param SEED: The seed value to use.
   :type SEED: int

   :returns: A PyTorch generator seeded with the given value.
   :rtype: torch.Generator


.. py:function:: seed_worker(worker_id: int, base_seed: int) -> None

   Seeds a worker for multiprocessing to ensure reproducibility.

   :param worker_id: The ID of the worker.
   :type worker_id: int
   :param base_seed: The base seed value.
   :type base_seed: int

   :returns: None


.. py:function:: plot_img(img, label: Optional[str] = None) -> None

   Plots an image using matplotlib.

   :param img: The image tensor to plot (shape: C x H x W).
   :type img: torch.Tensor
   :param label: The label to display as the title. Defaults to None.
   :type label: str, optional

   :returns: None


.. py:function:: unnormalize(img_tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor

   Unnormalizes an image tensor by reversing normalization.

   :param img_tensor: The normalized image tensor (shape: N x C x H x W or C x H x W).
   :type img_tensor: torch.Tensor
   :param mean: The mean used for normalization.
   :type mean: torch.Tensor
   :param std: The standard deviation used for normalization.
   :type std: torch.Tensor

   :returns: The unnormalized image tensor.
   :rtype: torch.Tensor


.. py:function:: is_hf_model(model) -> bool

   Checks if the given model is a Hugging Face PreTrainedModel.

   :param model: The model to check.

   :returns: True if the model is a Hugging Face PreTrainedModel, False otherwise.
   :rtype: bool


