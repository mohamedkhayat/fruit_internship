fruit_project.utils.datasets.det_dataset
========================================

.. py:module:: fruit_project.utils.datasets.det_dataset


Classes
-------

.. autoapisummary::

   fruit_project.utils.datasets.det_dataset.DET_DS


Module Contents
---------------

.. py:class:: DET_DS(root_dir: str, type: str, image_dir: str, label_dir: str, config_file: str, transforms: albumentations.Compose = None, input_size: int = 224)

   Bases: :py:obj:`torch.utils.data.Dataset`


   A custom dataset class for object detection tasks.

   This dataset class loads images and their corresponding labels from specified directories,
   applies transformations if provided, and returns the processed image along with target annotations.

   .. attribute:: root_dir

      The root directory containing the dataset.

      :type: str

   .. attribute:: type

      The type of dataset (e.g., 'train', 'val', 'test').

      :type: str

   .. attribute:: image_dir

      The subdirectory containing the images.

      :type: str

   .. attribute:: label_dir

      The subdirectory containing the label files.

      :type: str

   .. attribute:: config_file

      The path to the configuration file containing class names.

      :type: str

   .. attribute:: transforms

      A function or object to apply transformations to the images and annotations.

      :type: Albumentations Compose, optional

   .. attribute:: input_size

      The input size for the images (default is 224).

      :type: int

   .. attribute:: image_paths

      A list of valid image file paths.

      :type: list

   .. attribute:: labels

      A list of class names.

      :type: list

   .. attribute:: id2lbl

      A mapping from class IDs to class names.

      :type: dict

   .. attribute:: lbl2id

      A mapping from class names to class IDs.

      :type: dict

   .. method:: __len__()

      Returns the number of valid images in the dataset.

   .. method:: __getitem__(idx)

      Returns the processed image and target annotations for the given index.
      

   :param root_dir: The root directory containing the dataset.
   :type root_dir: str
   :param type: The type of dataset (e.g., 'train', 'val', 'test').
   :type type: str
   :param image_dir: The subdirectory containing the images.
   :type image_dir: str
   :param label_dir: The subdirectory containing the label files.
   :type label_dir: str
   :param config_file: The path to the configuration file containing class names.
   :type config_file: str
   :param transforms: A function or object to apply transformations to the images and annotations.
   :type transforms: Albumentations Compose, optional
   :param input_size: The input size for the images (default is 224).
   :type input_size: int, optional

   :raises FileNotFoundError: If the configuration file or label files are not found.
   :raises ValueError: If an image cannot be loaded or is invalid.


   .. py:attribute:: root_dir


   .. py:attribute:: type


   .. py:attribute:: image_dir


   .. py:attribute:: label_dir


   .. py:attribute:: transforms
      :value: None



   .. py:attribute:: input_size
      :value: 224



   .. py:attribute:: config_dir


   .. py:attribute:: image_paths
      :value: []



   .. py:attribute:: labels


   .. py:attribute:: id2lbl


   .. py:attribute:: lbl2id


   .. py:method:: __len__()

      :returns: The number of valid images in the dataset.
      :rtype: int



   .. py:method:: __getitem__(idx)

      Retrieves the processed image and target annotations for the given index.

      :param idx: The index of the image to retrieve.
      :type idx: int

      :returns:

                A tuple containing:
                    - img (numpy.ndarray): The processed image.
                    - target (dict): A dictionary containing target annotations, including:
                        - image_id (int): The index of the image.
                        - annotations (list): A list of dictionaries with bounding box, category ID, area, and iscrowd flag.
                        - orig_size (torch.Tensor): The original size of the image (height, width).
      :rtype: tuple



