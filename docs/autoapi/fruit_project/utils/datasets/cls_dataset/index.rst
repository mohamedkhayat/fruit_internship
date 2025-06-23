fruit_project.utils.datasets.cls_dataset
========================================

.. py:module:: fruit_project.utils.datasets.cls_dataset


Classes
-------

.. autoapisummary::

   fruit_project.utils.datasets.cls_dataset.CLS_DS


Module Contents
---------------

.. py:class:: CLS_DS(samples: List[Tuple[str, str]], labels: List, id2lbl, lbl2id, transforms=None)

   Bases: :py:obj:`torch.utils.data.Dataset`


   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.


   .. py:attribute:: samples


   .. py:attribute:: labels


   .. py:attribute:: id2lbl


   .. py:attribute:: lbl2id


   .. py:attribute:: transforms
      :value: None



   .. py:method:: __len__()


   .. py:method:: __getitem__(idx: int)


