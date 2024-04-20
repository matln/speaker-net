"""PyTorch compatible DataLoaders

Essentially we extend PyTorch DataLoader by adding the ability to save the
data loading state, so that a checkpoint may be saved in the middle of an
epoch.

Copyright 2020 Aku Rouhe
          2022 Jianchen Li
"""
import torch
import random
import logging
import itertools
import warnings
import functools
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data.dataloader import (
    _utils,
    _BaseDataLoaderIter,
    _MultiProcessingDataLoaderIter
)
if torch.__version__ >= "1.11":
    from torch.utils.data.dataloader import IterDataPipe, _share_dist_seed

from speakernet.training.checkpointer import (
    register_checkpoint_hooks,
    mark_as_saver,
    mark_as_loader,
)

logger = logging.getLogger(__name__)

# We essentially want to make the DataLoader iterators able to skip ahead
# after checkpoint recovery
# This should be handled by the DataLoader iterators' base class.
# To make the implementation here a little more maintainable
# we decide to patch some PyTorch functionality


def __new_init(self, loader, *args, **kwargs):
    self.__old_init__(loader, *args, **kwargs)
    if (
        hasattr(loader, "_speakernet_recovery_skip_to")
        and loader._speakernet_recovery_skip_to is not None
    ):
        # Fast forward the sampler iterator since we have recovered:
        for i in range(loader._speakernet_recovery_skip_to):
            try:
                next(self._sampler_iter)
            except StopIteration:
                MSG = "Tried to fast-forward Sampler after checkpoint "
                f"recovery by {loader._speakernet_recovery_skip_to} "
                "indices, but now Sampler raised StopIteration after "
                f"{i} indices. Ignoring this mismatch."
                warnings.warn(MSG)
                break
            self._num_yielded = i + 1
        # Mark recovery as done:
        loader._speakernet_recovery_skip_to = None


def __new_reset(self, loader, first_iter=False, *args, **kwargs):
    # Reset the worker queue cycle so it resumes next epoch at worker 0
    self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))

    # On the first iteration, these have already normally been set by the init anyway.
    # And we don't want to overwrite them if we've recovered
    if (
        first_iter
        and hasattr(loader, "_speakernet_next_worker")     # For val_loader that not used SaveableDataLoader
        and loader._speakernet_next_worker is not None
    ):
        # For MultiprocessDataLoader, start from the worker that we have recovered
        for _ in range(loader._speakernet_next_worker):
            next(self._worker_queue_idx_cycle)

        # Mark recovery as done:
        loader._speakernet_next_worker = None
    else:
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        self._IterableDataset_len_called = loader._IterableDataset_len_called
        loader._speakernet_num_batches = len(loader)
        if torch.__version__ >= "1.11" and isinstance(self._dataset, IterDataPipe):
            self._shared_seed = _share_dist_seed(loader.generator, self._pg)
            shared_rng = torch.Generator()
            shared_rng.manual_seed(self._shared_seed)
            self._dataset = torch.utils.data.graph_settings.apply_random_seed(self._dataset, shared_rng)

    self._send_idx = 0  # idx of the next task to be sent to workers
    self._rcvd_idx = 0  # idx of the next task to be returned in __next__
    # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
    # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
    #                  \ (worker_id, data)   if data is already fetched (out-of-order)
    self._task_info = {}
    self._tasks_outstanding = 0  # always equal to count(v for v in task_info.values() if len(v) == 1)
    # A list of booleans representing whether each worker still has work to
    # do, i.e., not having exhausted its iterable dataset object. It always
    # contains all `True`s if not using an iterable-style dataset
    # (i.e., if kind != Iterable).
    # Not that this indicates that a worker still has work to do *for this epoch*.
    # It does not mean that a worker is dead. In case of `_persistent_workers`,
    # the worker will be reset to available in the next epoch.
    self._workers_status = [True for i in range(self._num_workers)]
    # We resume the prefetching in case it was enabled
    if not first_iter:
        for idx in range(self._num_workers):
            self._index_queues[idx].put(_utils.worker._ResumeIteration(self._shared_seed))
        resume_iteration_cnt = self._num_workers
        while resume_iteration_cnt > 0:
            return_idx, return_data = self._get_data()
            if isinstance(return_idx, _utils.worker._ResumeIteration):
                assert return_data is None
                resume_iteration_cnt -= 1
    # prime the prefetch loop
    for _ in range(self._prefetch_factor * self._num_workers):
        self._try_put_index()





# functools.update_wrapper is meant for decorators, but it should basically
# preserve what we want:
functools.update_wrapper(__new_init, _BaseDataLoaderIter.__init__)
_BaseDataLoaderIter.__old_init__ = _BaseDataLoaderIter.__init__
_BaseDataLoaderIter.__init__ = __new_init
# if hasattr(_BaseDataLoaderIter, "_reset"):
#     _BaseDataLoaderIter._reset = __new_reset
_MultiProcessingDataLoaderIter._reset = __new_reset


@register_checkpoint_hooks
class SaveableDataLoader(DataLoader):
    """A saveable version of the PyTorch DataLoader.

    See `torch.utils.data.DataLoader` for usage. This class should work exactly
    like the PyTorch basic DataLoader, but this can be checkpointed with
    the Checkpointer.

    Note
    ----
    1. The saveability is implemented via some unfortunately slightly magical
    means.
    2. The data loader cannot recover after entering __iter__. Normally this is
    not a problem, as recovery should happen before training begins.  However,
    just before evaluation, it is also typical to recover the checkpoint at
    which performance was the best. Thus, if a checkpoint is loaded after
    entering __iter__, we just assume it is for this reason. A warning is
    logged, but that is all.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "SaveableDataLoader cannot save the position in an "
                "IterableDataset. Save the position on the dataset itself."
            )
        self._speakernet_recovery_skip_to = None
        self._speakernet_iterator = None
        # For _MultiProcessingDataLoaderIter._try_put_index().worker_queue_idx
        self._speakernet_next_worker = None
        self._speakernet_num_batches = len(self)

    def __iter__(self):
        iterator = super().__iter__()
        # Keep a reference to the iterator,
        # to be able to access the iterator._num_yielded value.
        # Keep a full reference (keeping the iterator alive)
        # rather than e.g. a weakref, as we may want to save a checkpoint
        # after the iterator has been exhausted, but before the full epoch has
        # ended (e.g. validation is still running)
        self._speakernet_iterator = iterator
        return iterator

    def _speakernet_next_worker_id(self):
        # idx of the next task to be returned in __next__
        rcvd_idx = self._speakernet_iterator._rcvd_idx
        if rcvd_idx < self._speakernet_num_batches:
            # task_info of the current task has been deleted with torch.utils.data.dataloader:1202
            # if of the next task:
            next_worker_id = self._speakernet_iterator._task_info[rcvd_idx][0]
        else:
            next_worker_id = len(self) % self.num_workers
        return next_worker_id

    @mark_as_saver
    def _save(self, path):
        if isinstance(self.dataset, IterableDataset):
            logging.warning(
                "Warning again: a checkpoint was requested on "
                "SaveableDataLoader, but the dataset is an IterableDataset. "
                "Cannot save the position in an IterableDataset. Not raising "
                "an error; assuming that you know what you're doing."
            )
        if self._speakernet_iterator is None:
            to_save = None
        else:
            to_save = [self._speakernet_iterator._num_yielded]

            # MultiprocessDataLoader
            if self.num_workers > 0:
                to_save.append(self._speakernet_next_worker_id())

                to_save.append(self.dataset.worker_state)
            else:
                if 'aug_state' in self.dataset.worker_state:
                    to_save.append(self.dataset.worker_state['aug_state'])

            torch.save(to_save, path)

    @mark_as_loader
    def _recover(self, path, end_of_epoch, device=None):
        del device  # Unused here
        if self._speakernet_iterator is not None:
            logging.debug(
                "SaveableDataLoader was requested to load a "
                "checkpoint, but the DataLoader has already been "
                "iterated. The DataLoader file will be ignored. "
                "This is normal in evaluation, when a checkpoint is "
                "loaded just to retrieve the best model."
            )
            return

        saved = torch.load(path)
        if saved is None:
            # Saved at a point where e.g. an iterator did not yet exist.
            return
        else:
            if self.num_workers == 0:
                if not end_of_epoch:
                    # Don't load at end of epoch, as we actually want to start a fresh
                    # epoch iteration next.
                    self._speakernet_recovery_skip_to = saved[0]
                if len(saved) > 1:
                    self.dataset.aug.recover_aug_state(saved[1])
            else:
                if not end_of_epoch:
                    # Don't load at end of epoch, as we actually want to start a fresh
                    # epoch iteration next.
                    self._speakernet_recovery_skip_to = saved[0]
                    self._speakernet_next_worker = saved[1]
                    self.dataset.worker_state = saved[2]
                    self._speakernet_num_batches -= self._speakernet_recovery_skip_to


def worker_init_fn(worker_id):
    dataset = torch.utils.data.get_worker_info().dataset
    if dataset.worker_state != {} and worker_id in dataset.worker_state:
        np.random.set_state(dataset.worker_state[worker_id]['np_state'])
        # random_state (tuple) will be converted to a list when pin_memory=True. See pytorch issue #48419
        random_state = dataset.worker_state[worker_id]['random_state']
        random_state = (random_state[0], tuple(random_state[1]), random_state[2])
        random.setstate(random_state)
        torch.set_rng_state(dataset.worker_state[worker_id]['torch_state'])
        if 'aug_state' in dataset.worker_state[worker_id]:
            dataset.aug.recover_aug_state(dataset.worker_state[worker_id]['aug_state'])
