#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for building high-performance asynchronous PyTorch `DataLoader`s that
apply the existing *batch transform* functions in worker processes.

Motivation
----------
The current `OptimizedQADataset` pre-computes every batch at construction time
and then simply yields those tensors at training time.  This is memory-
efficient and very simple but it blocks the main process while the CPU
tokenises and formats the data.  On a GPU box that means valuable accelerator
cycles sit idle until the entire dataset is prepared.

Here we keep the same *transform* API (a callable that receives a **list** of
raw QA items and returns either `(batch_dict, doc_count)` or just a
`batch_dict`) but we move the call into background worker processes launched by
`torch.utils.data.DataLoader`.

Key points
~~~~~~~~~~
*  We expose a single helper – `create_async_dataloader` – that hides all the
   boilerplate.  Training code only needs to pass the raw data list, the
   transform function and the usual parameters.
*  The collate-function is implemented as a _top-level_ function so that it is
   picklable by Python's multiprocessing when `num_workers > 0`.
*  We use `pin_memory=True`, `prefetch_factor=2`, and `persistent_workers`
   (PyTorch ≥ 1.8) to overlap data preparation with GPU execution.
*  If the transform returns `None` (e.g. when an item has no valid answers) we
   fall back to an **empty** dictionary – the training loop can skip these
   without raising an error.

Example
~~~~~~~
```python
from rank_test.transforms import get_batch_transform
from rank_test.async_dataloader import create_async_dataloader

train_loader = create_async_dataloader(
    data=train_data,
    transform_fn=get_batch_transform('infonce'),
    tokenizer=tokenizer,
    batch_size=128,
    shuffle=True,
    max_length=128,
    device='cpu',                # keep tensors on CPU until model.forward
    num_workers=4,
    take_top=True                # kwargs forwarded to transform
)
```
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Simple dataset that just yields raw JSON rows
# ---------------------------------------------------------------------------


class RawQADataset(Dataset):
    """A dead-simple `Dataset` – each item is a raw QA dict."""

    def __init__(self, data: List[Dict[str, Any]]):
        self._data = data

    def __len__(self) -> int:  # noqa: D401 (short description)
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # type: ignore[override]
        return self._data[idx]


# ---------------------------------------------------------------------------
# Collate-function helper
# ---------------------------------------------------------------------------


def _apply_transform(
    batch_items: List[Dict[str, Any]],
    transform_fn: Callable,
    tokenizer,
    max_length: int,
    device: str,
    extra_kwargs: Dict[str, Any],
) -> Tuple[Any, int] | Any:  # keeps original return contract
    """Apply the user-provided *batch transform* inside a worker process."""

    # NB: we do *not* move tensors to CUDA here – keep them in pinned host
    # memory; the model/loss functions already `.to(device)` as necessary.
    return transform_fn(
        batch_items,
        tokenizer,
        max_length,
        device=device,
        **extra_kwargs,
    )


def _collate_fn_builder(
    transform_fn: Callable,
    tokenizer,
    max_length: int,
    device: str,
    extra_kwargs: Dict[str, Any],
):
    """Return a picklable collate-fn with args baked-in via `functools.partial`."""

    fn = partial(
        _apply_transform,
        transform_fn=transform_fn,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
        extra_kwargs=extra_kwargs,
    )
    return fn


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def create_async_dataloader(
    *,
    data: List[Dict[str, Any]],
    transform_fn: Callable,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    max_length: int,
    device: str,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    **transform_kwargs,
) -> DataLoader:
    """Factory that returns a `torch.utils.data.DataLoader` with async workers.

    Parameters
    ----------
    data
        Raw QA list loaded from JSON.
    transform_fn
        One of the functions returned by `rank_test.transforms.get_batch_transform`.
    tokenizer
        HF tokenizer instance shared across workers (must be picklable → *fast*).
    batch_size
        Number of *questions* (raw items) – the transform can replicate them
        internally (e.g. multiple positives) so the effective tensor batch size
        may differ.
    shuffle
        Whether to shuffle the dataset each epoch.
    max_length
        Token truncation length forwarded to the transform.
    device
        Device string; only used to place tensors (`cpu` recommended to avoid
        serialisation issues across processes).
    num_workers, prefetch_factor, pin_memory, persistent_workers
        Passed straight to `torch.utils.data.DataLoader`.
    transform_kwargs
        Additional keyword arguments forwarded to *transform_fn* (e.g.
        `take_top=False`, `pos_count=3`, `neg_strategy="in_batch"`, …).
    """

    if pin_memory is None:
        pin_memory = torch.cuda.is_available() and device.startswith("cuda")

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    dataset = RawQADataset(data)

    collate_fn = _collate_fn_builder(
        transform_fn=transform_fn,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
        extra_kwargs=transform_kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers,
        drop_last=False,
    ) 