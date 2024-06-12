"""Data-Juicer Dataset."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Optional
from datasets import Dataset as HFD
from ray.data import Dataset as RD


class Dataset(ABC):
    """Base dataset of DJ"""
    @abstractmethod
    def filter(
        self,
        func: Callable, 
        num_proc: Optional[int],
        batched: bool = False,
    ) -> Dataset:
        pass

    @abstractmethod
    def mapper(
        self,
        func: Callable,
        num_proc: Optional[int],
        batched: bool = False,
    ) -> Dataset:
        pass

    @abstractmethod
    def export(self, path: str, **kwargs) -> bool:
        """export dataset into the path."""
        pass

    @classmethod
    def load(cls, path: str, mode="standalone") -> Dataset:
        """Load dataset from the path."""
        pass

class HFDataset(Dataset):
    """A wrapper of huggingface dataset.
    """

    def __init__(self, dataset: HFD) -> None:
        self.dataset = dataset

    def map(
        self,
        func: Callable, 
        num_proc: Optional[int],
        batched: bool = False,
    ) -> Dataset:
        return self.dataset.map(func, num_proc=num_proc, batched=batched)

    def filter(
        self,
        func: Callable,
        num_proc: Optional[int],
    ) -> Dataset:
        return self.dataset.filter(func, num_proc=num_proc)

class RayDataset(Dataset):
    """A wrapper of Ray dataset.
    """
    def __init__(self, dataset: RD) -> None:
        self.dataset = dataset

    def filter(
        self,
        func: Callable, 
        num_proc: Optional[int],
    ) -> Dataset:
        self.dataset.filter(func, num_cpus=num_proc)

    def map(
        self,
        func: Callable,
        num_proc: Optional[int],
        batched: bool = False,
    ) -> Dataset:
        if batched:
            return self.dataset.map_batches(func, num_cpus=num_proc)
        else:
            return self.dataset.map(func, num_cpus=num_proc)