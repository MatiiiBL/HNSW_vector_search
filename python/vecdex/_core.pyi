# Type stubs for the C++ extension module; the docstrings live in python/bindings.cpp.
import os
from typing import Literal

import numpy as np
import numpy.typing as npt

__version__: str

class Index:
    def __init__(
        self,
        space: Literal["l2", "ip", "cosine"],
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        seed: int = 42,
    ) -> None: ...
    def add_items(
        self, data: npt.NDArray[np.float32], *, num_threads: int = 1
    ) -> npt.NDArray[np.int64]: ...
    def knn_query(
        self,
        data: npt.NDArray[np.float32],
        k: int = 1,
        ef: int | None = None,
        *,
        num_threads: int = 1,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]: ...
    def save(self, path: str | os.PathLike[str]) -> None: ...
    @staticmethod
    def load(path: str | os.PathLike[str]) -> Index: ...
    @property
    def ef(self) -> int: ...
    @ef.setter
    def ef(self, value: int) -> None: ...
    def set_ef(self, ef: int) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def space(self) -> Literal["l2", "ip", "cosine"]: ...
    @property
    def M(self) -> int: ...
    @property
    def ef_construction(self) -> int: ...
    def memory_bytes(self) -> int: ...
    def __len__(self) -> int: ...
