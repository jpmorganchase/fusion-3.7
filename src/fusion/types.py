from queue import Queue
from typing import Any, List, Tuple, Union

WorkerQueueT = Queue[Tuple[int, int, int]]
PyArrowFilterT = Union[List[Tuple[Any]], List[List[Tuple[Any]]]]