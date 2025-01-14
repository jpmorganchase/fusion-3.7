from queue import Queue
from typing import Tuple

WorkerQueueT = Queue[Tuple[int, int, int]]