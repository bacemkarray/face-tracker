from collections import deque
import numpy as np

class SlidingAverage:
    def __init__(self, window_size=1):
        self.values = deque(maxlen=window_size)

    def update(self, value):
        self.values.append(value)
        return np.mean(self.values)

    def reset(self):
        self.values.clear()