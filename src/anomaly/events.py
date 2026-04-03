import numpy as np

def group_crashes(indices, min_gap=20):
    indices = list(indices)
    if len(indices) == 0:
        return []
    events = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] > min_gap:
            end = indices[i - 1]
            events.append((start, end))
            start = indices[i]
    events.append((start, indices[-1]))
    return events