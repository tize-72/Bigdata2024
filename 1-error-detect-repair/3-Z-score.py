"""Z-score代码"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def z_score(data):
    for idx, item in enumerate(data):
        z_score_item = (item - np.mean(data)) / np.std(data)
        z_score_item = np.abs(z_score_item)
        if z_score_item >= 2.5:
            print(item)

data = [1,4,8,90,98,44,35,56,2,41,11,24,23,45,500, 150]
z_score(data)