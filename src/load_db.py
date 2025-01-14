import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time


def decorator_timer(some_function):
    def wrapper(*args, **kwargs):
        t1 = time()
        result = some_function(*args, **kwargs)
        print(f"Execution time: {time() - t1:.6f} seconds")
        return result

    return wrapper

def get_selected_data(self):
    data_dict= None

    return data_dict
