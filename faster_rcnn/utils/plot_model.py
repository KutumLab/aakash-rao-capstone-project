import json
import argparse
import os
import numpy as np
import pandas as pd


def json_gen(path):
    # read json file
    info = pd.read_csv(os.path.join(path, 'metrics.csv'), index_col=False, header=0)
    info = info.sort_values(by=['iteration'])
    # remove nas
    info = info.dropna(axis=1)
     print(info.head()['iteration'])



    