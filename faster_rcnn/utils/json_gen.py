import json
import argparse
import os
import numpy as np
import pandas as pd


def json_gen(path):
    for folder in os.listdir(path):
        if 'faster' not in folder:
            continue
        fold_path = os.path.join(path, folder)
        # read json file
        json_data = []
        with open(os.path.join(fold_path, 'metrics.json'), 'r') as f:
            for line in f:
                json_data.append(json.loads(line))
        keys = np.array([])
        for item in json_data:
            items = list(item.keys())
            items = np.array(items)
            keys = np.concatenate((keys, items))
        keys = np.unique(keys)

        df  = pd.DataFrame(columns=keys)
        for item in json_data:
            df = pd.concat([df, pd.DataFrame(item, index=[0])], ignore_index=True)
        # sorting df to iteration
        df = df.sort_values(by=['iteration'])
        print(df.head()['iteration'])
        # save to csv
        df.to_csv(os.path.join(fold_path, 'metrics.csv'), index=False)


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--path', help='path to json file')
    args = argparse.parse_args()
    json_gen(args.path)


    