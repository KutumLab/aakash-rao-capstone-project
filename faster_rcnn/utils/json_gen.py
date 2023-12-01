import json
import argparse
import os


def json_gen(path):
    # read json file
    json_data = []
    with open(os.path.join(path, 'metrics.json'), 'r') as f:
        for line in f:
            json_data.append(json.loads(line))
    print(json_data)

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--path', help='path to json file')
    args = argparse.parse_args()
    json_gen(args.path)


    