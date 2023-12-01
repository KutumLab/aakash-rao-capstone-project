import json
import argparse


def json_gen(path):
    # read json file
    json_data = {}
    with open(path, 'r') as f:
        json_data = json.load(f)
    print(json_data)

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--path', help='path to json file')
    args = argparse.parse_args()
    json_gen(args.path)


    