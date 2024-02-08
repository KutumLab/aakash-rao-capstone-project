import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def clean(model_name, inpath, outpath):
    print (model_name)
    print (inpath)
    print (outpath)
    pass

if __name__ == '__main__':
    argparseer = argparse.ArgumentParser()
    argparseer.add_argument('--inpath', type=str, default='../outputs/detectron')
    argparseer.add_argument('--model_name', type=str, default='../data/plot.csv')
    argparseer.add_argument('--output_path', type=str, default='../data/plot.png')
    args = argparseer.parse_args()
    clean(args.model_name, args.inpath, args.output_path)