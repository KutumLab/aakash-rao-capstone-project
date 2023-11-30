import os
import numpy
import pandas
import matplotlib.pyplot as plt
import argparse



def plot(src_path, phase):
    if not os.path.exists(src_path):
        print("File not found: ", src_path)
        raise FileNotFoundError(src_path)
    elif len(os.listdir(src_path)) == 0:
        print("Empty folder: ", src_path)
        raise FileNotFoundError(src_path)
    else:
        for folder in os.listdir(src_path):
            outpath = os.path.join(src_path, folder)
            results = pandas.read_csv(os.path.join(outpath, "results.csv"))
            plot_save_path = os.path.join(outpath, "plot")
            if not os.path.exists(plot_save_path):
                os.makedirs(plot_save_path)
            if phase == "testing":
                print(results)
                return

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="results", help="path to the results folder")
    parser.add_argument("--phase", type=str, default="testing", help="phase to plot")
    args = parser.parse_args()
    plot(args.src_path, args.phase)
