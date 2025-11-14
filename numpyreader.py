import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    path = args.path

    data = np.load(path)
    print(data)

if __name__ == "__main__":
    main()