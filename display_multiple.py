import matplotlib.pyplot as plt
import numpy as np
import os, sys
from pprint import pprint as pp
# Paths will be passed in from the command line
# Get all paths to the files and validate them

def get_paths(path_to_file_containig_paths: str):
    if not os.path.exists(path_to_file_containig_paths):
        raise ValueError(f"Path {path_to_file_containig_paths} does not exist")
    
    with open(path_to_file_containig_paths, "r") as f:
        paths = f.readlines()
    
    paths = list(map(lambda x: x.replace("\n", ""), paths))
    paths = list(filter(lambda x: os.path.exists(x), paths))

    return paths


if __name__ == "__main__":
    paths = get_paths(sys.argv[1])
    pp(paths)