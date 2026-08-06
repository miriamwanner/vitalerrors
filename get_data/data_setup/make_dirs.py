"""Create the {dataset}/{subset}/{gen_type} directory tree under a root data dir.

Example:
    python -m get_data.data_setup.make_dirs --root_dir data
"""
import argparse
import os

from .data_info import datasets, subsets, gen_type


def main():
    parser = argparse.ArgumentParser(description="Create the dataset directory tree.")
    parser.add_argument("--root_dir", type=str, default="data")
    args = parser.parse_args()

    for d in datasets:
        for s in subsets[d]:
            for sub in gen_type:
                temp_path = os.path.join(args.root_dir, d, s, sub)
                os.makedirs(temp_path, exist_ok=True)


if __name__ == "__main__":
    main()
