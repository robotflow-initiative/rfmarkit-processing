import argparse
import datetime
import os
import os.path as osp
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor

import tqdm

PACKAGES = ('calibrations', 'metadata')
DATA = ('data/immobile', 'data/portable')


def worker(input_path, output_path):
    print(f"Compressing {input_path} to {output_path}")
    tar = tarfile.open(output_path, "w")
    tar.add(input_path, arcname=osp.basename(input_path))
    tar.close()


def main(args: argparse.Namespace):
    if not osp.exists(args.input):
        raise ValueError(f"Input directory {args.input} does not exist")
    if not osp.exists(args.output):
        os.makedirs(args.output)
        if not osp.exists(args.output):
            raise ValueError(f"Output directory {args.output} does not exist")
    output_name = osp.basename(args.input) + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = osp.join(args.output, output_name)
    os.makedirs(output_path)
    if not osp.exists(output_path):
        raise ValueError(f"Output directory {output_path} does not exist")

    for name in PACKAGES:
        shutil.copytree(osp.join(args.input, name), osp.join(output_path, name))

    for name in DATA:
        print(f"Processing {name}")
        os.makedirs(osp.join(output_path, name))
        if not osp.exists(osp.join(output_path, name)):
            raise ValueError(f"Output directory {osp.join(output_path, name)} does not exist")
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = []
            with tqdm.tqdm(total=len(os.listdir(osp.join(args.input, name)))) as pbar:
                for target in os.listdir(osp.join(args.input, name)):
                    pbar.update(1)
                    pbar.set_description(f"Submitting {target}")
                    results.append(executor.submit(worker, osp.join(args.input, name, target), osp.join(output_path, name, target + ".tar")))
                    # worker(osp.join(args.input, name, target), osp.join(output_path, name, target + ".tar"))
            [result.result() for result in results]


def entrypoint(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input directory")
    parser.add_argument("--output", type=str, help="Output directory")
    args = parser.parse_args(argv)
    main(args)


if __name__ == '__main__':
    import sys

    entrypoint(sys.argv[1:])
