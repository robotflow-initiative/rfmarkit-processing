import argparse
import os
import os.path as osp

import yaml
from rich.console import Console

from .model import DefaultExperiment


def main(args: argparse.Namespace):
    console = Console()
    new_experiment = DefaultExperiment(name=args.name)

    if not osp.exists(args.output):
        os.makedirs(args.output)

    experiment_path = osp.join(args.output, args.name)
    if osp.exists(experiment_path):
        console.print(f"Experiment {args.name} already exists in {args.output}", style="red")
        exit(1)
    else:
        os.mkdir(experiment_path)

    with open(osp.join(experiment_path, "index.yaml"), "w") as f:
        yaml.dump({'articulated': new_experiment.to_dict()}, f, sort_keys=False)
        console.print(f"Experiment {args.name} created in {args.output}", style="green")

    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", type=str, default="default", help="name of the experiment")
    parser.add_argument("--output", "-o", type=str, default="./experiments", help="path to output experiment configuration bundle (directory is expected)")
    args = parser.parse_args()
    main(args)
