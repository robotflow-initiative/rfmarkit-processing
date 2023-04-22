import argparse
import os.path as osp

import yaml
from markit_processing.cmd.model import DefaultExperiment, DefaultTarget
from rich.console import Console


def main(args: argparse.Namespace):
    console = Console()

    if args.csv is None or not osp.exists(args.csv):
        console.print(f"CSV file {args.csv} does not exist", style="red")
        exit(1)

    if args.experiment is None or not osp.exists(args.experiment):
        console.print(f"Experiment {args.experiment} does not exist", style="red")
        exit(1)
    else:
        if not osp.exists(osp.join(args.experiment, "index.yaml")):
            console.print(f"Experiment {args.experiment} does not have index.yaml", style="red")
            exit(1)

    new_experiment = DefaultExperiment()
    index_path = osp.join(args.experiment, "index.yaml")
    with open(index_path) as f:
        new_experiment.from_dict(yaml.load(f, Loader=yaml.SafeLoader)['articulated'])

    csv_paths = args.csv.split(",")
    if any(map(lambda x: not osp.exists(x), csv_paths)):
        console.print(f"Not all CSV file(s) {args.csv} exist", style="red")
        exit(1)

    tags = []
    for csv_path in csv_paths:
        with open(csv_path) as f:
            for line in f.readlines():
                tags.append(line.strip().split(",")[0])

    console.print(f"Found {len(tags)} tags in CSV file(s) {args.csv}", style="green")

    if not args.keep:
        new_experiment.targets = []

    for tag in tags:
        new_target = DefaultTarget(tag)
        new_experiment.targets.append(new_target.to_dict())

    with open(index_path, "w") as f:
        yaml.dump({'articulated': new_experiment.to_dict()}, f, sort_keys=False)
        console.print(f"Experiment {args.experiment} patched with {len(tags)} targets", style="green")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="path to csv file(s) that include the experiment tag(which is expected to be the first column)")
    parser.add_argument("--experiment", "-i", type=str, default=None, help="path to input experiment(directory is expected)")
    parser.add_argument("--keep", action="store_true", help="keep existing targets in the experiment")
    args = parser.parse_args()
    main(args)
