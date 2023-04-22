import argparse
import cmd
import json
import logging
import multiprocessing as mp
import os
import os.path as osp
import shutil
import threading
import time

import cv2
import yaml
from markit_processing.client import RealsenseRecorderClient, TcpBrokerClient
from markit_processing.cmd.model import DefaultExperiment
from py_cli_interaction import must_parse_cli_sel
from rich.console import Console
from rich.progress import track
from rich.table import Table


class ExperimentConsole(cmd.Cmd):
    intro = "Welcome to the Articulated Kit Experiment Launcher.   Type help or ? to list commands.\n"
    current_calibration_profile = None
    current_experiment_tag = None
    current_experiment_tag_sel: int = None

    @property
    def prompt(self):
        return f"(experiment [{self.current_calibration_profile.split('_')[1] if self.current_calibration_profile is not None else None}] / {self.current_experiment_tag}) "

    def __init__(self, args: argparse.Namespace):
        super().__init__()

        if os.path.exists(args.experiment):
            # metadata files exist, load them
            self.experiment_index_path = args.experiment
            self.experiment_index_filename = osp.join(args.experiment, "index.yaml")
            self.realsense_config_filename = args.realsense_config
            self.imu_config_filename = args.imu_config

            # load index.yaml
            self.index_yaml_dict = yaml.load(open(self.experiment_index_filename), Loader=yaml.SafeLoader)
            self.experiment = DefaultExperiment()
            self.experiment.from_dict(self.index_yaml_dict['articulated'])

            # set logs
            logging.basicConfig(level=logging.DEBUG) if self.experiment.debug else logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger('articulated')
            logger.setLevel(logging.DEBUG) if self.experiment.debug else logger.setLevel(logging.INFO)

            self.console = Console()

            self.console.log(f"using {self.experiment.base_dir} as storage backend")
            if not osp.exists(self.experiment.base_dir):
                self.console.log(f"creating {self.experiment.base_dir}")
                os.mkdir(self.experiment.base_dir)

            self.run_dir = osp.join(self.experiment_index_path, "runs")
            if not osp.exists(self.run_dir):
                os.mkdir(self.run_dir)

            calibration_candidates, calibration_candidate_idx = self._get_existing_calibrations()
            self.current_calibration_profile = calibration_candidates[-1] if len(calibration_candidates) > 0 else None
            try:
                self.current_experiment_tag_sel = [x['finished'] == False for x in self.experiment.targets].index(True)
            except ValueError as _ :
                self.current_experiment_tag_sel = len(self.experiment.targets) -1
            self.current_experiment_tag = self.experiment.targets[self.current_experiment_tag_sel]['tag']

        else:
            logging.error(f"Config file {args.experiment} not found")
            exit(1)

    def _get_existing_calibrations(self):
        candidates = list(filter(lambda x: x.startswith("_cali-") and osp.isdir(osp.join(self.experiment.base_dir, x)), os.listdir(self.experiment.base_dir)))
        candidate_idx = list(map(lambda x: int(x.split("-")[1]), candidates))
        return candidates, candidate_idx

    def _flush_experiment(self):
        self.index_yaml_dict['articulated'] = self.experiment.to_dict()
        with open(self.experiment_index_filename, 'w') as f:
            yaml.dump(self.index_yaml_dict, f)
        pass

    def do_imu(self, arg):
        from tcpbroker.scripts.cli import main as entry_point
        _arg = argparse.Namespace()
        _arg.config = self.imu_config_filename
        entry_point(_arg)

    def do_list_target(self, arg):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("INDEX")
        table.add_column("TAG")
        table.add_column("UUID")
        table.add_column("Finished")
        table.add_column("Recordings")

        for idx, target in enumerate(self.experiment.targets):
            table.add_row(
                f"[cyan]{str(idx + 1)}", f"{target['tag']}", f"{target['uuid']}", "[red]FALSE" if not target['finished'] else "[green]TRUE", f"{target['recordings']}"
            )

        self.console.print(table)

    def do_ls(self, arg):
        return self.do_list_target(arg)

    def do_show(self, arg):
        raise NotImplementedError

    def do_select_calibration(self, arg):
        candidates, candidate_idx = self._get_existing_calibrations()
        if len(candidates) > 0:
            sel = must_parse_cli_sel("Select calibration", candidates, min=1)
            self.current_calibration_profile = candidates[sel - 1]
        else:
            self.console.log("No calibration profile found")

    def do_select_target(self, arg):
        self.do_list_target(None)
        sel = must_parse_cli_sel("Select target", [], min=1, max=len(self.experiment.targets))
        self.current_experiment_tag = self.experiment.targets[sel - 1]['tag']
        self.current_experiment_tag_sel = sel - 1

    def do_next(self, arg):
        if self.current_experiment_tag_sel < len(self.experiment.targets):
            self.current_experiment_tag_sel += 1
            self.current_experiment_tag = self.experiment.targets[self.current_experiment_tag_sel]['tag']

    def do_new(self, arg):
        if self.current_calibration_profile is None:
            self.console.log("No calibration profile selected")
            return
        if self.current_experiment_tag is None:
            self.console.log("No target selected")
            return

        experiment_tag = self.current_experiment_tag
        try:
            self.console.log(f"showing thumbnail for {experiment_tag}, check opencv window")
            mat = cv2.imread(osp.join(self.experiment_index_path, "thumbnails", f"{experiment_tag}.jpg"))
            cv2.imshow("Press any key to continue", mat[::4, ::4, :])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as _:
            self.console.log(f"Thumbnail for {experiment_tag} not found")

        collision = list(filter(lambda x: x.startswith(experiment_tag) and osp.isdir(osp.join(self.experiment.base_dir, x)), os.listdir(self.experiment.base_dir)))
        collision.extend(self.experiment.targets[self.current_experiment_tag_sel]['recordings'])
        collision_idx = list(map(lambda x: int(x.split("-")[2]), collision))
        collision_idx.extend(map(lambda x: int(x.split("-")[2]), self.experiment.targets[self.current_experiment_tag_sel]['recordings']))
        if len(collision) > 0:
            self.console.log(f"Collision detected, renaming to {experiment_tag}-{max(collision_idx) + 1}")
            experiment_tag = f"{experiment_tag}-{max(collision_idx) + 1}"
        else:
            experiment_tag = f"{experiment_tag}-1"

        self.console.log(f"Creating new recording {experiment_tag} for self.current_experiment_tag")

        tag_for_realsense_recorder = osp.join(experiment_tag, "realsense")
        tag_for_tcpbroker = osp.join(experiment_tag, "imu")

        realsense_recorder_client = RealsenseRecorderClient(base_url=self.experiment.realsense_recorder_endpoint, timeout=5, verify_ssl=False)
        tcpbroker_client = TcpBrokerClient(base_url=self.experiment.tcpbroker_endpoint, timeout=5, verify_ssl=False)

        from markit_processing.client.realsense_recorder.api.default import status_v1_status_get as realsense_recorder_status
        from markit_processing.client.realsense_recorder.api.default import stop_process_v1_stop_post as realsense_recorder_stop
        from markit_processing.client.realsense_recorder.api.default import start_process_v1_start_post as realsense_recorder_start
        from markit_processing.client.realsense_recorder.api.default import ready_v1_ready_get as realsense_recorder_ready

        from markit_processing.client.tcpbroker.api.default import status_v1_status_get as tcpbroker_status
        from markit_processing.client.tcpbroker.api.default import stop_process_v1_stop_post as tcpbroker_stop
        from markit_processing.client.tcpbroker.api.default import start_process_v1_start_post as tcpbroker_start
        from markit_processing.client.tcpbroker.api.default import imu_control_v1_imu_control_post as tcpbroker_control
        from markit_processing.client.tcpbroker.api.default import imu_connection_v1_imu_connection_get as tcpbroker_connection

        while True:
            try:
                ret = json.loads(realsense_recorder_status.sync_detailed(client=realsense_recorder_client).content)
                if ret['active_processes'] <= 0:
                    break
                else:
                    realsense_recorder_stop.sync_detailed(client=realsense_recorder_client)
                    time.sleep(2)
                print(".", end="")
            except Exception as err:
                pass

        while True:
            try:
                ret = json.loads(tcpbroker_status.sync_detailed(client=realsense_recorder_client).content)
                if ret['active_processes'] <= 0:
                    break
                else:
                    tcpbroker_stop.sync_detailed(client=realsense_recorder_client)
                    time.sleep(2)
                print(".", end="")
            except Exception as err:
                self.console.log(err)
                pass

        tcpbroker_start.sync_detailed(client=tcpbroker_client, tag=tag_for_tcpbroker)
        [time.sleep(0.1) for _ in track(range(20), description="Waiting for tcpbroker to start")]
        tcpbroker_control.sync_detailed(client=tcpbroker_client, command="start")

        realsense_recorder_start.sync_detailed(client=realsense_recorder_client, tag=tag_for_realsense_recorder)
        for _ in track(range(5), description="Waiting for realsense recorder to start"):
            ret = json.loads(realsense_recorder_ready.sync_detailed(client=realsense_recorder_client).content)
            if ret['code'] == 200 and ret['ready']:
                break
            else:
                time.sleep(1)

        def check_connection(ev):
            while True:
                conn = json.loads(tcpbroker_connection.sync_detailed(client=tcpbroker_client).content)
                self.console.log(f"Number of connected IMU: {conn['count']}")
                time.sleep(10)
                if ev.is_set():
                    break

        event = mp.Event()
        event.clear()
        check_thread = threading.Thread(target=check_connection, args=(event,))
        check_thread.start()

        try:
            self.console.input("[green] \nYou can now move!!!, Press \\[enter] to stop \n")
        except KeyboardInterrupt:
            pass

        self.console.print("[cyan]Stopping recording[/cyan]")
        event.set()
        realsense_recorder_stop.sync_detailed(client=realsense_recorder_client)
        tcpbroker_stop.sync_detailed(client=tcpbroker_client)
        check_thread.join(timeout=10)

        for _ in track(range(20), description="Waiting for processes to clean up"):
            time.sleep(0.2)

        if os.path.exists(osp.join(self.experiment.base_dir, experiment_tag, "realsense", "metadata_all.json")):
            self.console.log(f"Recording {experiment_tag} completed")
            self.experiment.targets[self.current_experiment_tag_sel]['finished'] = True
            self.experiment.targets[self.current_experiment_tag_sel]['recordings'].append(experiment_tag)
            self.experiment.targets[self.current_experiment_tag_sel]['associated_calibration'] = self.current_calibration_profile
            self._flush_experiment()

    def do_quit(self, arg):
        """exit - exit program"""
        self.console.log("Thank you, bye!")
        self._flush_experiment()
        exit(0)

    def emptyline(self) -> bool:
        experiment_tag = self.current_experiment_tag
        try:
            self.console.log(f"showing thumbnail for {experiment_tag}, check opencv window")
            mat = cv2.imread(osp.join(self.experiment_index_path, "thumbnails", f"{experiment_tag}.jpg"))
            cv2.imshow("Press any key to continue", mat[::4, ::4, :])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return False
        except Exception as _:
            self.console.log(f"Thumbnail for {experiment_tag} not found")
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='./experiments/default', help="path to experiment definition")
    parser.add_argument('--realsense_config', type=str, default='./realsense_config.yaml', help="path to realsense config file")
    parser.add_argument('--imu_config', type=str, default='./imu_config.yaml', help="path to imu config file")

    console = ExperimentConsole(parser.parse_args())
    console.cmdloop()
