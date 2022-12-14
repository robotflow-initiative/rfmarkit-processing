import argparse
import datetime
import logging
import multiprocessing as mp
import os
import select
import signal
from json.decoder import JSONDecodeError
from typing import Dict, List
import json

from hipnuc import hipnuc_module

logging.basicConfig(level=logging.INFO)

MAX_LISTEN = 8
POLL_READ = select.POLLIN | select.POLLPRI | select.POLLHUP | select.POLLERR


def record_csv(measurement_filename: str, config: Dict):
    """Start recording. This function is the core.

    Args:
        measurement_filename (str): Filename of CSV formatted recording
        config: IMU configuration
                e.g.:
                {
                    "port": /dev/cu.usbserial-124440,
                    "baudrate": 921600,
                    "report_datatype": {
                        "imusol": True,
                        "gwsol": False,
                        "id": True,
                        "acc": False,
                        "gyr": False,
                        "mag": False,
                        "euler": False,
                        "quat": False,
                        "imusol_raw": True,
                        "gwsol_raw": True
                    }
                } 

    """
    IMU: hipnuc_module = hipnuc_module(config)
    logging.info(f"Started measurement at: {datetime.datetime.utcnow().timestamp()}")
    logging.info("Press Ctrl-C to terminate.")
    logging.info(f'Saving measurement to {measurement_filename}')

    try:
        # Create csv file
        IMU.create_csv(measurement_filename)

        with open(measurement_filename, 'a') as f:
            while True:
                try:
                    data = IMU.get_module_data(10)
                    #write to file as csv format, only work for 0x91, 0x62(IMUSOL), or there will be error
                    IMU.write2csv_handle(data, f)

                except Exception as err:
                    IMU.close()
                    logging.error(f"Error: {err}")
                    break

    except KeyboardInterrupt:
        IMU.close()
        logging.warn("Stopping measurement on KeyboardInterrupt")
        pass

    logging.info("Recording is terminated.")


class MeasurementServer(object):
    READ_LEN: int = 1024
    POLL_INTERVAL_MS: int = 1000

    def __init__(self, pipe_in, pipe_out, config: Dict) -> None:
        super().__init__()
        self.pipe_in = pipe_in
        self.pipe_out = pipe_out
        self.config = config

        self.make_pipe(pipe_in)
        self.make_pipe(pipe_out)

        self.rf: int = os.open(pipe_in, os.O_RDONLY | os.O_NONBLOCK)
        self.wf: int = os.open(pipe_out, os.O_SYNC | os.O_CREAT | os.O_RDWR)

        self.poller: select.poll = select.poll()
        self.poller.register(self.rf, POLL_READ)

        self.processes: List[mp.Process] = []

    def __del__(self):
        # Terminate subprocess
        if 'processes' in self.__dir__():
            for proc in self.processes:
                proc.terminate()
                proc.join()

        # Close file descriptor
        if list(set(['rf', 'wf', 'pipe_in',
                     'pipe_out']).intersection(set(self.__dir__()))) == ['rf', 'wf', 'pipe_in', 'pipe_out']:
            os.close(self.rf)
            os.close(self.wf)
            os.unlink(self.pipe_in)
            os.unlink(self.pipe_out)

    def make_pipe(self, pipe: str):
        # Remove the pipe if exist
        if os.path.exists(pipe):
            os.remove(pipe)
        os.mkfifo(pipe)

    def make_response(self, status: str, **kwargs):
        resp = {'status': status}
        resp.update(**kwargs)
        resp['timestamp'] = datetime.datetime.utcnow().timestamp()

        return os.write(self.wf, bytes(json.dumps(resp) + '\n', encoding='utf-8'))

    def join_all_subprocess(self):
        for proc in self.processes:
            if proc.is_alive():
                os.kill(proc.pid, signal.SIGINT)
            proc.join()
            del proc

    def exec(self, cmd: Dict):
        """Execute command

        Args:
            cmd (Dict): Command in JSON format
        """
        logging.debug(cmd)
        logging.debug(f"Received command at: {datetime.datetime.utcnow().timestamp()}")

        if 'start' in cmd['type']:
            if len(self.processes) == 0:
                measurement_filename = cmd['measurement_filename'] if 'measurement_filename' in cmd.keys(
                ) else 'imu_mem_' + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S.csv") # TODO: Send measurement name via PIPE api

                proc = mp.Process(None, record_csv, record_csv.__name__, (measurement_filename, self.config, ))
                proc.start()
                self.processes.append(proc)

                self.make_response(status="OK")
            else:
                logging.error("A process is already running")
                self.make_response(status="FAIL", msg="A process is already running")

        elif 'stop' in cmd['type']:
            self.join_all_subprocess()
            self.make_response(status="OK")

        elif 'quit' in cmd['type']:
            self.join_all_subprocess()
            raise KeyboardInterrupt

    def serve(self):
        """Serving Measurement from local Unix pipe
        """
        try:
            while True:
                # Poll
                epoll_list = self.poller.poll(self.POLL_INTERVAL_MS)
                for fd, events in epoll_list:
                    if events & (select.POLLIN | select.POLLPRI) and fd is self.rf:
                        while True:
                            data: bytes = os.read(self.rf, self.READ_LEN)
                            if len(data) <= 0:
                                break

                            try:
                                cmd: Dict = json.loads(str(data, 'utf-8'))
                                logging.info(f"Got incoming command{cmd}")
                                self.exec(cmd)
                            except JSONDecodeError:
                                logging.warn(f"Invalid command format:\n{data}")

                        logging.info(f"Client closed connection\n")
                    else:
                        logging.warn(f"Unrecognized event: {events}")
                pass
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, default="chlog.csv")  # Measurement file, effective only in single shot mode
    parser.add_argument('--port', '-p', type=str, default="/dev/cu.usbserial-124440")  # Serial port of IMU
    parser.add_argument('-l', '--listen', action="store_true")
    parser.add_argument('--pipe_in', type=str, default="/tmp/measurement_pipe_in")  # Named pipe for receiving command
    parser.add_argument('--pipe_out', type=str, default="/tmp/measurement_pipe_out")  # Named pipe for sending response
    args: argparse.Namespace = parser.parse_args()

    CONFIG = {
        "port": args.port,
        "baudrate": 921600,
        "report_datatype": {
            "imusol": True,
            "gwsol": False,
            "id": True,
            "acc": False,
            "gyr": False,
            "mag": False,
            "euler": False,
            "quat": False,
            "imusol_raw": True,
            "gwsol_raw": True
        }
    }  # Default Configuration of IMU
    # TODO: Add support for multi-IMU configuration
    # e.g.: Send IMU config via PIPE api

    # When -l is detected, the script will launch a server
    if args.listen:
        MS = MeasurementServer(args.pipe_in, args.pipe_out, CONFIG)
        MS.serve()
    else:
        # The script start a single recording
        record_csv(args.file)

    # >>> python measure.py -l

    # Debug garbage collection
    # gc.set_debug(gc.DEBUG_LEAK)
    # del MS
    # gc.collect()
