import py_cli_interaction
from rich.console import Console
import os.path as osp
import datetime
import json
import time
import multiprocessing as mp
import threading

from rich.progress import track

from markit_processing.client import RealsenseRecorderClient, TcpBrokerClient

from markit_processing.client.realsense_recorder.api.default import status_v1_status_get as realsense_recorder_status
from markit_processing.client.realsense_recorder.api.default import stop_process_v1_stop_post as realsense_recorder_stop
from markit_processing.client.realsense_recorder.api.default import start_process_v1_start_post as realsense_recorder_start
from markit_processing.client.realsense_recorder.api.default import ready_v1_ready_get as realsense_recorder_ready

from markit_processing.client.tcpbroker.api.default import status_v1_status_get as tcpbroker_status
from markit_processing.client.tcpbroker.api.default import stop_process_v1_stop_post as tcpbroker_stop
from markit_processing.client.tcpbroker.api.default import start_process_v1_start_post as tcpbroker_start
from markit_processing.client.tcpbroker.api.default import imu_control_v1_imu_control_post as tcpbroker_control
from markit_processing.client.tcpbroker.api.default import imu_connection_v1_imu_connection_get as tcpbroker_connection

CONFIG_REALSENSE_ENDPOINT = "http://127.0.0.1:5050"
CONFIG_TCPBROKER_ENDPOINT = "http://127.0.0.1:18889"


def main():
    console = Console()
    console.log(f"creating new recording")
    experiment_tag, err = py_cli_interaction.parse_cli_string("experiment_tag")
    if err is not None:
        experiment_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    tag_for_realsense_recorder = osp.join(experiment_tag, "realsense")
    tag_for_tcpbroker = osp.join(experiment_tag, "imu")

    realsense_recorder_client = RealsenseRecorderClient(base_url=CONFIG_REALSENSE_ENDPOINT, timeout=5, verify_ssl=False)
    tcpbroker_client = TcpBrokerClient(base_url=CONFIG_TCPBROKER_ENDPOINT, timeout=5, verify_ssl=False)

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
            console.log(err)
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
            console.log(f"Number of connected IMU: {conn['count']}")
            time.sleep(10)
            if ev.is_set():
                break

    event = mp.Event()
    event.clear()
    check_thread = threading.Thread(target=check_connection, args=(event,))
    check_thread.start()

    try:
        console.input("[green] \nYou can now move!!!, Press \\[enter] to stop \n")
    except KeyboardInterrupt:
        pass

    console.print("[cyan]Stopping recording[/cyan]")
    event.set()
    realsense_recorder_stop.sync_detailed(client=realsense_recorder_client)
    tcpbroker_stop.sync_detailed(client=tcpbroker_client)
    check_thread.join(timeout=10)

    for _ in track(range(20), description="Waiting for processes to clean up"):
        time.sleep(0.2)


if __name__ == '__main__':
    main()
    print("Done!")
