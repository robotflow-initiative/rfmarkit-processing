import argparse
import datetime
import json
import logging
import os

import requests
from tcpbroker.applications.control import broadcast_command
from tcpbroker.config import BrokerConfig

logging.basicConfig(level=logging.INFO)
DEFAULT_CTX = {
    "realsense": ["http://10.233.234.1:5050", "http://10.233.234.1:5051", "http://10.233.234.1:5052", "http://10.233.234.1:5053"],
    "imu": ["http://10.233.234.1:18889"]
}

CTX = None
TAG = None
SUBNET = None  # TODO Merge these into CTX
IMU_PORT = 18888  # TODO Merge these into CTX


def _start(endpoint="http://127.0.0.1:5000", tag: str = None):
    data = {
        'tag': tag if (tag is not None or tag == "") else str(datetime.datetime.utcnow().timestamp())
    }
    headers = {'Content-Type': 'application/json'}
    resp = requests.get(url=endpoint + "/start",
                        headers=headers, data=json.dumps(data), timeout=20)
    print(resp.json())
    return resp.json()


def _stop(endpoint="http://127.0.0.1:5000"):
    resp = requests.get(endpoint + "/stop", timeout=2)
    print(resp.json())
    return resp.json()


def open_experiment(ctx):
    try:
        global TAG
        TAG = input("输入一个实验标签，按下回车后等待相机和传感器启动\n> ")
        if TAG == "":
            TAG = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        endpoint_results = []
        for imu_endpoint in ctx['imu']:
            endpoint_results.append(_start(imu_endpoint, tag=TAG))
        for realsense_endpoint in ctx["realsense"]:
            endpoint_results.append(_start(realsense_endpoint, tag=TAG))

        if len(list(filter(lambda x: 'START OK' in x['msg'], endpoint_results))) == (len(ctx['imu']) + len(ctx['realsense'])):
            print("正常")
        else:
            print("异常")
        imu_results = broadcast_command(SUBNET, IMU_PORT, "start", None)
        print(f"共有{len(imu_results)}个传感器在线")

    except Exception as e:
        print(e)
        logging.warning("Failed to open")


def close_experiment(ctx):
    try:
        broadcast_command(SUBNET, IMU_PORT, "stop", None)
        for realsense_endpoint in ctx["realsense"]:
            _stop(realsense_endpoint)
        for imu_endpoint in ctx['imu']:
            _stop(imu_endpoint)
    except Exception as e:
        print(e)
        logging.warning("Failed to close")


def run_healthcheck(ctx):
    endpoint_results = []
    try:
        for realsense_endpoint in ctx["realsense"]:
            endpoint_results.append(requests.get(realsense_endpoint + "/", timeout=2).json())
        for imu_endpoint in ctx['imu']:
            endpoint_results.append(requests.get(imu_endpoint + "/", timeout=2).json())
        if len(list(filter(lambda x: x['status'] == 200, endpoint_results))) == (len(ctx['imu']) + len(ctx['realsense'])):
            print("正常")
        else:
            print("异常")
    except Exception as e:
        logging.warning(e)
        logging.warning("异常")


def print_help():
    print("\
    > 1. 开始记录\n\
    > 2. 强制停止记录\n\
    > 3. 健康检查\n\
    > 4. 退出\n\n")


def parse_config(path: str):
    global CTX
    if os.path.exists(path):
        try:
            tmp = json.load(open(path, 'r'))
            assert 'endpoints' in tmp.keys()
            assert 'realsense' in tmp['endpoints'].keys()
            assert 'imu' in tmp['endpoints'].keys()
            CTX = tmp['endpoints']
        except:
            pass
    logging.info(CTX)


def main(args):
    global SUBNET
    config = BrokerConfig(args.config)
    SUBNET = list(map(lambda x: int(x), config.DEFAULT_SUBNET.split("."))) if config.DEFAULT_SUBNET is not None else None
    ctx = DEFAULT_CTX
    # Moify CTX
    parse_config(args.config)

    # Wait for command
    while True:
        try:
            print_help()
            cmd = input("> ")
            try:
                cmd = int(cmd)
            except ValueError:
                print("\n错误的输入\n")
                continue
            if cmd == 1:
                open_experiment(ctx)
                print("\n------ 现在可以开始做动作了!! ------\n")
                input("再次按下回车键停止录制：")
                print("停止录制")
                close_experiment(ctx)
            elif cmd == 2:
                close_experiment(ctx)
            elif cmd == 3:
                close_experiment(ctx)
                run_healthcheck(ctx)
            elif cmd == 4:
                print(f"Exiting....")
                exit(0)
            else:
                print("\n错误的输入\n")
        except KeyboardInterrupt:
            logging.info("Exiting....")
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.json')
    args = parser.parse_args()
    main(args)
