import uuid
from dataclasses import dataclass
from typing import Dict, List, Any

import yaml


class DefaultExperiment:
    name: str = 'default'
    realsense_recorder_endpoint: str = None
    tcpbroker_endpoint: str = None
    imu_friendly_name_mapping: Dict[str, str] = None
    realsense_friendly_name_mapping: Dict[str, str] = None
    base_dir: str = None
    debug: bool = False
    targets: List[Dict[str, Any]] = None

    def __init__(self, name: str = 'default', path_to_yaml_file: str = None):
        self.name = name
        if path_to_yaml_file is not None:
            with open(path_to_yaml_file) as f:
                base_cfg_dict = yaml.load(f, Loader=yaml.SafeLoader)
            self.from_dict(base_cfg_dict['articulated'])
        else:
            self.__post_init__()

    def __post_init__(self):
        if self.realsense_recorder_endpoint is None:
            self.realsense_recorder_endpoint = f'http://<endpoint>:<port>'

        if self.tcpbroker_endpoint is None:
            self.tcpbroker_endpoint = f'http://<endpoint>:<port>'

        if self.imu_friendly_name_mapping is None:
            self.imu_friendly_name_mapping = {'<friendly_name_1>': '<id_1>', '<friendly_name_2>': '<id_2>'}

        if self.realsense_friendly_name_mapping is None:
            self.realsense_friendly_name_mapping = {'<friendly_name_1>': '<id_1>', '<friendly_name_2>': '<id_2>'}

        if self.base_dir is None:
            self.base_dir = 'path/to/record/volume'

        if self.targets is None:
            self.targets = []

    def to_dict(self):
        return {
            'name': self.name,
            'realsense_recorder_endpoint': self.realsense_recorder_endpoint,
            'tcpbroker_endpoint': self.tcpbroker_endpoint,
            'imu_friendly_name_mapping': self.imu_friendly_name_mapping,
            'realsense_friendly_name_mapping': self.realsense_friendly_name_mapping,
            'base_dir': self.base_dir,
            'debug': self.debug,
            'targets': self.targets,
        }

    def from_dict(self, cfg_dict):
        self.name = cfg_dict['name']
        self.realsense_recorder_endpoint = cfg_dict['realsense_recorder_endpoint']
        self.tcpbroker_endpoint = cfg_dict['tcpbroker_endpoint']
        self.imu_friendly_name_mapping = cfg_dict['imu_friendly_name_mapping']
        self.realsense_friendly_name_mapping = cfg_dict['realsense_friendly_name_mapping']
        self.base_dir = cfg_dict['base_dir']
        self.debug = cfg_dict['debug']
        self.targets = cfg_dict['targets']
        self.__post_init__()


@dataclass()
class DefaultTarget:
    tag: str = None
    uuid: str = None
    finished: bool = None
    imu_dependency: List[str] = None
    recordings: List[str] = None
    targets: List[Dict[str, Any]] = None
    associated_calibration: str = None

    def __post_init__(self):
        if self.uuid is None:
            self.uuid = str(uuid.uuid1())
        if self.finished is None:
            self.finished = False
        if self.imu_dependency is None:
            self.imu_dependency = []
        if self.recordings is None:
            self.recordings = []

    def to_dict(self):
        return {
            'tag': self.tag,
            'uuid': self.uuid,
            'finished': self.finished,
            'imu_dependency': self.imu_dependency,
            'recordings': self.recordings,
            'associated_calibration': self.associated_calibration
        }

    def from_dict(self, cfg_dict):
        self.tag = cfg_dict['tag']
        self.uuid = cfg_dict['uuid']
        self.finished = cfg_dict['finished']
        self.imu_dependency = cfg_dict['imu_dependency']
        self.recordings = cfg_dict['recordings']
        self.associated_calibration = cfg_dict['associated_calibration']
