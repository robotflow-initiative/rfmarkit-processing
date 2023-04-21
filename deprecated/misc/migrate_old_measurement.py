import os

from cvt_measurement import convert_measurement

ROOT_DIR = "E:/"

walker = os.walk(ROOT_DIR)
imu_dirs = []

for path, dir_list, file_list in walker:
    if any(map(lambda x: '.dat' in x, file_list)) and any(map(lambda x: 'README' in x, file_list)):
        imu_dirs.append(path)

for dir in imu_dirs:
    convert_measurement(dir)
