from torch.utils.data import Dataset
from torch.utils.data import T_co
import re

class IMUDatasetEntry:
    LABEL_SUBPATH = 'label'
    STIMULIS_SUBPATH = 'stimulis'
    LABEL_PATTERN = ["cartesianPos_{}.csv", "cartesianVec_{}.csv", "cartesianAec_{}.csv"]
    STIMULIS_SUBPATH = ["imu_{}.csv"]
    def __init__(self, path_to_data, ) -> None:
        pass

class IMUDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
    
    def __getitem__(self, index) -> T_co:
        return super().__getitem__(index)
    
    def __len__(self):
        return 0