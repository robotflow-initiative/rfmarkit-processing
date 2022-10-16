from typing import Callable


class FolderReader:
    def __init__(self, path_to_folder: str, sorting_function: Callable):
        self.path_to_folder = path_to_folder
        self.sorting_function = sorting_function
        self.curr_idx: int = 0

    def load(self, in_memory: bool = False):
        # Probe files
        pass

    def next(self):
        self.curr_idx += 1
        pass

    def prev(self):
        self.curr_idx -= 1
        pass

    @property
    def eof(self):
        return False
