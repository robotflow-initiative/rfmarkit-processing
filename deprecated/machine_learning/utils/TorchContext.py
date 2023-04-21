from typing import Dict
import time


class TorchContextTimer:
    _start: float = .0

    def __init__(self):
        self.reset()

    def __enter__(self):
        self.reset()
        return self

    @property
    def duration(self):
        return time.time() - self._start

    def reset(self):
        self._start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TorchContext:
    current_epoch: int = 1
    timers: Dict[str, TorchContextTimer] = {"main": TorchContextTimer()}
    tot_loss: float = .0
    tot_acc: float = .0
    tot_recall: float = .0
    tot_precision: float = .0
    n_item: int = 0

    def __init__(self):
        pass

    def add_timer(self, key: str = 'main') -> None:
        """Add a timer

        Args:
            key: name of the timer
        """
        self.timers[key] = TorchContextTimer()

    def del_timer(self, key: str = 'main') -> bool:
        """Delete a timer

        Args:
            key: name of the timer
        """
        if key in self.timers.keys():
            del self.timers[key]
            return True
        else:
            return False

    def reset_timer(self, key: str = 'main') -> bool:
        """reset a timer

        Args:
            key: name of the timer
        """
        if key == 'all':
            for key in self.timers.keys():
                self.reset_timer(key)
        else:
            if key in self.timers.keys():
                self.timers[key].reset()
                return True
            else:
                return False

    @staticmethod
    def _detach_tensor(x):
        return x.detach().cpu().numpy()

    def update_loss(self, loss: float) -> None:
        """Accumulate loss value

        Args:
            loss: loss of current batch
        """
        self.tot_loss += loss

    def next_batch(self) -> None:
        self.n_item += 1

    def update_acc(self, score: float) -> None:
        self.tot_acc += score

    def update_precision(self, score: float) -> None:
        self.tot_precision += score

    def update_recall(self, score: float) -> None:
        self.tot_recall += score

    @property
    def f1_score(self) -> float:
        """ Calculate F1 score
        Returns:
            F1 score
        """
        return 2 * self.tot_precision * self.tot_recall / (self.tot_precision + self.tot_recall)

    def reset_epoch(self) -> None:
        """Reset on epoch end
        Returns:
        """
        self.tot_loss = .0
        self.tot_acc = .0
        self.tot_precision = .0
        self.tot_recall = .0
        self.n_item = 0

    @property
    def avg_loss(self) -> float:
        return self.tot_loss / (self.n_item + 0.001)

    @property
    def avg_score(self) -> float:
        return self.tot_acc / (self.n_item + 0.001)

    def next_epoch(self) -> None:
        self.current_epoch += 1
