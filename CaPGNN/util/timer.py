import sys
import time
import torch
import os
from contextlib import contextmanager
# from ..communicator import Communicator as comm
from enum import Enum, unique
import time
import torch
import pickle
import statistics
from collections import defaultdict
import torch.distributed as dist
import tempfile

class Colors(object):
    '''
    Colors class:reset all colors with colors.reset; two
    sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,
    underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold
    '''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'

class Printer(object):
    @staticmethod
    def red(*args, **kwargs):
        print(Colors.fg.red, *args, **kwargs)
        print(Colors.reset, end="")

    @staticmethod
    def green(*args, **kwargs):
        print(Colors.fg.green, *args, **kwargs)
        print(Colors.reset, end="")

    @staticmethod
    def blue(*args, **kwargs):
        print(Colors.fg.blue, *args, **kwargs)
        print(Colors.reset, end="")

    @staticmethod
    def cyan(*args, **kwargs):
        print(Colors.fg.cyan, *args, **kwargs)
        print(Colors.reset, end="")

    @staticmethod
    def orange(*args, **kwargs):
        print(Colors.fg.orange, *args, **kwargs)
        print(Colors.reset, end="")

    @staticmethod
    def purple(*args, **kwargs):
        print(Colors.fg.purple, *args, **kwargs)
        print(Colors.reset, end="")

    @staticmethod
    def yellow(*args, **kwargs):
        print(Colors.fg.yellow, *args, **kwargs)
        print(Colors.reset, end="")

    @staticmethod
    def error(*args, **kwargs):
        print(Colors.fg.red, *args, **kwargs, file=sys.stderr)
        print(Colors.reset, end="")

@unique
class TimerKeys(Enum):
    START   = 'start'   # The time stamp when the start function is called this time
    END     = 'end'     # Timestamp when calling the end function
    ELAPSED = 'elapsed' # Total time spent calling start and end functions multiple times
    TOTAL   = 'total'   # The end-start time spent this time
    HISTORY = 'history' # Save each elapsed


import time


class TimerRecord:
    """
    A timing context manager that measures the execution time of a code block. It can measure time in seconds or nanoseconds according to user needs. 
    property:
        use_ns (bool): If True, the timer measures time in nanoseconds; if False, the unit is seconds. 
        start (float|int): The time to start the measurement. 
        end (float|int): The time at which the measurement ends. 
        interval (float|int): The duration between the calculated start and end times.
    """

    def __init__(self, use_ns=False, prefix='', show=True):
        """
        Use to select whether to initialize the Timer with nanosecond precision. 
        parameter:
            use_ns (bool): Determines whether to use nanoseconds for time measurement, default is False.
        """
        self.use_ns = use_ns
        self.start = None
        self.end = None
        self.interval = None
        self.prefix = prefix
        self.show = show

    def __enter__(self):
        """
        Start the timer. Record the start time when entering the context block. 
        return:
            Timer: Returns its own object to access properties outside the context.
        """
        if self.use_ns:
            self.start = time.perf_counter_ns()
        else:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        End timer. Record the end time when the context block is exited. 
        This function also calculates the time interval and prints the elapsed time. 
        parameter:
            excc_type: If an exception is raised in the context, it is an exception type. 
            exc_value: If an exception is raised, it is an outlier. 
            traceback: If an exception occurs, it is the traceback details. 
        return:
            None
        """
        if self.use_ns:
            self.end = time.perf_counter_ns()
            self.interval = self.end - self.start
            if self.show: print(f"{self.prefix} Elapsed time: {self.interval} nanoseconds")
        else:
            self.end = time.perf_counter()
            self.interval = self.end - self.start
            if self.show: print(f"{self.prefix} Elapsed time: {self.interval:.6f} seconds")




class Timer(object):
    def __init__(self, device: torch.device):
        super(Timer, self).__init__()
        self._record = {}
        self._total_record = []
        self._device = device
        self._record_sxf = {}
        self.is_train = True
        self.pretrain = False

    @contextmanager
    def record2(self, name: str, show=False):
        self.start(name)
        yield
        self.stop(name)

    @contextmanager
    def record(self, name: str, show=False):
        if not self.is_train or self.pretrain:
            yield
            return
        if name not in ['epoch', 'msg_comm', 'aggregation']:
            yield
            return
        self._record_sxf[name] = self._record_sxf.get(name, 0)
        # torch.cuda.current_stream(self._device).synchronize()
        start = time.perf_counter()
        yield
        # torch.cuda.current_stream(self._device).synchronize()
        gap = time.perf_counter() - start
        self._record_sxf[name] += gap
        if show: print(f"{name} Time passed：{gap} 秒")

    def epoch_traced_time(self):
        total_com = 0
        for name, (start, end) in self._record.items():
            if 'communication' in name:
                total_com += end - start
        return [total_com, ]

    def clear(self, is_train: bool = True):
        # store in the _total_record for backup
        if is_train:
            self._total_record.append(self.epoch_traced_time())
        self._record = {}

    def persist(self, run: int, bit_type, exp_dir: str = 'exp'):
        store_dir = f'{exp_dir}/time_record'
        mode = 'full'
        if not os.path.exists(store_dir):
            os.mkdir(store_dir)
        # torch.save(self._total_record, f'{store_dir}/run{run}_{mode}worker{comm.get_rank()}_time_record.pt')
        torch.save(self._total_record, f'{store_dir}/run{run}_{mode}worker_time_record.pt')
        self._total_record = []

    def start(self, name, history=False):
        '''Start timing and distinguish different timers by name; if history is enabled at start, even if history is not enabled at the next start, history will still be used. 
        Unless reset_all or reset_item is explicitly called, and then start is called and history does not turn on, it will not be recorded.'''
        if not self.is_train: return
        if self.pretrain: return
        if name not in ['epoch', 'msg_comm']: return
        # torch.cuda.current_stream(self._device).synchronize()
        if self._record_sxf.get(name):
            self._record_sxf[name].pop(TimerKeys.END)
            self._record_sxf[name].pop(TimerKeys.ELAPSED)
        else:
            self._record_sxf[name] = {}
            self._record_sxf[name][TimerKeys.TOTAL] = 0
            if history: self._record_sxf[name][TimerKeys.HISTORY] = []
        self._record_sxf[name][TimerKeys.START] = time.perf_counter()

    def stop(self, name, store=True):
        '''Calculate the time spent on the specified name'''
        if not self.is_train: return
        if self.pretrain: return
        if name not in ['epoch', 'msg_comm']: return
        # torch.cuda.current_stream(self._device).synchronize()
        if self._record_sxf.get(name) and self._record_sxf[name].get(TimerKeys.START):
            self._record_sxf[name][TimerKeys.END] = time.perf_counter()
            self._record_sxf[name][TimerKeys.ELAPSED] = self._record_sxf[name][TimerKeys.END] - self._record_sxf[name][TimerKeys.START]
            return self.store(name) if store else self._record_sxf[name][TimerKeys.ELAPSED]
        else:
            print(f'>> This timer [{name}] does not exist, please start first')
        return None

    def store(self, name):
        '''Calculate the cumulative time spent on a specified name'''
        if not self._record_sxf.get(name):
            print(f'>> This timer [{name}] does not exist, please start first')
            return None
        if not self._record_sxf[name].get(TimerKeys.ELAPSED):
            print(f'>> Please stop first')
            return None
        self._record_sxf[name][TimerKeys.TOTAL] += self._record_sxf[name][TimerKeys.ELAPSED]
        if self._record_sxf[name].get(TimerKeys.HISTORY) is not None:
            self._record_sxf[name][TimerKeys.HISTORY].append(self._record_sxf[name][TimerKeys.ELAPSED])
        return self._record_sxf[name][TimerKeys.TOTAL]

    def show_store(self):
        '''Shows the cumulative time consumption of all items'''
        print(self._record_sxf)

    def pretty_show_store(self, prefix='', simple=True):
        print(prefix if prefix else '')
        print("{", end='')
        for key, value in self._record_sxf.items():
            if simple:
                if type(value) is dict:
                    print(f"{key}: {value[TimerKeys.TOTAL]}")
                else:
                    print(f"{key}: {value}") # [TimerKeys.TOTAL]
            else:
                print(f"{key}: {{")
                for enum_key, enum_value in value.items():
                    print(f"    {enum_key.name if isinstance(enum_key, Enum) else enum_key}: {enum_value},")
                print("  },")
        print("}")

    def get_store(self):
        return self._record_sxf

    def peak_item2(self, name, key=None):
        if key:
            return self._record_sxf[name].get(key) if self._record_sxf.get(name) else None
        return self._record_sxf.get(name)

    def peak_item(self, name, key=None):
        if self._record_sxf.get(name) is None: return None
        if type(self._record_sxf[name]) is dict: return self._record_sxf[name].get(key)
        return self._record_sxf.get(name)

    def reset_item(self, name):
        self._record_sxf.pop(name)

    def reset_all(self):
        self._record_sxf = {}



class TimerCtx:
    def __init__(self, timer, key, cuda, device=None):
        self.device = device
        self.cuda = cuda
        self.timer = timer
        self.key = key
        if cuda is True and device is None: raise RuntimeError("Device must be specified when using CUDA timing")
        self.start_event = torch.cuda.Event(enable_timing=True) if cuda else None
        self.end_event = torch.cuda.Event(enable_timing=True) if cuda else None
        self.stream = torch.cuda.current_stream(device)

    def __enter__(self):
        # if self.cuda: torch.cuda.synchronize()
        # self.timer.start_time_dict[self.key] = time.perf_counter()
        if self.cuda:
            self.start_event.record(self.stream)
        else:
            self.timer.start_time_dict[self.key] = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        # if self.cuda: torch.cuda.synchronize()
        # duration = time.perf_counter() - self.timer.start_time_dict[self.key]
        if self.cuda:
            self.end_event.record(self.stream)
            self.stream.synchronize()
            duration = self.start_event.elapsed_time(self.end_event) / 1000.0  # Convert to seconds
        else:
            duration = time.perf_counter() - self.timer.start_time_dict[self.key]
        self.timer._log_duration(self.key, duration)

class DistTimer:
    def __init__(self, device, world_size, rank):
        self.start_time_dict = {}
        self.duration_dict = defaultdict(float)
        self.count_dict = defaultdict(int)
        self.timers = {}
        self.world_size = world_size
        self.device = device
        self.rank = rank
        try:
            self.store = dist.FileStore(os.path.join(tempfile.gettempdir(), 'torch-dist'), self.world_size)
        except:
            self.store = dist.FileStore(os.path.join(tempfile.gettempdir(), 'torch-dist-sxf'), self.world_size)

    def summary(self):
        summary_list = []
        for key in self.duration_dict:
            summary_list.append({'key': key, 'avg': self.duration_dict[key], 'std':0, 'count': self.count_dict[key]})
        summary_str = '\ntimer summary:\n' + "\n".join(f"{self.duration_dict[key]:6.2f}s {self.count_dict[key]:5d} {key}" for key in self.duration_dict)
        return summary_list, summary_str

    def sync_duration_dicts(self, barrier=True):
        self.store.set(f'duration_dict_{self.rank}', pickle.dumps(self.duration_dict))
        if barrier: dist.barrier(self.world_size)
        self.all_durations = [pickle.loads(self.store.get(f'duration_dict_{rank}')) for rank in range(self.world_size)]

    def summary_all(self, barrier=True):
        self.sync_duration_dicts(barrier)
        avg_dict = {}
        std_dict = {}
        summary_list = []
        for key in self.duration_dict:
            data = [d[key] for d in self.all_durations]
            if len(data) > 1:
                avg_dict[key], std_dict[key] = statistics.mean(data), statistics.pstdev(data)
            elif len(data) == 1:
                avg_dict[key], std_dict[key] = data[0], 0.0  # If there is only one data point, the standard deviation is 0
            else:
                avg_dict[key], std_dict[key] = float('nan'), float('nan')  # No data points, set to NaN
            summary_list.append({
                'key': key,
                'avg': avg_dict[key],
                'std': std_dict[key],
                'count': self.count_dict[key]
            })
        summary_str = '\nave timer summary:\n' + "\n".join("%6.2fs %6.2fs %5d %s" % (avg_dict[key], std_dict[key], self.count_dict[key], key) for key in self.duration_dict)
        return summary_list, summary_str

    def detail_all(self, barrier=True):
        self.sync_duration_dicts(barrier)
        avg_dict = {}
        std_dict = {}
        detail_dict = {}
        for key in self.duration_dict:
            data = [d[key] for d in self.all_durations]
            avg_dict[key], std_dict[key] = statistics.mean(data), statistics.pstdev(data)
            detail_dict[key] = ' '.join("%6.2f"%x for x in data)
        detail_str = '\ndetail timer summary:\n'
        detail_str += "\n".join(
            f"{avg_dict[key]:6.2f}s {std_dict[key]:6.2f}s {self.count_dict[key]:5d} {key}\ndetail: {detail_dict[key]}\n--------------"
            for key in self.duration_dict)
        return detail_str

    def _log_duration(self, key, duration):
        self.duration_dict[key] += duration
        self.count_dict[key] += 1

    def timing(self, key):
        return TimerCtx(self, key, cuda=False)

    def timing_cuda(self, key):
        return TimerCtx(self, key, cuda=True, device=self.device)




if __name__ == '__main__':
    timer = Timer(torch.device('cuda:0'))

    timer.start('a')
    time.sleep(5)
    timer.stop('a')
    # {'a': {'total': 5.005435228347778, 'start': 1693419100.180317, 'end': 1693419105.1857522, 'elapsed': 5.005435228347778}}
    timer.show_store()
    print()

    timer.start('a')
    time.sleep(2)
    timer.stop('a')
    # {'a': {'total': 7.007752180099487, 'start': 1693419105.1859245, 'end': 1693419107.1882415, 'elapsed': 2.002316951751709}}
    timer.show_store()
    print()

    timer.start('b')
    time.sleep(3)
    timer.stop('b')
    # {'a': {'total': 7.007752180099487, 'start': 1693419105.1859245, 'end': 1693419107.1882415, 'elapsed': 2.002316951751709},
    #  'b': {'total': 3.0033228397369385, 'start': 1693419107.1884048, 'end': 1693419110.1917276, 'elapsed': 3.0033228397369385}}
    timer.show_store()

    timer.reset_all()
    print()

    timer.start('c')
    time.sleep(3)
    timer.start('d')
    time.sleep(3)
    timer.stop('d')
    # {'c': {'total': 0, 'start': 1693419110.1919253},
    # 'd': {'total': 3.003229856491089, 'start': 1693419113.1927958, 'end': 1693419116.1960256, 'elapsed': 3.003229856491089}}
    timer.show_store()
    timer.stop('c')
    # {'c': {'total': 6.0042500495910645, 'start': 1693419110.1919253, 'end': 1693419116.1961753, 'elapsed': 6.0042500495910645},
    #  'd': {'total': 3.003229856491089, 'start': 1693419113.1927958, 'end': 1693419116.1960256, 'elapsed': 3.003229856491089}}
    timer.show_store()

    timer.reset_all()
    print()

    timer.start('e')
    time.sleep(3)
    timer.start('f')
    time.sleep(3)
    timer.stop('e')
    # {'e': {'total': 6.004979848861694, 'start': 1693419433.8564444, 'end': 1693419439.8614242, 'elapsed': 6.004979848861694},
    #  'f': {'total': 0, 'start': 1693419436.859731}}
    timer.show_store()
    timer.stop('f')
    # {'e': {'total': 6.004979848861694, 'start': 1693419433.8564444, 'end': 1693419439.8614242, 'elapsed': 6.004979848861694},
    #  'f': {'total': 3.00180983543396, 'start': 1693419436.859731, 'end': 1693419439.8615408, 'elapsed': 3.00180983543396}}
    timer.pretty_show_store()

