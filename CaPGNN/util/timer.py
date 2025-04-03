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
    START   = 'start'   # 本次调用start函数时的时间戳
    END     = 'end'     # 本次调用end函数时的时间戳
    ELAPSED = 'elapsed' # 多次调用start和end函数的总耗时
    TOTAL   = 'total'   # 本次所耗end-start的时间
    HISTORY = 'history' # 保存每次的elapsed


import time


class TimerRecord:
    """
    一个计时上下文管理器，用于测量代码块的执行时间。
    它可以根据用户的需求以秒或纳秒为单位测量时间。

    属性:
        use_ns (bool): 如果为True，则计时器以纳秒为单位测量时间；如果为False，则以秒为单位。
        start (float|int): 测量开始的时间。
        end (float|int): 测量结束的时间。
        interval (float|int): 计算的开始和结束时间之间的持续时间。
    """

    def __init__(self, use_ns=False, prefix='', show=True):
        """
        使用选择是否使用纳秒精度初始化 Timer。
        参数:
            use_ns (bool): 确定是否使用纳秒进行时间测量，默认为False。
        """
        self.use_ns = use_ns
        self.start = None
        self.end = None
        self.interval = None
        self.prefix = prefix
        self.show = show

    def __enter__(self):
        """
        启动计时器。当进入上下文块时记录开始时间。
        返回:
            Timer: 返回自身对象，以便在上下文外部访问属性。
        """
        if self.use_ns:
            self.start = time.perf_counter_ns()
        else:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        结束计时器。当退出上下文块时记录结束时间。
        此函数还计算时间间隔并打印经过的时间。
        参数:
            exc_type: 如果在上下文中引发异常，则为异常类型。
            exc_value: 如果引发异常，则为异常值。
            traceback: 如果发生异常，则为回溯详细信息。
        返回:
            None
        """
        if self.use_ns:
            self.end = time.perf_counter_ns()
            self.interval = self.end - self.start
            if self.show: print(f"{self.prefix} 经过时间：{self.interval} 纳秒")
        else:
            self.end = time.perf_counter()
            self.interval = self.end - self.start
            if self.show: print(f"{self.prefix} 经过时间：{self.interval:.6f} 秒")




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
        if show: print(f"{name} 经过时间：{gap} 秒")

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
        '''开始计时，通过name区分不同的计时器；
        在start时如果开启了history，就算下次start时没有开启history，history仍然会使用，
        除非显式调用了reset_all或reset_item，然后再调用start并不开启history，就不会记录。'''
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
        '''计算指定name的本次耗时'''
        if not self.is_train: return
        if self.pretrain: return
        if name not in ['epoch', 'msg_comm']: return
        # torch.cuda.current_stream(self._device).synchronize()
        if self._record_sxf.get(name) and self._record_sxf[name].get(TimerKeys.START):
            self._record_sxf[name][TimerKeys.END] = time.perf_counter()
            self._record_sxf[name][TimerKeys.ELAPSED] = self._record_sxf[name][TimerKeys.END] - self._record_sxf[name][TimerKeys.START]
            return self.store(name) if store else self._record_sxf[name][TimerKeys.ELAPSED]
        else:
            print(f'>> 不存在此计时器[{name}]，请先start')
        return None

    def store(self, name):
        '''计算指定name的累计耗时'''
        if not self._record_sxf.get(name):
            print(f'>> 不存在此计时器[{name}]，请先start')
            return None
        if not self._record_sxf[name].get(TimerKeys.ELAPSED):
            print(f'>> 请先stop')
            return None
        self._record_sxf[name][TimerKeys.TOTAL] += self._record_sxf[name][TimerKeys.ELAPSED]
        if self._record_sxf[name].get(TimerKeys.HISTORY) is not None:
            self._record_sxf[name][TimerKeys.HISTORY].append(self._record_sxf[name][TimerKeys.ELAPSED])
        return self._record_sxf[name][TimerKeys.TOTAL]

    def show_store(self):
        '''显示所有项目的累计耗时'''
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
            duration = self.start_event.elapsed_time(self.end_event) / 1000.0  # 转换为秒
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
                avg_dict[key], std_dict[key] = data[0], 0.0  # 如果只有一个数据点，标准差为0
            else:
                avg_dict[key], std_dict[key] = float('nan'), float('nan')  # 没有数据点，设置为NaN
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

