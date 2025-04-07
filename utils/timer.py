import time
import torch
import pickle
import statistics
from collections import defaultdict
import util


class TimerCtx:
    def __init__(self, timer, key, cuda, device=None, barrier=False):
        self.device = device
        self.cuda = cuda
        self.timer = timer
        self.key = key
        if cuda is True and device is None: raise RuntimeError("Device must be specified when using CUDA timing")
        self.start_event = torch.cuda.Event(enable_timing=True) if cuda else None
        self.end_event = torch.cuda.Event(enable_timing=True) if cuda else None
        self.barrier = barrier

    def __enter__(self):
        if self.cuda: torch.cuda.synchronize()
        self.timer.start_time_dict[self.key] = time.perf_counter()
        # if self.cuda:
        #     if self.barrier: self.timer.env.barrier_all()
        #     with torch.cuda.device(self.device):
        #         self.start_event.record()
        # else:
        #     self.timer.start_time_dict[self.key] = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if self.cuda: torch.cuda.synchronize()
        duration = time.perf_counter() - self.timer.start_time_dict[self.key]
        # if self.cuda:
        #     if self.barrier: self.timer.env.barrier_all()
        #     with torch.cuda.device(self.device):
        #         self.end_event.record()
        #     torch.cuda.synchronize(self.device)
        #     duration = self.start_event.elapsed_time(self.end_event) / 1000.0  # 转换为秒
        # else:
        #     duration = time.perf_counter() - self.timer.start_time_dict[self.key]
        self.timer._log_duration(self.key, duration)





class DistTimer:
    def __init__(self, env):
        self.env = env
        self.start_time_dict = {}
        self.duration_dict = defaultdict(float)
        self.duration_list = defaultdict(list)
        self.count_dict = defaultdict(int)
        self.timers = {}

    def summary(self):
        summary_list = []
        for key in self.duration_dict:
            std = 0 if len(self.duration_list[key])<2 else statistics.pstdev(self.duration_list[key])
            ave = statistics.mean(self.duration_list[key])
            total = self.duration_dict[key]
            count = self.count_dict[key]
            summary_list.append({'key': key, 'total': total, 'ave':ave, 'std': std, 'count': count})
        # summary_str = '\ntimer summary:\n' + "\n".join(f"{self.duration_dict[key]:6.2f}s {self.count_dict[key]:5d} {key}" for key in self.duration_dict)
        summary_str = "\nTimer Summary:\n"
        summary_str += "{:<10} {:>10} {:>10} {:>10} {:>10}\n".format(
            "Key", "Total", "Ave", "Std", "Count"
        )
        summary_str += "-" * 50 + "\n"
        summary_str += "\n".join(
            [
                "{:<10} {:>10.4f} {:>10.4f} {:>10.4f} {:>10d}".format(
                    item["key"], item["total"], item["ave"], item["std"], item["count"]
                )
                for item in summary_list
            ]
        )
        return summary_list, summary_str

    def sync_duration_dicts(self, barrier=True):
        self.env.store.set(f'duration_dict_{self.env.rank}', pickle.dumps(self.duration_dict))
        if barrier: self.env.barrier_all()
        self.all_durations = [pickle.loads(self.env.store.get(f'duration_dict_{rank}')) for rank in range(self.env.world_size)]

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
                'total': self.duration_dict[key],
                'ave': avg_dict[key],
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
        self.duration_list[key].append(duration)
        self.duration_dict[key] += duration
        self.count_dict[key] += 1

    def timing(self, key):
        return TimerCtx(self, key, cuda=False)

    def timing_cuda(self, key, barrier=False):
        return TimerCtx(self, key, cuda=True, device=self.env.device, barrier=barrier)


if __name__ == '__main__':
    class Temp:
        def __init__(self):
            self.start_time_dict = {}
            self.duration_dict = defaultdict(float)
            self.count_dict = defaultdict(int)

        def _log_duration(self, key, duration):
            self.duration_dict[key] += duration
            self.count_dict[key] += 1


    device = torch.device('cuda:0')

    temp = Temp()
    with TimerCtx(temp, key='a', cuda=True, device=device):
        # util.high_precision_sleep(5)
        util.warm_up_gpus(device)
    print(temp.duration_dict.items())
    # [('a', 5.00178125)]

    with TimerCtx(temp, key='b', cuda=False):
        # util.high_precision_sleep(5)
        util.warm_up_gpus(device)
    print(temp.duration_dict.items())
    # [('a', 5.00178125), ('b', 5.000027532922104)]




