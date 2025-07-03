import csv
import os
import datetime as dt
import time
import threading
from queue import Queue

class DistLogger:
    def __init__(self, env, gpus_note='', log_prefix=''):
        self.env = env
        # self.log_root = os.path.join(os.path.dirname(__file__), '..', 'logs', f'world_size_{env.world_size}', str(int(time.time())))
        self.log_root = os.path.join(os.path.dirname(__file__), 'logs', log_prefix or f'world_size_{env.world_size}', str(gpus_note))
        os.makedirs(self.log_root, exist_ok=True)
        self.log_fname = os.path.join(self.log_root, f'rank_{self.env.rank}.txt')
        self.queue = Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()

    def save_log(self, msg):
        with open(self.log_fname, 'a+') as f:
            f.write(msg)
            f.flush()

    def _process_queue(self):
        with open(self.log_fname, 'a+') as f:
            while not self.stop_event.is_set() or not self.queue.empty():
                time.sleep(1)
                try:
                    log_entry = self.queue.get(timeout=1)
                    f.write(log_entry)
                    f.flush()
                except:
                    continue

    def log(self, *args, oneline=False, rank=-1, prefix='', suffix='', show=False, save_log=False):
        '''Too many log outputs will affect the timing accuracy of rank 0.
        It is recommended to keep it off during regular use and only turn it on with show=True when necessary.'''
        if not show and self.env.rank!=rank: return  # or rank!=-1

        head = f'{dt.datetime.now().time()} [{self.env.rank}] '
        tail = '\r' if oneline else '\n'
        # the_whole_line = head+' '.join(map(str, args))+tail
        # print(the_whole_line, end='', flush=True)  # to prevent line breaking
        # with open(self.log_fname, 'a+') as f:
        #     print(the_whole_line, end='', file=f, flush=True)  # to prevent line breaking
        log_entry = prefix + head + ' '.join(map(str, args)) + tail + suffix
        print(log_entry, end='', flush=True)
        if save_log: self.queue.put(log_entry)

    def get_log_root(self):
        return self.log_root

    def save_summary_to_csv(self, summary, file_name='summary.csv', fieldnames=None):
        fieldnames = fieldnames or ['key', 'total', 'ave', 'std', 'count']
        csv_file = os.path.join(self.log_root, file_name)
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'w+', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()  # Write the header only when the file does not exist.
            for row in summary:
                writer.writerow(row)

    def print_summary(self, summary):
        print('\ntimer summary:')
        for entry in summary:
            print(f"{entry['avg']:6.2f}s {entry['std']:6.2f}s {entry['count']:5d} {entry['key']}")

    def close(self):
        self.stop_event.set()
        self.thread.join()


if __name__ == '__main__':
    pass

