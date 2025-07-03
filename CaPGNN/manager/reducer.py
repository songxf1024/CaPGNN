import torch
import torch.distributed as dist
from multiprocessing.pool import ThreadPool


class Reducer(object):
    '''Used to synchronize model parameter gradients'''
    def __init__(self):
        super(Reducer, self).__init__()
        self._data_cpu = {} # Store gradient data for each parameter
        self._pool = None
        self._handles = []
        self._stream = None

    def init(self, model):
        '''Allocate memory on the CPU for each parameter and create a distributed group'''
        cnt = 0
        for i, (name, param) in enumerate(model.named_parameters()):
            cnt += 1
            self._data_cpu[name] = (torch.zeros_like(param.data, pin_memory=True, device='cpu'), dist.new_group())
        self._pool = ThreadPool(processes=cnt)
        self._stream = torch.cuda.Stream()

    def reduce(self, param, name, data, n_train):
        def create_stream():
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                data.div_(n_train)
                data_cpu, group = self._data_cpu[name]
                data_cpu.copy_(data)
                dist.all_reduce(data_cpu, op=dist.ReduceOp.SUM, group=group)
                param.grad.copy_(data_cpu, non_blocking=True)
        self._handles.append(self._pool.apply_async(create_stream))

    def synchronize(self):
        for handle in self._handles: handle.wait()
        self._handles.clear()
        torch.cuda.current_stream().wait_stream(self._stream)
