import dgl
import gevent
import torch
from typing import Any, Tuple, Union
from functools import wraps
from dgl import DGLHeteroGraph, DGLError
from torch import Tensor
from contextlib import contextmanager
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
from dgl import function as fn

from .op_util import msg_all2all_GLOO
from ..communicator import Communicator as comm
from ..helper import ProprogationMode, BitType
from ..manager import DecompGraph
from ..manager import GraphEngine as engine
from ..util.timer import TimerRecord


def GCN_aggregation(graph: DGLHeteroGraph,
                    feats: Tensor,
                    mode: ProprogationMode = ProprogationMode.Forward,
                    allow_zero_in_degree=False):
    with graph.local_scope():
        if not allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError(
                  "There are 0-in-degree nodes in the graph, and the output of these nodes will be invalid. This is harmful to some applications and will cause performance degradation." 
                  "Call g = dgl.add_self_loop(g) and add a self-loop on the input graph to solve this problem. When building this model "
                  " block, setting allow_zero_in_degree to True will suppress the check and let the code run. Calling " 
                  "add_self_loop will not work for some graphs, such as heterogeneous graphs, because the edge type of self_loop edge cannot be determined." 
                  "In this case, set allow_zero_in_degree to True to unblock the code and manually handle the zero-degree node." 
                  "The common way to handle this is to filter out zero-in-degree when used after conv node. "
                )
        # if dgl.__version__ < '0.10':
        #     aggregate_fn = fn.copy_src('h', 'm')  # dgl v0.9
        # else:
        aggregate_fn = fn.copy_u('h', 'm')  # dgl v2.2
        if mode == ProprogationMode.Forward:
            # norm1 = graph.ndata['out_degrees'].float().clamp(min=1).pow(-0.5) # out degrees for forward
            # norm2 = graph.ndata['in_degrees'].float().clamp(min=1).pow(-0.5) # in degrees for forward
            norm1 = graph.out_degrees().float().clamp(min=1).pow(-0.5) # out degrees for forward
            norm2 = graph.in_degrees().float().clamp(min=1).pow(-0.5) # in degrees for forward
        elif mode == ProprogationMode.Backward:
            # norm1 = graph.ndata['in_degrees'].float().clamp(min=1).pow(-0.5)  # in degrees for backward
            # norm2 = graph.ndata['out_degrees'].float().clamp(min=1).pow(-0.5)  # out degrees for backward
            norm1 = graph.in_degrees().float().clamp(min=1).pow(-0.5)  # in degrees for backward
            norm2 = graph.out_degrees().float().clamp(min=1).pow(-0.5)  # out degrees for backward
        else:
            raise ValueError(f'Invalid mode {mode}')
        feats = feats * norm1.view(-1, 1)
        graph.srcdata['h'] = feats
        graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
        rst = graph.dstdata['h'] * norm2.view(-1, 1)
        return rst

def SAGE_aggregation(graph: DGLHeteroGraph, feats: Tensor, mode: ProprogationMode = ProprogationMode.Forward, aggregator_type='mean'):
    with graph.local_scope():
        # aggregate_fn = fn.copy_src('h', 'm')
        aggregate_fn = fn.copy_u('h', 'm')  # dgl v2.2
        if mode == ProprogationMode.Forward:
            graph.srcdata['h'] = feats
            if aggregator_type == 'mean':
                graph.update_all(aggregate_fn, fn.mean(msg='m', out='neigh'))
                h_neigh = graph.dstdata['neigh']
            elif aggregator_type == 'gcn':
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='neigh'))
                degs = graph.ndata['in_degrees'].float().clamp(min=1).view(-1, 1)
                h_neigh = (graph.dstdata['neigh'] + graph.srcdata['h']) / (degs + 1)
            else:
                raise ValueError(f'Invalid aggregator_type {aggregator_type}')
        elif mode == ProprogationMode.Backward:
            if aggregator_type == 'mean':
                norm = graph.ndata['out_degrees'].float().clamp(min=1)
                norm = torch.pow(norm, -1)
                feats = feats * norm.view(-1, 1)
                graph.srcdata['h'] = feats
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='neigh'))
                h_neigh = graph.dstdata['neigh']
            elif aggregator_type == 'gcn':
                norm = graph.ndata['out_degrees'].float().clamp(min=1) + 1
                norm = torch.pow(norm, -1).view(-1, 1)
                feats = feats * norm
                graph.srcdata['h'] = feats
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='neigh'))
                h_neigh = graph.dstdata['neigh'] + graph.srcdata['h']
            else:
                raise ValueError(f'Invalid aggregator_type {aggregator_type}')
        else:
            raise ValueError(f'Invalid mode {mode}')
        return h_neigh

class DistAggConv(Function):
    '''
    Custom distributed aggregation function class allows GCN to aggregate local and remote neighbor features.
    '''
    @staticmethod
    @custom_fwd
    def forward(ctx, local_messages: Tensor,
                graph: Union[DGLHeteroGraph, DecompGraph],
                layer: int, is_train: bool) -> Tensor:
        return full_graph_propagation(ctx, local_messages, graph, layer, is_train, ProprogationMode.Forward, DistAggConv.__name__)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, *grad_outputs: Tuple[Tensor, ...]) -> Tensor:
        layer = ctx.saved
        local_messages = grad_outputs[0]
        return full_graph_propagation(ctx, local_messages, engine.ctx.bwd_graph, layer, True, ProprogationMode.Backward, DistAggConv.__name__)

class DistAggSAGE(Function):
    '''
    customized distributed aggregation Function class which aggregates features from both local and remote neighbors for GraphSAGE.
    '''
    @staticmethod
    @custom_fwd
    def forward(ctx, local_messages: Tensor, graph: Union[DGLHeteroGraph, DecompGraph], layer: int, is_train: bool) -> Tensor:
        return full_graph_propagation(ctx, local_messages, graph, layer, is_train, ProprogationMode.Forward, DistAggSAGE.__name__)

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, *grad_outputs: Tuple[Tensor, ...]) -> Tensor:
        layer = ctx.saved
        local_messages = grad_outputs[0]
        return full_graph_propagation(ctx, local_messages, engine.ctx.bwd_graph, layer, True, ProprogationMode.Backward, DistAggSAGE.__name__)

'''
*************************************************
*********** forward/backward functions **********
*************************************************
'''

def feat_concat(rank, size, layer, feat):
    tmp = [feat]
    _f_recv = comm.ctx.comm_buffer.test_recv_buffers_gpu
    for i in range(size):
        if i != rank:
            tmp.append(_f_recv[layer][0][i])
    return torch.cat(tmp)

def full_graph_propagation(ctx,
                           local_messages: Tensor,
                           graph: Union[DGLHeteroGraph, DecompGraph],
                           layer: int,
                           is_train: bool,
                           mode: ProprogationMode,
                           class_name: str) -> Union[Tensor, Tuple[Tensor, None, None, None]]:
    # exchange messages (features/embeddings/embedding gradients)
    # All halo features on the local graph, pay attention to cat, so they are one-dimensional and need to be used with send_idx used to mark the start and end range.
    # torch.cuda.current_stream().synchronize()
    send_messages = local_messages[engine.ctx.total_send_idx]
    name = f'forward{layer}' if mode == ProprogationMode.Forward else f'backward{layer}'
    if engine.ctx.use_pipeline is False:
        # engine.ctx.timer.start(f'msg_comm')
        remote_messages = msg_all2all_GLOO(send_messages, name, is_train)
        # engine.ctx.timer.stop(f'msg_comm')
        full_messages = torch.cat([local_messages, remote_messages], dim=0)
        # response = engine.ctx.marginal_pool.apply_async(msg_all2all_GLOO, args=(send_messages, name, is_train))
        # remote_messages = response.get()
        full_messages = torch.cat([local_messages, remote_messages], dim=0)
    else:
        rank, size = comm.get_rank(), comm.get_world_size()
        if engine.ctx.curr_epoch > 0:
            # if is_train: engine.ctx.timer.start(f'msg_comm')
            with engine.ctx.timer.record('msg_comm'):
                engine.ctx._f_cpu_event[layer].wait()
                torch.cuda.current_stream().wait_event(engine.ctx._f_cuda_event[layer])
                engine.ctx._f_cpu_event[layer].clear()
            # if is_train: engine.ctx.timer.stop(f'msg_comm')
        full_messages = feat_concat(rank, size, layer, local_messages)
        engine.ctx.marginal_pool.apply_async(msg_all2all_GLOO, args=(send_messages, name, is_train))


    # aggregate messages
    # engine.ctx.timer.start('aggregation')
    with engine.ctx.timer.record('aggregation'):
        if class_name == 'DistAggConv':
            rst = GCN_aggregation(graph, full_messages, mode=mode, allow_zero_in_degree=True)
        elif class_name == 'DistAggSAGE':
            rst = SAGE_aggregation(graph, full_messages, mode=mode, aggregator_type=engine.ctx.agg_type)
        else:
            raise ValueError(f'Invalid class_name {class_name}')
    # engine.ctx.timer.stop('aggregation')
    len_local = local_messages.shape[0]
    # Extract the part of the local message from the aggregate result rst
    return_messages = rst[:len_local]
    # store layer info in forward propagation 
    if mode == ProprogationMode.Forward:
        ctx.saved = layer
        return return_messages
    # return gradients in backward propagation
    else:
        return return_messages, None, None, None
