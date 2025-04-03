import logging
import random
import time
import torch
from typing import Any, List, Union
from dgl import DGLHeteroGraph
from torch import Tensor
from torch import nn
from torch.optim import Optimizer
import numpy as np
from torch.cuda.amp import autocast
from ..communicator import Communicator as comm
from ..manager import GraphEngine as engine

'''
*************************************************
**************** setup functions ****************
*************************************************
'''

def setup_logger(log_file, level=logging.INFO, with_file=True):
    """Function setup as many loggers as you want"""
    config_logger = logging.getLogger('trainer')
    config_logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # file handler
    if with_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        config_logger.addHandler(file_handler)
    return config_logger

def fix_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sync_seed():
    '''
    同步所有workers的随机种子
    '''
    if comm.get_rank() == 0:
        seed = int(time.time() % (2 ** 32 - 1))
        obj_list = [seed]
        comm.broadcast_any(obj_list, src = 0)
        fix_seed(seed)
    else:
        obj_list = [None]
        comm.broadcast_any(obj_list, src = 0)
        seed = obj_list[0]
        fix_seed(seed)

def sync_model(model: nn.Module):
    '''
    synchronize the model parameters of all workers
    '''
    state_dict = model.state_dict()
    for _, s_value in state_dict.items():
        if comm.get_rank() != 0: s_value.zero_()
        comm.all_reduce_sum(s_value.data)

'''
********************************qt_msg_exchange*****************
*************** runtime functions ***************
*************************************************
'''

def average_gradients(model: nn.Module):
    '''
    average所有workers的梯度
    '''
    # print('\n'.join([f'Name: {name}, Shape: {param.shape}' for name, param in model.named_parameters() if param.requires_grad]))
    for name, param in model.named_parameters():
        if param.requires_grad:
            comm.all_reduce_sum(param.grad.data)


def train_for_one_epoch(our,
                        epoch: int,
                        graph: DGLHeteroGraph,
                        model: nn.Module,
                        input_data: Tensor,
                        labels: Tensor,
                        optimizer: Optimizer,
                        scaler,
                        criterion: Union[nn.Module, Any],
                        total_num_training_samples: int,
                        train_mask: Tensor,
                        reducer=False,
                        usecast=False):
    '''
    train for one epoch
    '''
    model.train()
    optimizer.zero_grad()
    # engine.ctx.timer.start('epoch')
    # 前向传播
    # with torch.profiler.record_function('train_model'):
    # engine.ctx.timer.start('forward')
    if usecast:
        with autocast(True, dtype=torch.float16):
            # with engine.ctx.timer.record('forward'):
            logits = model(graph, input_data)
            # 计算误差
            loss = criterion(logits[train_mask], labels[train_mask]) / total_num_training_samples
    else:
        # with engine.ctx.timer.record('forward'):
        logits = model(graph, input_data)
        # 计算误差
        loss = criterion(logits[train_mask], labels[train_mask]) / total_num_training_samples
    # engine.ctx.timer.stop('forward')

    # 反向传播
    # engine.ctx.timer.start('backward')
    # with engine.ctx.timer.record('backward'):
    if scaler: scaler.scale(loss).backward()
    else: loss.backward()
    # engine.ctx.timer.stop('backward')
    # 获取所有workers的梯`度并更新
    # engine.ctx.timer.start('reduce_grad')
    with engine.ctx.timer.record('reduce_grad'):
        # with torch.profiler.record_function('reduce_grad'):
        # torch.distributed.barrier()
        if reducer: engine.ctx.reducer.synchronize()
        else: average_gradients(model)
    # engine.ctx.timer.stop('reduce_grad')
    if scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    # engine.ctx.timer.stop('epoch')
    return loss

@torch.no_grad()
def val_test(graph: DGLHeteroGraph,
             model: nn.Module,
             input_data: Tensor,
             labels: Tensor,
             train_mask: Tensor,
             val_mask: Tensor,
             test_mask: Tensor,
             is_multilabel: bool = False):
    '''
    perform validation / test
    '''
    # inference
    model.eval()
    with autocast(False):
        logits = model(graph, input_data)
    metrics = []
    metrics.extend(get_metrics(labels[train_mask], logits[train_mask], is_multilabel))
    metrics.extend(get_metrics(labels[val_mask], logits[val_mask], is_multilabel))
    metrics.extend(get_metrics(labels[test_mask], logits[test_mask], is_multilabel))
    engine.ctx.timer.clear(is_train=False)
    return metrics

'''
*************************************************
*************** metric functions ****************
*************************************************
'''

def get_metrics(labels: Tensor, logits: Tensor, is_F1):
    '''
    get metrics for evaluation, use F1-score for multilabel classification tasks and accuracy for single label classification tasks
    '''
    if is_F1:
        # prepare for F1-score
        y_pred = (logits > 0)
        # get TP FP FN
        TP = torch.logical_and(y_pred == 1, labels == 1).float().sum()
        FP = torch.logical_and(y_pred == 1, labels == 0).float().sum()
        FN = torch.logical_and(y_pred == 0, labels == 1).float().sum()
        return [TP, TP + FP, TP + FN]
    else:
        # prepare for accuracy
        y_pred = torch.argmax(logits, dim = -1)
        is_label_correct = (y_pred == labels).float().sum()
        return [is_label_correct, labels.shape[0]]

def aggregate_accuracy(loss: Tensor, metrics: List[Union[float, int]], epoch: int) -> str:
    '''
    汇聚每个worker的metrics，以供评估
    '''
    metrics = torch.FloatTensor(metrics)
    comm.all_reduce_sum(metrics)
    (train_acc, val_acc, test_acc) =  \
    (metrics[0] / metrics[1],
     metrics[2] / metrics[3],
     metrics[4] / metrics[5])
    comm.all_reduce_sum(loss)
    epoch_metrics = [train_acc, val_acc, test_acc, loss.item()]
    engine.ctx.recorder.add_new_metrics(epoch, epoch_metrics)
    return epoch_metrics, loss.item(), f'Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc {train_acc * 100:.2f}% | Val Acc {val_acc * 100:.2f}% | Test Acc {test_acc * 100:.2f}%'

def aggregate_F1(loss: Tensor, metrics: List[Union[float, int]], epoch: int) -> str:
    '''
    aggregate metrics from each worker for evaluation
    '''
    def _safe_divide(numerator, denominator):
        if denominator == 0:
            denominator = 1
        return numerator / denominator
    metrics = torch.FloatTensor(metrics)
    comm.all_reduce_sum(metrics)
    # calculate precision and recall
    (train_precision, val_precision, test_precision) = \
        (_safe_divide(metrics[0], metrics[1]),
         _safe_divide(metrics[3], metrics[4]),
         _safe_divide(metrics[6], metrics[7]))
    (train_recall, val_recall, test_recall) = \
        (_safe_divide(metrics[0], metrics[2]),
         _safe_divide(metrics[3], metrics[5]),
         _safe_divide(metrics[6], metrics[8]))
    comm.all_reduce_sum(loss)
    train_f1_micro = _safe_divide(2 * train_precision * train_recall, train_precision + train_recall)
    val_f1_micro = _safe_divide(2 * val_precision * val_recall, val_precision + val_recall)
    test_f1_micro = _safe_divide(2 * test_precision * test_recall, test_precision + test_recall)
    epoch_metrics = [train_f1_micro, val_f1_micro, test_f1_micro, loss.item()]
    engine.ctx.recorder.add_new_metrics(epoch, epoch_metrics)
    return epoch_metrics, loss.item(), f'Epoch {epoch:05d} | Loss {loss.item():.4f} | Train F1 micro {train_f1_micro * 100:.2f}% | Val F1 micro {val_f1_micro * 100:.2f}% | Test F1 micro {test_f1_micro * 100:.2f}%'






