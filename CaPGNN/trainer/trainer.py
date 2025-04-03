import os
import csv
import time
from multiprocessing import Event
from multiprocessing.pool import ThreadPool
import pandas as pd
import yaml
import torch
from argparse import Namespace
from typing import Dict, Tuple
from tqdm import tqdm
from .runtime_util import *
from ..helper import DistGNNType, BitType
from ..model import DistGCN, DistSAGE
from ..communicator import Communicator as comm
from ..manager import GraphEngine as engine
from ..util.timer import TimerKeys
from tqdm import tqdm
import swanlab
from torch.cuda.amp import GradScaler

MODEL_MAP: Dict[str, DistGNNType] = {'gcn': DistGNNType.DistGCN, 'sage': DistGNNType.DistSAGE}


class Trainer(object):
    def __init__(self, runtime_args: Namespace, storage_server):
        # set runtime config
        runtime_args = vars(runtime_args)
        dataset = runtime_args['dataset']
        # load offline config
        current_path = os.path.dirname(__file__)
        # 这是自定义配置文件
        offline_config_path = os.path.join(os.path.dirname(current_path), 'config', f'{dataset}.yaml')
        offline_config = yaml.load(open(offline_config_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
        # 这是公共配置文件
        offline_baseconfig_path = os.path.join(os.path.dirname(current_path), 'config', 'base.yaml')
        offline_baseconfig = yaml.load(open(offline_baseconfig_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

        # 根据运行时设置更新公共配置文件
        offline_baseconfig['runtime'].update(runtime_args)
        # 将自定义配置文件中的值，合并或覆盖过来
        for key in offline_config:
            if key in offline_baseconfig:
                offline_baseconfig[key].update(offline_config[key])
            else:
                offline_baseconfig[key] = offline_config[key]
        # set config
        self.config = offline_baseconfig
        runtime_config = self.config['runtime']
        # 实验结果的输出路径
        exp_path = runtime_config['exp_path']
        num_parts = runtime_config['num_parts']
        model_name = runtime_config['model_name']
        log_level = runtime_config['logger_level']

        if runtime_args['our_cache'] and runtime_args['our_partition']:
            mark = 'our_cache_partition'
        elif runtime_args['our_cache']:
            mark = 'our_cache'
        elif runtime_args['our_partition']:
            mark = 'our_partition'
        else:
            mark = 'adaqp'
        if runtime_args['use_pipeline']:
            mark += '_pipe'
        # exp_path = f'{exp_path}/{dataset}/{num_parts}part/{model_name}/{mark}'
        exp_path = f'{exp_path}/{dataset}/{num_parts}part/{model_name}/{runtime_args["gpus"]}/{mark}'
        self._n_layers = runtime_config["n_layers"]
        self.cache_size = runtime_config['gcache_size']
        # set up logger
        self.logger = setup_logger(f'trainer.log', log_level, with_file=True)
        # set up communicator
        self._set_communicator()
        # set up graph engine
        self._set_engine(runtime_args, storage_server)
        # set exp_path
        if not os.path.exists(exp_path) and comm.get_rank() == 0:
            os.makedirs(exp_path)
        self.exp_path = exp_path
        # set up comm buffer
        self._set_buffer()
        # set up model
        self._set_model()
        print('>> Trainer init complete!')

    '''
    *************************************************
    ***************** setup methods *****************
    *************************************************
    '''

    def _set_communicator(self):
        runtime_config = self.config['runtime']
        self.communicator = comm(runtime_config['backend'], runtime_config['init_method'])
        self.logger.info(repr(self.communicator))


    def _set_engine(self, runtime_args, storage_server):
        # fetch corresponding config
        data_config = self.config['data']
        runtime_config = self.config['runtime']
        model_config = self.config['model']
        msg_precision_type, use_parallel = ('full', False)
        if runtime_config['model_name'] not in MODEL_MAP:
            raise ValueError(f'Invalid model type: {model_config["model"]}')
        model_type = MODEL_MAP[runtime_config['model_name']]
        # setup
        self.engine = engine(runtime_args,
                             runtime_config['num_epoches'],
                             data_config['partition_path'],
                             runtime_config['dataset'],
                             msg_precision_type,
                             model_type,
                             storage_server,
                             use_parallel)
        engine.ctx.agg_type = model_config['aggregator_type']
        self.logger.info(repr(self.engine))


    def _set_buffer(self):
        # fetch corresponding config
        data_config = self.config['data']
        model_config = self.config['model']
        buffer_shape = torch.zeros(model_config['num_layers'], dtype=torch.int32)
        buffer_shape[0] = data_config['num_feats']
        buffer_shape[1:] = model_config['hidden_dim']
        # setup
        comm.ctx.init_buffer(buffer_shape.tolist(), engine.ctx.send_idx, engine.ctx.recv_idx,
                             engine.ctx.bit_type, self.cache_size, self.engine.boundary, self.engine.recv_shape,)


    def _set_model(self):
        # fetch corresponding config
        data_config = self.config['data']
        model_config = self.config['model']
        runtime_config = self.config['runtime']
        if runtime_config['model_name'] not in MODEL_MAP:
            raise ValueError(f'Invalid model type: {model_config["model"]}')
        model_type = MODEL_MAP[runtime_config['model_name']]
        if model_type == DistGNNType.DistGCN:
            self.model = DistGCN(data_config['num_feats'], model_config['hidden_dim'], data_config['num_classes'], model_config['num_layers'], model_config['dropout_rate'], model_config['use_norm']).to(comm.ctx.device)
        elif model_type == DistGNNType.DistSAGE:
            self.model = DistSAGE(data_config['num_feats'], model_config['hidden_dim'], data_config['num_classes'], model_config['num_layers'], model_config['dropout_rate'], model_config['use_norm'], model_config['aggregator_type']).to(comm.ctx.device)
        else:
            raise ValueError(f'Invalid model type: {model_type}')
        # self.model = torch.compile(self.model)


    '''
    *************************************************
    **************** runtime methods ****************
    *************************************************
    '''

    def reduce_hook(self, param, name, n_train):
        def fn(grad): self.engine.ctx.reducer.reduce(param, name, grad, n_train)
        return fn

    def train(self, rank):
        # fetch needed config
        runtime_config = self.config['runtime']
        is_multilabel = self.config['data']['is_multilabel']
        # 同步所有workers的随机种子
        sync_seed()
        self.model.reset_parameters()
        sync_model(self.model)

        if runtime_config["reducer"]:
            self.engine.ctx.reducer.init(self.model)
            # 为每个参数注册一个hook函数，在反向传播时被调用。允许在梯度被计算后、被优化器优化之前，对梯度进行操作
            for i, (name, param) in enumerate(self.model.named_parameters()):
                param.register_hook(self.reduce_hook(param, name, runtime_config["num_parts"]))

        # 获取训练所需的参数
        epoches = runtime_config['num_epoches']
        input_data = self.engine.ctx.feats
        labels = self.engine.ctx.labels
        train_mask = self.engine.ctx.train_mask
        val_mask = self.engine.ctx.val_mask
        test_mask = self.engine.ctx.test_mask
        # 获得总的节点数
        total_number_nodes = torch.LongTensor([train_mask.numel()])
        comm.all_reduce_sum(total_number_nodes)
        # 获取reduce后的结果
        total_number_nodes = total_number_nodes.item()
        # 设置优化器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=runtime_config['learning_rate'], weight_decay=runtime_config['weight_decay'], amsgrad=True)
        scheduler = None if not runtime_config["scheduler"] else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, min_lr=1e-6, verbose=True)

        scaler = None if not runtime_config["scaler"] else GradScaler()
        # 设置损失函数
        if is_multilabel:
            criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
            labels = labels.float()
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        prof = None
        # with torch.profiler.profile(with_stack=True,
        #                             activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA, ],
        #                             on_trace_ready=torch.profiler.tensorboard_trace_handler("./logs/trace")) as prof:
        # self.model.train()
        our = self.config['runtime']['our_cache'] or self.config['runtime']['our_partition']
        if runtime_config['pretrain']:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            engine.ctx.timer.pretrain = True
            for epoch in tqdm(range(0, 20), desc=f'warmup-train', ncols=100):
                train_for_one_epoch(our, epoch, engine.ctx.graph,
                                    self.model, input_data, labels,
                                    optimizer, scaler, criterion,
                                    total_number_nodes, train_mask,
                                    self.config['runtime']['reducer'])
                engine.ctx.curr_epoch += 1
            torch.cuda.synchronize()
            torch.distributed.barrier()
            engine.ctx.timer.pretrain = False
        torch.cuda.reset_peak_memory_stats()
        best_record = {"train_acc": 0,"val_acc": 0,"test_acc": 0,"loss": 1e6}
        for epoch in range(1, epoches + 1):
            # with torch.profiler.record_function('train_epoch'):
            # note: replace, instead of delete
            # if engine.ctx.our_cache: engine.ctx.storage_server.clear_cache(rank)
            engine.ctx.timer.is_train = True
            with engine.ctx.timer.record('epoch'):
                loss = train_for_one_epoch(our, epoch, engine.ctx.graph,
                                           self.model, input_data, labels,
                                           optimizer, scaler, criterion,
                                           total_number_nodes, train_mask,
                                           self.config['runtime']['reducer'],
                                           self.config['runtime']['usecast'])
            engine.ctx.timer.is_train = False
            # 验证和测试，可以在不需要计时的时候取消
            if self.config['runtime']['eval']:
                epoch_metrics = val_test(engine.ctx.graph, self.model, input_data, labels, train_mask, val_mask, test_mask, is_multilabel)
                metrics_val, loss_val, metrics_info = aggregate_accuracy(loss, epoch_metrics, epoch) if not is_multilabel else aggregate_F1(loss, epoch_metrics, epoch)
                if scheduler:
                    scheduler.step(metrics_val[1])
                    last_lr = scheduler.state_dict()["_last_lr"][0]
                    metrics_info += f' | _last_lr: {scheduler.state_dict()["_last_lr"][0]:.6f}'
                if runtime_config["swanlab"] and comm.get_rank()==0:
                    log_info = {f"train_acc_{rank}": metrics_val[0], f"val_acc_{rank}": metrics_val[1], f"test_acc_{rank}": metrics_val[2], f"loss_{rank}": loss_val}
                    if scheduler: log_info.update({f"last_lr_{rank}": last_lr})
                    swanlab.log(log_info)
                best_record["train_acc"] = max(best_record["train_acc"], metrics_val[0])
                best_record["val_acc"] = max(best_record["val_acc"], metrics_val[1])
                best_record["test_acc"] = max(best_record["test_acc"], metrics_val[2])
                best_record["loss"] = min(best_record["loss"], loss_val)
                # 打印训练信息
                if epoch % runtime_config['log_steps']==0:
                    if comm.get_rank() == 0:
                        print(f'{epoch}/{epoches}')
                        print(metrics_info)
                        print(best_record)
                    comm.barrier()
            else:
                if epoch % runtime_config['log_steps'] == 0 and comm.get_rank() == 0:
                    print(f'{epoch}/{epoches}')
            #         if runtime_config["swanlab"]: swanlab.log({f"loss_{rank}": loss.item()})
            #         print(f'Loss {loss.item():.4f}')
            # prof.step()
            # engine.ctx.timer.is_train = True
            # torch.cuda.synchronize() # 测cache hit时候才打开！!!!!!!!!!!!!
            # comm.barrier()           # 测cache hit时候才打开！!!!!!!!!!!!!
            # if comm.get_rank() == 0: print('#' * 50)

        forward_time = engine.ctx.timer.peak_item('forward', TimerKeys.TOTAL) or 0
        check_cache_time = engine.ctx.timer.peak_item('check_cache', TimerKeys.TOTAL) or 0
        pick_cache_time = engine.ctx.timer.peak_item('pick_cache', TimerKeys.TOTAL) or 0
        backward_time = engine.ctx.timer.peak_item('backward', TimerKeys.TOTAL) or 0
        epoch_time = engine.ctx.timer.peak_item('epoch', TimerKeys.TOTAL) or 0
        reduce_time = engine.ctx.timer.peak_item('reduce_grad', TimerKeys.TOTAL) or 0
        msg_comm = engine.ctx.timer.peak_item('msg_comm', TimerKeys.TOTAL) or 0
        mm = engine.ctx.timer.peak_item('mm', TimerKeys.TOTAL) or 0
        aggregation = engine.ctx.timer.peak_item('aggregation', TimerKeys.TOTAL) or 0
        cpu_time_ave = cpu_time_total = gpu_time_ave = gpu_time_total = 0
        # if comm.get_rank() == 0:
        # 遍历性能分析结果列表，找到名称为'train_epoch'的记录对应的条目
        if prof:
            prof_results = prof.key_averages()
            cpu_time_ave = cpu_time_total = gpu_time_ave = gpu_time_total = 0
            for item in prof_results:
                if item.key == 'train_epoch':
                    cpu_time_ave = item.cpu_time / 1e6
                    cpu_time_total = item.cpu_time_total / 1e6
                    gpu_time_ave = item.cuda_time / 1e6
                    gpu_time_total = item.cuda_time_total / 1e6
                    break
            print(comm.get_rank(), prof_results.table(sort_by='cuda_time_total', row_limit=15))

        total_time_records = torch.tensor([epoch_time, forward_time, backward_time,
                                           check_cache_time, pick_cache_time, reduce_time,
                                           msg_comm, mm, aggregation, cpu_time_ave,
                                           cpu_time_total, gpu_time_ave, gpu_time_total])
        #comm.ctx.delete_buffer()
        if rank == 0: engine.ctx.timer.pretty_show_store()
        return total_time_records

    def save_csv(self, time_records: Tensor):
        if comm.get_rank() == 0:
            # gather time records
            obj_list = [None] * comm.get_world_size()
            comm.gather_any(time_records, obj_list, dst=0)
            # set up path
            save_path = self.exp_path
            metrics_path = f'{save_path}/metrics'
            time_path = f'{save_path}/time'
            val_curve_path = f'{save_path}/val_curve'
            if not os.path.exists(metrics_path):
                os.makedirs(metrics_path)
            if not os.path.exists(time_path):
                os.makedirs(time_path)
            if not os.path.exists(val_curve_path):
                os.makedirs(val_curve_path)
            name = self.config['runtime']['mode'] + '_' + str(self.config['runtime']['num_parts'])
            # save metrics and val_curve
            engine.ctx.recorder.display_final_statistics(f'{metrics_path}/{name}.txt', f'{val_curve_path}/{name}.pt', self.config['runtime']['model_name'])
            # save time
            set_title = True if not os.path.exists(f'{time_path}/{name}.csv') else False
            with open(f'{time_path}/{name}.csv', 'a+', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                if set_title:
                    # total_train_time, total_reduce_time, ave_train_time, ave_reduce_time
                    writer.writerow(['Worker', 'GPU', 'total_epoch', 'forward', 'backward',
                                     'check_cache', 'pick_cache', 'reduce', 'msg_comm',
                                     'mm', 'aggregation', 'cpu_time_ave', 'cpu_time_total',
                                     'gpu_time_ave', 'gpu_time_total'])
                for worker in tqdm(range(comm.get_world_size()), desc='写入csv中...'):
                    device_name = torch.cuda.get_device_name(worker)
                    write_data = [f'Worker {worker}', device_name]
                    write_data.extend(obj_list[worker].numpy())
                    writer.writerow(write_data)
            comm.barrier()
        else:
            comm.gather_any(time_records, [], dst=0)
            comm.barrier()

    def save_excel(self, time_records: Tensor):
        if comm.get_rank() == 0:
            # gather time records
            obj_list = [None] * comm.get_world_size()
            comm.gather_any(time_records, obj_list, dst=0)
            # set up path
            save_path = self.exp_path
            metrics_path = f'{save_path}/metrics'
            time_path = f'{save_path}/time'
            val_curve_path = f'{save_path}/val_curve'
            if not os.path.exists(metrics_path):
                os.makedirs(metrics_path)
            if not os.path.exists(time_path):
                os.makedirs(time_path)
            if not os.path.exists(val_curve_path):
                os.makedirs(val_curve_path)
            name = self.config['runtime']['mode'] + '_' + str(self.config['runtime']['num_parts'])
            # save metrics and val_curve
            engine.ctx.recorder.display_final_statistics(f'{metrics_path}/{name}.txt', f'{val_curve_path}/{name}.pt',
                                                         self.config['runtime']['model_name'])
            # save time
            excel_path = f'{time_path}/{name}.xlsx'
            data = []
            for worker in tqdm(range(comm.get_world_size()), desc='Writing to Excel...'):
                device_name = torch.cuda.get_device_name(worker)
                write_data = [f'Worker {worker}', device_name]
                write_data.extend(obj_list[worker].numpy())
                data.append(write_data)
            columns = ['Worker', 'GPU', 'total_epoch', 'forward', 'backward', 'check_cache', 'pick_cache',
                       'reduce', 'msg_comm', 'mm', 'aggregation', 'cpu_time_ave',
                       'cpu_time_total', 'gpu_time_ave', 'gpu_time_total']
            df = pd.DataFrame(data, columns=columns)
            if not os.path.exists(excel_path):
                df.to_excel(excel_path, index=False)
            else:
                with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='overlay', engine_kwargs={'encoding': 'utf-8'}) as writer:
                    df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
            comm.barrier()
            # 打印平均值
            print(df.drop(columns=['Worker', 'GPU']).mean())
        else:
            comm.gather_any(time_records, [], dst=0)
            comm.barrier()

    def save(self, time_records: Tensor, suffix='xlsx'):
        if suffix.lower() == 'xlsx':
            self.save_excel(time_records)
        elif suffix.lower() == 'csv':
            self.save_csv(time_records)
        else:
            print(f'>> {suffix} not support <<')

