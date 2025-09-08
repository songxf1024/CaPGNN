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
        self.server_num = runtime_args['server_num']
        # load offline config
        current_path = os.path.dirname(__file__)
        # This is a custom configuration file
        offline_config_path = os.path.join(os.path.dirname(current_path), 'config', f'{dataset}.yaml')
        offline_config = yaml.load(open(offline_config_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
        # This is a public configuration file
        offline_baseconfig_path = os.path.join(os.path.dirname(current_path), 'config', 'base.yaml')
        offline_baseconfig = yaml.load(open(offline_baseconfig_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)

        # Update public configuration files according to runtime settings
        offline_baseconfig['runtime'].update(runtime_args)
        # Merge or overwrite values in custom configuration files
        for key in offline_config:
            if key in offline_baseconfig:
                offline_baseconfig[key].update(offline_config[key])
            else:
                offline_baseconfig[key] = offline_config[key]
        # set config
        self.config = offline_baseconfig
        runtime_config = self.config['runtime']
        # The output path of the experimental results
        exp_path = runtime_config['exp_path']
        num_parts = runtime_config['num_parts']
        model_name = runtime_config['model_name']
        log_level = runtime_config['logger_level']

        if runtime_args['our_cache'] and runtime_args['our_partition']: mark = 'our_cache_partition'
        elif runtime_args['our_cache']: mark = 'our_cache'
        elif runtime_args['our_partition']: mark = 'our_partition'
        else: mark = 'adaqp'
        if runtime_args['use_pipeline']: mark += '_pipe'
        self._n_layers = runtime_config["n_layers"]
        self.cache_size = runtime_config['gcache_size']
        # set up logger
        self.logger = setup_logger(f'trainer.log', log_level, with_file=True)
        print('setup_logger done')
        # set up communicator
        self._set_communicator()
        print('_set_communicator done')
        # set up graph engine
        self._set_engine(runtime_args, storage_server)
        print('_set_engine done')

        # exp_path = f'{exp_path}/{dataset}/{num_parts}part/{model_name}/{mark}'
        exp_path = f'{exp_path}/{dataset}/{runtime_args["server_num"]}server/server{runtime_args["server_id"]}/{num_parts}part/{model_name}/{runtime_args["gpus"]}/{mark}'
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
        self.communicator = comm(runtime_config['backend'], runtime_config['init_method'], server_num=self.server_num)
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
                             runtime_args['part_dir'],
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
        sync_seed()
        self.model.reset_parameters()
        comm.ctx.hierarchical_barrier()
        sync_model(self.model)

        if runtime_config["reducer"]:
            self.engine.ctx.reducer.init(self.model)
            # Register a hook function for each parameter and is called when backpropagated. Allows operation on the gradient after it is calculated and before it is optimized by the optimizer.
            for i, (name, param) in enumerate(self.model.named_parameters()):
                param.register_hook(self.reduce_hook(param, name, comm.get_world_size()))

        # Get the parameters required for training
        epoches = runtime_config['num_epoches']
        input_data = self.engine.ctx.feats
        labels = self.engine.ctx.labels
        train_mask = self.engine.ctx.train_mask
        val_mask = self.engine.ctx.val_mask
        test_mask = self.engine.ctx.test_mask
        # Get the total number of nodes
        total_number_nodes = torch.tensor([train_mask.numel()], device=comm.ctx.device, dtype=torch.int64)
        comm.all_reduce_sum(total_number_nodes) # , group=comm.ctx.global_group
        total_number_nodes = total_number_nodes.item()
        if comm.get_rank() == 0: print(f"[Init] global_train_samples = {total_number_nodes}, train_mask={train_mask.shape}")
        # Setting up the optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=runtime_config['learning_rate'], weight_decay=runtime_config['weight_decay'], amsgrad=True)
        scheduler = None if not runtime_config["scheduler"] else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, min_lr=1e-6, verbose=True)

        scaler = None if not runtime_config["scaler"] else GradScaler()
        # Set the loss function
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
            comm.barrier()
            engine.ctx.timer.pretrain = True
            for epoch in tqdm(range(0, 20), desc=f'warmup-train', ncols=100):
                train_for_one_epoch(our, epoch, engine.ctx.graph,
                                    self.model, input_data, labels,
                                    optimizer, scaler, criterion,
                                    total_number_nodes, train_mask,
                                    self.config['runtime']['reducer'],
                                    self.config['runtime']['usecast'])
                engine.ctx.curr_epoch += 1
            engine.ctx.timer.pretrain = False
        torch.cuda.synchronize()
        comm.barrier()
        torch.cuda.reset_peak_memory_stats()
        best_record = {"train_acc": 0,"val_acc": 0,"test_acc": 0,"loss": 1e6}
        # the fixed staleness threshold, can be adjusted as needed, -1 indicating unbounded staleness
        STALENESS_THRESHOLD = -1
        staleness_counter = 0
        for epoch in range(1, epoches + 1):
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
            staleness_counter += 1
            need_sync = (STALENESS_THRESHOLD > 0) and (staleness_counter >= STALENESS_THRESHOLD)
            # Verification and testing can be canceled when no timer is required
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
                # Print training information
                if epoch % runtime_config['log_steps']==0:
                    if comm.ctx.local_rank == 0:
                        print(f'{epoch}/{epoches}')
                        print(metrics_info)
                        print(best_record)
                    comm.barrier()
            else:
                if epoch % runtime_config['log_steps'] == 0 and comm.ctx.local_rank == 0:
                    print(f'{epoch}/{epoches}')
            #         if runtime_config["swanlab"]: swanlab.log({f"loss_{rank}": loss.item()})
            #         print(f'Loss {loss.item():.4f}')
            # prof.step()
            # engine.ctx.timer.is_train = True
            # torch.cuda.synchronize() # only open when test the cache hit!
            # comm.barrier()           # only open when test the cache hit!
            # if comm.get_rank() == 0: print('#' * 50)

            # perform a forced synchronization operation
            if need_sync:
                if rank==0: print(f"start all sync")
                # engine.ctx.storage_server.clear_cache(rank)
                engine.ctx.storage_server.cache_server.sync_all_procs()
                staleness_counter = 0

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
        # Iterate through the performance analysis results list and find the entry corresponding to the record named 'train_epoch'
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
                    # device_name = torch.cuda.get_device_name(worker)
                    device_name = worker
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

