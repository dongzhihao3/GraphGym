import torch
import time
import logging

from graphgym.config import cfg
from graphgym.loss import compute_loss, compute_loss_Tfg
from graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt



import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_geometric.utils import tf_utils
import tensorflow as tf
import tf_geometric as tfg
from tqdm import tqdm
import time
from tf_geometric.data.graph import Graph




def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()




def train_epoch_Tfg(logger, loader, model, optimizer, scheduler):
    loss_s = 0
    time_start = time.time()
    for batch in loader:
        graph = Graph(x=batch.node_feature.numpy(), edge_index=batch.edge_index.numpy(), y=batch.node_label.numpy())
        with tf.GradientTape() as tape:

            logits = model([graph.x, graph.edge_index, graph.edge_weight], training=True)
            #print(logits)
            loss = compute_loss_Tfg(logits, batch.node_label_index, batch.node_label, tape.watched_variables())
            vars = tape.watched_variables()
            grads = tape.gradient(loss, vars)
            optimizer.apply_gradients(zip(grads, vars))
        logger.update_stats(loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()






@torch.no_grad()
def eval_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        logger.update_stats(true=true.detach().cpu(),
                            pred=pred_score.detach().cpu(),
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()






def train(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        if not cfg.dataset.format[:3] == 'Tfg':
            train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        else:
            train_epoch_Tfg(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model)
                loggers[i].write_epoch(cur_epoch)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.out_dir))
