# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later, synchronize
from fcos_core.utils.metric_logger import MetricLogger
from fcos_core.engine.inference import inference

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

import os
import random
def do_train_fed(
    cfg,
    models,
    data_loaders,
    data_loader_vals,
    optimizers,
    schedulers,
    checkpointers,
    device,
    checkpoint_period,
    arguments_fed,
    server_model,
    client_weights,
):
    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    assert len(models) == len(optimizers) == len(schedulers) == len(data_loaders) == len(data_loader_vals)

    # Initialize fed rounds
    rounds = cfg.SOLVER.CONNECT_ROUNDS
    if cfg.SOLVER.METHOD == 'ori':
        for round in range(rounds):
            for cidx in range(len(arguments_fed)): # Refresh the iterations to 0 for each round
                arguments_fed[cidx]["iteration"] = 0
            for cidx in range(len(models)):
                start_iter = arguments_fed[cidx]["iteration"]
                # Initialize
                model = models[cidx]
                optimizer = optimizers[cidx]
                scheduler = schedulers[cidx]
                data_loader = data_loaders[cidx]
                data_loader_val = data_loader_vals[cidx]
                max_iter = len(data_loader)
                save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TRAIN[cidx])
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                gnet = model['gnet']
                pnet = model['pnet']
                gnet.train()
                pnet.train()
                start_training_time = time.time()
                end = time.time()
                pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
                for iteration, (images, targets, _, _) in enumerate(data_loader, start_iter):
                    data_time = time.time() - end
                    iteration = iteration + 1
                    arguments_fed[cidx]["iteration"] = iteration

                    # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
                    if not pytorch_1_1_0_or_later:
                        scheduler["gnet"].step()
                        scheduler["pnet"].step()
                    images = images.to(device)
                    targets = [target.to(device) for target in targets]

                    features = gnet(images.tensors)
                    _, loss_dict = pnet(images, features, targets=targets)
                    losses = sum(loss for loss in loss_dict.values())

                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters.update(loss=losses_reduced, **loss_dict_reduced)
                    
                    optimizer["gnet"].zero_grad()
                    optimizer["pnet"].zero_grad()
                    losses.backward()
                    # for k in optimizer:
                    optimizer["gnet"].step()
                    optimizer["pnet"].step()

                    if pytorch_1_1_0_or_later:
                        scheduler["gnet"].step()
                        scheduler["pnet"].step()

                    batch_time = time.time() - end
                    end = time.time()
                    meters.update(time=batch_time, data=data_time)

                    eta_seconds = meters.time.global_avg * (max_iter - iteration)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))


                    if iteration % 100 == 0 or iteration == max_iter:
                        logger.info(
                            meters.delimiter.join(
                                [
                                    "round: {round}",
                                    "client: {client}",
                                    "eta: {eta}",
                                    "iter: {iter}",
                                    "{meters}",
                                    "lr: {lr:.6f}",
                                    "max mem: {memory:.0f}",
                                ]
                            ).format(
                                round=str(round),
                                client=str(cidx),
                                eta=eta_string,
                                iter=iteration,
                                meters=str(meters),
                                lr=optimizer['gnet'].param_groups[0]["lr"],
                                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            )
                        )
                    if iteration == max_iter:
                        checkpointers[cidx].save("model_client{}".format(cidx), **arguments_fed[cidx])
                    # m = False
                    # if m:
                    if data_loader_val is not None and iteration == max_iter:
                        synchronize()
                        _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                            model,
                            data_loader_val,
                            dataset_name=cfg.DATASETS.TEST[cidx],
                            iou_types=("bbox",),
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=save_dir,
                        )
                        # checkpointers[cidx].save("model_client{}_round{}".format(cidx, round), **arguments_fed[cidx])
                        if os.path.exists(os.path.join(save_dir, "result.txt")):
                            with open(os.path.join(save_dir, "result.txt"), "r") as f:
                                cur_map = float(f.readline()[5:]) # mAP: 
                        logger.info('+-----------------------------+')
                        logger.info('| CLIENT {} MODEL UPDATED after round {}  |'.format(cidx, round))
                        logger.info('|     mAP: {}           |'.format(cur_map))
                        logger.info('+-----------------------------+')
                        synchronize()


                total_training_time = time.time() - start_training_time
                total_time_str = str(datetime.timedelta(seconds=total_training_time))
                logger.info(
                    "Total training time: {} ({:.4f} s / it)".format(
                        total_time_str, total_training_time / (max_iter)
                    )
                )

            ##################################################################################
            models = do_connect(server_model, models, client_weights, cfg.SOLVER.CONNECT_METHOD, logger)
            ##################################################################################
            if cfg.SOLVER.CONNECT_METHOD:
                for cidx in range(len(models)):
                    model = models[cidx]
                    data_loader_val = data_loader_vals[cidx]
                    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TRAIN[cidx])
                    if data_loader_val is not None and iteration == max_iter:
                        synchronize()
                        with open(os.path.join(save_dir, "result.txt"), "r") as f:
                            prev_map = float(f.readline()[5:]) # mAP: 
                        _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                            model,
                            data_loader_val,
                            dataset_name=cfg.DATASETS.TEST[cidx],
                            iou_types=("bbox",),
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=save_dir,
                        )
                        with open(os.path.join(save_dir, "result.txt"), "r") as f:
                            cur_map = float(f.readline()[5:]) # mAP: 
                        # if prev_map < cur_map:
                        #     logger.info('^0^ Federated Learning Has Improved The Performance At Client {} After Round {}'.format(cidx, round))
                        # else:
                        #     logger.info('TAT Federated Learning Has Deteriorate The Performance At Client {} After Round {}'.format(cidx, round))
                        checkpointers[cidx].save("model_client_agg{}".format(cidx), **arguments_fed[cidx])
                        logger.info('+-----------------------------+')
                        logger.info('| ROUND {} SERVER AGG FOR: CLIENT {}   |'.format(round, cidx))
                        logger.info('|      mAP: {} --> {}         |'.format(prev_map, cur_map))
                        logger.info('+-----------------------------+')
                        with open(os.path.join(save_dir, "result_change.txt"), "a+") as f:
                            f.write("round"+str(round)+":"+"\t"+str(prev_map)+"\t"+str(cur_map)+"\n") 
                        synchronize()
    ##########################################################
    #                                                        #
    #                         Ours                           #
    #                                                        #
    ##########################################################
    elif cfg.SOLVER.METHOD == 'att':
        logger.info("Using ATT (Extra Attention Layer) to train")
        for round in range(rounds):
            for cidx in range(len(arguments_fed)): # Refresh the iterations to 0 for each round
                arguments_fed[cidx]["iteration"] = 0
            for cidx in range(len(models)):
                start_iter = arguments_fed[cidx]["iteration"]
                # Initialize
                model = models[cidx]
                optimizer = optimizers[cidx]
                scheduler = schedulers[cidx]
                data_loader = data_loaders[cidx]
                data_loader_val = data_loader_vals[cidx]
                max_iter = len(data_loader)
                save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TRAIN[cidx])
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)

                gnet = model['gnet']
                pnet = model['pnet']
                attnet = server_model["attnet"]
                gnet.train()
                pnet.train()
                attnet.train()
                start_training_time = time.time()
                end = time.time()
                pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
                for iteration, (images, targets, _, _) in enumerate(data_loader, start_iter):
                    data_time = time.time() - end
                    iteration = iteration + 1
                    arguments_fed[cidx]["iteration"] = iteration

                    # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
                    if not pytorch_1_1_0_or_later:
                        scheduler["gnet"].step()
                        scheduler["pnet"].step()
                        if cidx == len(models) - 1:
                            scheduler["attnet"].step()
                    images = images.to(device)
                    targets = [target.to(device) for target in targets]

                    features = gnet(images.tensors)
                    _, loss_dict = pnet(images, features, targets=targets)

                    # deconfound
                    sgnet = server_model["gnet"]
                    sgnet.train()
                    
                    features_ser = sgnet(images.tensors)
                    features_l = list(features)
                    features_ser_l = list(features_ser)
                    for i in range(len(features_ser_l)):
                        if cfg.SOLVER.FEATURE_SHUFFLE:
                            features_ser_shuffled = torch.stack((features_ser_l[i][1], features_ser_l[i][0]))
                        else:
                            features_ser_shuffled = features_ser_l[i]
                        weight = attnet(features[i], features_ser_shuffled)
                        if cfg.SOLVER.COSINE_WEIGHT:
                            lvl_sim = torch.nn.CosineSimilarity()(features[i], features_ser[i])
                            lvl_sim = lvl_sim/(torch.sum(lvl_sim))
                            weight = weight + lvl_sim.unsqueeze(1)
                            weight = torch.clamp(weight, max=1.)
                        features_l[i] = weight * features[i] + (1 - weight) * features_ser_shuffled.clone().detach()
                    features_l = tuple(features_l)
                    # features_ser = tuple(features_ser)
                    del features_ser, features_ser_shuffled

                    _, loss_dict_ = pnet(images, features_l, targets=targets)
                    loss_total_dict = {k: v + loss_dict_[k] for k, v in loss_dict.items()}

                    losses_total = sum(loss for loss in loss_total_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_total_dict)
                    losses_total_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters.update(loss=losses_total_reduced, **loss_dict_reduced)
                    
                    optimizer["gnet"].zero_grad()
                    optimizer["pnet"].zero_grad()
                    if cidx == len(models) - 1:
                        optimizer["attnet"].zero_grad()
                    losses_total.backward()
                    optimizer["gnet"].step()
                    optimizer["pnet"].step()
                    if cidx == len(models) - 1:
                        optimizer["attnet"].step()

                    if pytorch_1_1_0_or_later:
                        scheduler["gnet"].step()
                        scheduler["pnet"].step()
                        if cidx == len(models) - 1:
                            scheduler["attnet"].step()

                    batch_time = time.time() - end
                    end = time.time()
                    meters.update(time=batch_time, data=data_time)

                    eta_seconds = meters.time.global_avg * (max_iter - iteration)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))


                    if iteration % 100 == 0 or iteration == max_iter:
                        logger.info(
                            meters.delimiter.join(
                                [
                                    "round: {round}",
                                    "client: {client}",
                                    "eta: {eta}",
                                    "iter: {iter}",
                                    "{meters}",
                                    "lr: {lr:.6f}",
                                    "max mem: {memory:.0f}",
                                ]
                            ).format(
                                round=str(round),
                                client=str(cidx),
                                eta=eta_string,
                                iter=iteration,
                                meters=str(meters),
                                lr=optimizer['gnet'].param_groups[0]["lr"],
                                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                            )
                        )
                    if iteration == max_iter:
                        checkpointers[cidx].save("model_client{}".format(cidx), **arguments_fed[cidx])
                    # m = False
                    # if m:
                    if data_loader_val is not None and iteration == max_iter:
                        synchronize()
                        _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                            model,
                            data_loader_val,
                            dataset_name=cfg.DATASETS.TEST[cidx],
                            iou_types=("bbox",),
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=save_dir,
                        )
                        # checkpointers[cidx].save("model_client{}_round{}".format(cidx, round), **arguments_fed[cidx])
                        if os.path.exists(os.path.join(save_dir, "result.txt")):
                            with open(os.path.join(save_dir, "result.txt"), "r") as f:
                                cur_map = float(f.readline()[5:]) # mAP: 
                        logger.info('+-----------------------------+')
                        logger.info('| CLIENT {} MODEL UPDATED after round {}  |'.format(cidx, round))
                        logger.info('|     mAP: {}           |'.format(cur_map))
                        logger.info('+-----------------------------+')
                        synchronize()


                total_training_time = time.time() - start_training_time
                total_time_str = str(datetime.timedelta(seconds=total_training_time))
                logger.info(
                    "Total training time: {} ({:.4f} s / it)".format(
                        total_time_str, total_training_time / (max_iter)
                    )
                )

            ##################################################################################
            models = do_connect(server_model, models, client_weights, cfg.SOLVER.CONNECT_METHOD, logger)
            ##################################################################################
            if cfg.SOLVER.CONNECT_METHOD:
                for cidx in range(len(models)):
                    model = models[cidx]
                    data_loader_val = data_loader_vals[cidx]
                    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASETS.TRAIN[cidx])
                    if data_loader_val is not None and iteration == max_iter:
                        synchronize()
                        with open(os.path.join(save_dir, "result.txt"), "r") as f:
                            prev_map = float(f.readline()[5:]) # mAP: 
                        _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                            model,
                            data_loader_val,
                            dataset_name=cfg.DATASETS.TEST[cidx],
                            iou_types=("bbox",),
                            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                            device=cfg.MODEL.DEVICE,
                            expected_results=cfg.TEST.EXPECTED_RESULTS,
                            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                            output_folder=save_dir,
                        )
                        with open(os.path.join(save_dir, "result.txt"), "r") as f:
                            cur_map = float(f.readline()[5:]) # mAP: 
                        # if prev_map < cur_map:
                        #     logger.info('^0^ Federated Learning Has Improved The Performance At Client {} After Round {}'.format(cidx, round))
                        # else:
                        #     logger.info('TAT Federated Learning Has Deteriorate The Performance At Client {} After Round {}'.format(cidx, round))
                        checkpointers[cidx].save("model_client_agg{}".format(cidx), **arguments_fed[cidx])
                        logger.info('+-----------------------------+')
                        logger.info('| ROUND {} SERVER AGG FOR: CLIENT {}   |'.format(round, cidx))
                        logger.info('|      mAP: {} --> {}         |'.format(prev_map, cur_map))
                        logger.info('+-----------------------------+')
                        with open(os.path.join(save_dir, "result_change.txt"), "a+") as f:
                            f.write("round"+str(round)+":"+"\t"+str(prev_map)+"\t"+str(cur_map)+"\n") 
                        synchronize()

def do_connect(server_model, models, client_weights, method=None, logger=None):
    for model in models:
        model['gnet'].eval()
        model['pnet'].eval()

    if method == 'fedavg':
        logger.info("Using {FedAvg} For Federated Connection.")
        with torch.no_grad():
            for nc in server_model.keys():
                for key in server_model[nc].state_dict().keys():
                    # num_batches_tracked is a non trainable LongTensor and
                    # num_batches_tracked are the same for all clients for the given datasets
                    if 'num_batches_tracked' in key:
                        server_model[nc].state_dict()[key].data.copy_(models[0][nc].state_dict()[key])
                    else:
                        temp = torch.zeros_like(server_model[nc].state_dict()[key])
                        for client_idx in range(len(client_weights)):
                            temp += client_weights[client_idx] * models[client_idx][nc].state_dict()[key]
                        server_model[nc].state_dict()[key].data.copy_(temp)
                        for client_idx in range(len(client_weights)):
                            models[client_idx][nc].state_dict()[key].data.copy_(server_model[nc].state_dict()[key])
    else:
        logger.info("No Federated Connection. Pass")
    return models
