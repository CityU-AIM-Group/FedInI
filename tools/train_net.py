# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader, make_federated_data_loader
from fcos_core.solver import make_lr_scheduler
from fcos_core.solver import make_optimizer, make_cdr_optimizer
from fcos_core.engine.inference import inference
from fcos_core.engine.trainer import do_train, do_train_fed, do_connect
from fcos_core.modeling.detector import build_detection_model
from fcos_core.modeling.backbone import build_backbone
from fcos_core.modeling.rpn.rpn import build_rpn
from fcos_core.modeling.rpn.fcos.attention_layer import build_attention
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, \
    get_rank, is_pytorch_1_1_0_or_later
from fcos_core.utils.imports import import_file
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir

import copy

def train(cfg, local_rank, distributed, logger):
    device = torch.device(cfg.MODEL.DEVICE)
    if cfg.SOLVER.TRAIN_FED:
        gnet = build_backbone(cfg).to(device)
        pnet = build_rpn(cfg, gnet.out_channels).to(device)
        if cfg.SOLVER.METHOD == "att":
            attnet = build_attention(cfg, gnet.out_channels).to(device)
            print(attnet)
    else:
        model = build_detection_model(cfg)

    assert cfg.MODEL.USE_SYNCBN is False

    if cfg.SOLVER.TRAIN_FED:
        server_gnet = gnet
        server_pnet = pnet
        client_num = len(cfg.DATASETS.TRAIN)
        client_weights = [1/client_num for i in range(client_num)]
        gnets = [copy.deepcopy(server_gnet).to(device) for idx in range(client_num)]
        pnets = [copy.deepcopy(server_pnet).to(device) for idx in range(client_num)]
        if cfg.SOLVER.METHOD == "cdr":
            optimizers_gnets = [make_cdr_optimizer(cfg, gnets[i]) for i in range(client_num)]
            optimizers_pnets = [make_cdr_optimizer(cfg, pnets[i]) for i in range(client_num)]
        else:
            optimizers_gnets = [make_optimizer(cfg, gnets[i]) for i in range(client_num)]
            optimizers_pnets = [make_optimizer(cfg, pnets[i]) for i in range(client_num)]
        schedulers_gnets = [make_lr_scheduler(cfg, optimizers_gnets[i]) for i in range(client_num)]
        schedulers_pnets = [make_lr_scheduler(cfg, optimizers_pnets[i]) for i in range(client_num)]
        if cfg.SOLVER.METHOD == "att":
            if cfg.SOLVER.METHOD == "cdr":
                optimizer_att = make_cdr_optimizer(cfg, attnet)
            else:
                optimizer_att = make_optimizer(cfg, attnet)
            scheduler_att = make_lr_scheduler(cfg, optimizer_att)
    else:
        if cfg.SOLVER.METHOD == "cdr":
            optimizer = make_cdr_optimizer(cfg, model)
        else:
            optimizer = make_optimizer(cfg, model)
        scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        if cfg.SOLVER.TRAIN_FED:
            server_gnet = torch.nn.parallel.DistributedDataParallel(
                server_gnet, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False,
            )
            server_pnet = torch.nn.parallel.DistributedDataParallel(
                server_pnet, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False,
            )
            gnets = [torch.nn.parallel.DistributedDataParallel(
                gnet, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False,
            ) for gnet in gnets]
            pnets = [torch.nn.parallel.DistributedDataParallel(
                pnet, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False,
            ) for pnet in pnets]
            if cfg.SOLVER.METHOD == "att":
                attnet = torch.nn.parallel.DistributedDataParallel(
                    attnet, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False,
                )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
            )
    

    ################## Make each model a dict, with two keys: gnet, pnet ###############################
    if cfg.SOLVER.TRAIN_FED:
        server_model = {}
        server_model['gnet'] = server_gnet
        server_model['pnet'] = server_pnet
        if cfg.SOLVER.METHOD == "att":
            server_model['attnet'] = attnet

        models = []
        for i in range(len(gnets)):
            models.append({})
            models[i]["gnet"] = gnets[i]
            models[i]["pnet"] = pnets[i]
            if cfg.SOLVER.METHOD == "att":
                models[i]['attnet'] = attnet

        optimizers = []
        for i in range(len(gnets)):
            optimizers.append({})
            optimizers[i]["gnet"] = optimizers_gnets[i]
            optimizers[i]["pnet"] = optimizers_pnets[i]
            if cfg.SOLVER.METHOD == "att":
                optimizers[i]['attnet'] = optimizer_att

        schedulers = []
        for i in range(len(gnets)):
            schedulers.append({})
            schedulers[i]["gnet"] = schedulers_gnets[i]
            schedulers[i]["pnet"] = schedulers_pnets[i]
            if cfg.SOLVER.METHOD == "att":
                schedulers[i]['attnet'] = scheduler_att

    if cfg.SOLVER.TRAIN_FED:
        arguments_fed = []
        for i in range(len(gnets)):
            arguments_fed.append({})
            arguments_fed[i]["iteration"] = 0
    else:
        arguments = {}
        arguments["iteration"] = 0

    ################## Saving options ###############################
    if cfg.SOLVER.TRAIN_FED:
        output_dirs = [os.path.join(cfg.OUTPUT_DIR, str(i)) for i in cfg.DATASETS.TRAIN]
    else:
        output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    if cfg.SOLVER.TRAIN_FED:
        checkpointers = [DetectronCheckpointer(
            cfg, models[i], optimizers[i], schedulers[i], output_dirs[i], save_to_disk
        ) for i in range(len(models))]
        for cidx, ckpt in enumerate(checkpointers):
            weight = cfg.MODEL.WEIGHT
            if not weight.endswith('.pth'):
                if not weight.endswith('.pkl'): # Only give the dir of all clients
                    extra_checkpoint_data = ckpt.load(os.path.join(weight, cfg.DATASETS.TRAIN[cidx], 'model_client_agg{}.pth'.format(cidx)))
                else: # Use R-50.pkl
                    extra_checkpoint_data = ckpt.load(weight)
            else:
                extra_checkpoint_data = ckpt.load(weight)
            arguments_fed[i].update(extra_checkpoint_data)
    else:
        checkpointer = DetectronCheckpointer(
            cfg, model, optimizer, scheduler, output_dir, save_to_disk
        )
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
        arguments.update(extra_checkpoint_data)

    ################## DATA LOADER ###############################
    if cfg.SOLVER.TRAIN_FED: # For federated setting, the DATASETS.TRAIN are not concatenated
        data_loaders = []
        for i in range(len(cfg.DATASETS.TRAIN)):
            data_loaders.append(make_federated_data_loader(
                cfg,
                is_train=True,
                is_distributed=distributed,
                start_iter=arguments_fed[i]["iteration"],
                client_index = i,
            ))
    else:
        data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
        )

    ################## TEST DURING TRAINING ###############################
    if cfg.SOLVER.TRAIN_FED:
        data_loader_vals = []
        for i in range(len(cfg.DATASETS.TEST)):
            data_loader_vals.append(make_federated_data_loader(cfg, is_train=False, is_distributed=distributed, 
                client_index = i, is_for_period=True))
    else:
        test_period = cfg.SOLVER.TEST_PERIOD
        if test_period > 0:
            data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
        else:
            data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if cfg.SOLVER.CONNECT_ONLY:
        models = do_connect(server_model, models, client_weights, method=cfg.SOLVER.CONNECT_METHOD, logger=logger)

    if cfg.SOLVER.TRAIN_FED:    
        if not cfg.SOLVER.TEST_ONLY:
            do_train_fed(cfg, models, data_loaders, data_loader_vals, optimizers, schedulers, \
                checkpointers, device, checkpoint_period, arguments_fed, server_model, client_weights)
        return models
    else:
        if not cfg.SOLVER.TEST_ONLY:
            do_train(cfg,model,data_loader,data_loader_val,optimizer,scheduler, \
                checkpointer,device,checkpoint_period,test_period,arguments)
        return model

def run_test(cfg, model, distributed):
    if cfg.SOLVER.TRAIN_FED:   
        if distributed:
            for m in model:
                m = m.module
        for cidx in range(len(cfg.DATASETS.TEST)):
            data_loader_val = make_federated_data_loader(cfg, is_train=False, is_distributed=distributed, client_index=cidx, is_for_period=True)
            print('start evaluating with images count: ', len(data_loader_val)*cfg.TEST.IMS_PER_BATCH)
            inference(
                model[cidx],
                data_loader_val,
                dataset_name=cfg.DATASETS.TEST[cidx],
                iou_types=("bbox",),
                box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            print('finish')
            synchronize()
    else:
        if distributed:
            model = model.module
        torch.cuda.empty_cache()  # TODO check if it helps
        iou_types = ("bbox",)
        if cfg.MODEL.MASK_ON:
            iou_types = iou_types + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            iou_types = iou_types + ("keypoints",)
        output_folders = [None] * len(cfg.DATASETS.TEST)
        dataset_names = cfg.DATASETS.TEST
        if cfg.OUTPUT_DIR:
            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
                mkdir(output_folder)
                output_folders[idx] = output_folder
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
        for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
            inference(
                model,
                data_loader_val,
                dataset_name=dataset_name,
                iou_types=iou_types,
                box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=output_folder,
            )
            synchronize()

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("fcos_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args.local_rank, args.distributed, logger)

    if not args.skip_test:
        run_test(cfg, model, args.distributed)


if __name__ == "__main__":
    main()
