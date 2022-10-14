# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
from tkinter import image_names
import numpy as np

import torch
from tqdm import tqdm

from fcos_core.config import cfg
from fcos_core.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug
from fcos_core.structures.image_list import to_image_list

DATASET_COUNT = 1
def _foward_detector(model, images, targets=None, return_maps=False):
    generator = model["gnet"]
    predictor_module = model["pnet"] 
    images = to_image_list(images)
    features = generator(images.tensors)
    boxes, losses = predictor_module(images, features, targets=targets)
    return boxes

def compute_on_dataset(model, data_loader, device, timer=None):
    if isinstance(model, dict):
        for k in model:
            model[k].eval()
    else:
        model.eval()
    results_dict = {}
    # results_dict2 = {}
    # results2 = []
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(data_loader):
        images, targets, image_ids, img_names = batch
        # print(img_names)
        images = images.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            if cfg.TEST.BBOX_AUG.ENABLED:
                output = im_detect_bbox_aug(model, images, device)
            else:
                if isinstance(model, dict):
                    output = _foward_detector(model, images, targets=None, return_maps=False)
                else:
                    output = model(images.to(device))
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
            # print(output[0].get_field("scores"))
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
        # results_dict2.update(
        #     {img_name: result for img_name, result in zip(img_names, output)}
        # )
        # for result in output:
        #     lab = torch.max(result.get_field("labels")).item()
        #     results2.append(int(lab))
    # def takeSecond(elem):
    #     return elem[1]
    # # Get IDN noise samples:
    # idn = []
    # for img_id, result in results_dict2.items():
    #     lab = torch.max(result.get_field("labels")).item()
    #     idn.append((img_id, lab))
    # idn.sort(key=takeSecond)
    
    # global DATASET_COUNT
    # results2 = np.array(results2)
    # print(results2)
    # np.save("np_fedini/fedini_{}.npy".format(str(DATASET_COUNT)), results2)    # .npy extension is added if not given
    # print("saved to {}".format("np_fedini/fedini_{}.npy".format(str(DATASET_COUNT))))
    # d = np.load('test3.npy')
    # global DATASET_COUNT
    # if os.path.exists("fedavg_{}.txt".format(str(DATASET_COUNT)):
    #     os.remove("fedavg_{}.txt".format(str(DATASET_COUNT))
    # for p in idn:
    #     with open("fedavg_{}.txt".format(str(DATASET_COUNT)), "a+") as f:
    #         f.write(str(p[0])+"\t"+str(p[1])+"\n") 
    # print("Ok")
    # DATASET_COUNT += 1
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("fcos_core.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("fcos_core.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    # if output_folder:
    #     torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
