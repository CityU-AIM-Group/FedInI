# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import boxlist_iou


import cv2
import matplotlib as mpl
import matplotlib.figure as mplfigure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm

DCOUNT = 0 
class VisImage:
    def __init__(self, img, scale=2.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.set_xlim(0.0, self.width)
        ax.set_ylim(self.height)

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        if filepath.lower().endswith(".jpg") or filepath.lower().endswith(".png"):
            # faster than matplotlib's imshow
            cv2.imwrite(filepath, self.get_image()[:, :, ::-1])
        else:
            # support general formats (e.g. pdf)
            self.ax.imshow(self.img, interpolation="nearest")
            self.fig.savefig(filepath)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        if (self.width, self.height) != (width, height):
            img = cv2.resize(self.img, (width, height))
        else:
            img = self.img

        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        # imshow is slow. blend manually (still quite slow)
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)

        try:
            import numexpr as ne  # fuse them with numexpr

            visualized_image = ne.evaluate("img * (1 - alpha / 255.0) + rgb * (alpha / 255.0)")
        except ImportError:
            alpha = alpha.astype("float32") / 255.0
            visualized_image = img * (1 - alpha) + rgb * alpha

        visualized_image = visualized_image.astype("uint8")

        return visualized_image

import torch
import numpy as np
def do_voc_evaluation(dataset, predictions, output_folder, logger):
    # TODO need to make the use_07_metric format available
    # for the user to choose
    pred_boxlists = []
    gt_boxlists = []

    #################################### FOR PIE
    cls_counter = {}
    per_cls_all = {}
    per_cls_top_k = {}
    # logging.info("counting per class K")
    for image_id, prediction in enumerate(predictions):
        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_labels = gt_boxlist.get_field("labels")
        for cls in gt_labels:
            cls = str(int(cls))
            if cls not in cls_counter:
                cls_counter[cls] = 1
            else:
                cls_counter[cls] += 1
    
    # logging.info("accumulating all boxes")
    for image_id, prediction in enumerate(predictions):
        pred_score = prediction.get_field("scores").numpy()
        pred_clses = prediction.get_field("labels").numpy()
        for box_idx_in_img, pred_cls in enumerate(list(pred_clses)):
            pred_cls = str(int(pred_cls))
            if pred_cls not in per_cls_all:
                per_cls_all[pred_cls] = [(image_id, box_idx_in_img, pred_score[box_idx_in_img])]
            else:
                per_cls_all[pred_cls].append((image_id, box_idx_in_img, pred_score[box_idx_in_img]))

    # logging.info("preserve top k boxes per class")
    def takeThird(elem):
        return elem[2]
    per_cls_top_k = {k: [] for k in cls_counter}
    per_cls_type = {k: [] for k in cls_counter}
    for key, val in per_cls_all.items():
        if key in cls_counter:
            per_cls_top_k[key] = sorted(per_cls_all[key], key=takeThird, reverse=True)[:cls_counter[key]]
            per_cls_type[key] = [0, 0, 0]

    del cls_counter, per_cls_all
    print(per_cls_top_k)

    # logging.info("test predictions' quality")
    for cls, test_boxes_per_cls in per_cls_top_k.items():
        for box in test_boxes_per_cls:
            image_id, box_idx, score = box[0], box[1], box[2]
            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            prediction = predictions[image_id]
            prediction = prediction.resize((image_width, image_height))
            gt_boxlist = dataset.get_groundtruth(image_id)
            match_quality_matrix = boxlist_iou(prediction, gt_boxlist) # [pred, gt] -> [pred]
            match_quality_matrix = torch.max(match_quality_matrix, dim=1)[0]
            iou = match_quality_matrix[box_idx]
            if iou >= 0.75:
                per_cls_type[cls][0] += 1
            elif iou < 0.5:
                per_cls_type[cls][2] += 1
            else:
                per_cls_type[cls][1] += 1
    # logging.info(per_cls_type)
    dic = {k: [0,0,0] for k in per_cls_type}
    mean = [0,0,0]
    for i in per_cls_type:
        dic[i][0] = per_cls_type[i][0]/sum(per_cls_type[i])
        dic[i][1] = per_cls_type[i][1]/sum(per_cls_type[i])
        dic[i][2] = per_cls_type[i][2]/sum(per_cls_type[i])

    for i in dic:
        mean[0] += dic[i][0]
        mean[1] += dic[i][1]
        mean[2] += dic[i][2]
    mean = [k/len(dic) for k in mean]
    print("Mean", mean)
    
    summer = [0,0,0]
    for i in per_cls_type:
        summer[0] += per_cls_type[i][0]
        summer[1] += per_cls_type[i][1]
        summer[2] += per_cls_type[i][2]

    summer = [k/sum(summer) for k in summer]

    # logging.info(mean)
    print("Summer", summer)
    ##########################################################


    # k = []
    for image_id, prediction in enumerate(predictions):
        # img_info = dataset.get_img_infov2(image_id)
        # img_name = img_info["name"]
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        
        # th=prediction.get_field("scores")[torch.argmax(prediction.get_field("scores"))].item()
        # img_id, img = dataset.get_img_for_vis(image_id)
        # if "27132" in img_id:
        #     output = VisImage(img)
        #     pred_score = prediction.get_field("scores").numpy()
        #     prediction_for_vis = prediction[np.nonzero(pred_score >= th)]
        #     pred_bbox_for_vis = prediction_for_vis.bbox.numpy()
        #     pred_label_for_vis = prediction_for_vis.get_field("labels").numpy()
        #     assigned_colors = np.array([dataset.map_class_id_to_class_color()[int(x)] for x in pred_label_for_vis])
        #     if pred_bbox_for_vis.shape[0]:
        #         for i in range(pred_bbox_for_vis.shape[0]):
        #             color = assigned_colors[i]
        #             x0, y0, x1, y1 = pred_bbox_for_vis[i]
        #             width = x1 - x0
        #             height = y1 - y0
        #             _default_font_size = max(np.sqrt(img.shape[0] * img.shape[1]) // 90, 10)
        #             linewidth = max(_default_font_size / 4, 1)

        #             output.ax.add_patch(
        #                 mpl.patches.Rectangle(
        #                     (x0, y0),
        #                     width,
        #                     height,
        #                     fill=False,
        #                     edgecolor=color,
        #                     linewidth=linewidth * output.scale * 1,
        #                     alpha=0.8,
        #                     # linestyle=line_style,
        #                 )
        #             )

        #             text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
        #             horiz_align = "left"
        #             height_ratio = (y1 - y0) / np.sqrt(img.shape[0] * img.shape[1])
        #             font_size = (
        #                 np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
        #                 * 0.9
        #                 * _default_font_size
        #             )

        #             x, y = text_pos
        #             output.ax.text(
        #                 x,
        #                 y,
        #                 dataset.map_class_id_to_class_name(pred_label_for_vis[i]),
        #                 size=font_size * output.scale,
        #                 # size='x-large',
        #                 family="sans-serif",
        #                 # bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
        #                 verticalalignment="top",
        #                 horizontalalignment=horiz_align,
        #                 color=color,
        #                 zorder=10,
        #             )
        #     global DCOUNT
        #     output.save(os.path.join("./LOOK1", str(DCOUNT)+img_id))
        #     DCOUNT += 1


        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)
        # if "27132" in img_id:
        #     output = VisImage(img)
        #     pred_bbox_for_vis = gt_boxlist.bbox.numpy()
        #     pred_label_for_vis = gt_boxlist.get_field("labels").numpy()
        #     print(pred_label_for_vis)
        #     assigned_colors = np.array([dataset.map_class_id_to_class_color()[int(x)] for x in pred_label_for_vis])
        #     if pred_bbox_for_vis.shape[0]:
        #         for i in range(pred_bbox_for_vis.shape[0]):
        #             color = assigned_colors[i]
        #             x0, y0, x1, y1 = pred_bbox_for_vis[i]
        #             width = x1 - x0
        #             height = y1 - y0
        #             _default_font_size = max(np.sqrt(img.shape[0] * img.shape[1]) // 90, 10)
        #             linewidth = max(_default_font_size / 4, 1)

        #             output.ax.add_patch(
        #                 mpl.patches.Rectangle(
        #                     (x0, y0),
        #                     width,
        #                     height,
        #                     fill=False,
        #                     edgecolor=color,
        #                     linewidth=linewidth * output.scale * 1,
        #                     alpha=0.8,
        #                     # linestyle=line_style,
        #                 )
        #             )

        #             text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
        #             horiz_align = "left"
        #             height_ratio = (y1 - y0) / np.sqrt(img.shape[0] * img.shape[1])
        #             font_size = (
        #                 np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
        #                 * 0.9
        #                 * _default_font_size
        #             )

        #             x, y = text_pos
        #             output.ax.text(
        #                 x,
        #                 y,
        #                 dataset.map_class_id_to_class_name(pred_label_for_vis[i]),
        #                 size=font_size * output.scale,
        #                 # size='x-large',
        #                 family="sans-serif",
        #                 # bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
        #                 verticalalignment="top",
        #                 horizontalalignment=horiz_align,
        #                 color=color,
        #                 zorder=10,
        #             )
        #     output.save(os.path.join("./LOOK1", "gt"+img_id))

    #     a = "{}: {} | {}".format(img_name, str(prediction.get_field("labels")[torch.argmax(prediction.get_field("scores"))].item()), str(gt_boxlist.get_field("labels")[0].item()))
    #     k.append(a)
    # global DCOUNT
    # with open("ini{}.txt".format(str(DCOUNT)), "a+") as f:
    #     for a in k:
    #         f.write(a+"\n") 
    # DCOUNT += 1

    result = eval_detection_voc(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=True,
    )
    ###################################################################
    th_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ths = []
    for i in th_list:
        ths.append(eval_detection_voc(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=i,
            use_07_metric=True,
        )["map"])
    logger.info(ths)
    # input()
    ##################################################################
    result_str = "mAP: {:.4f}\n".format(result["map"])
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), ap
        )
    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "w") as fid:
            fid.write(result_str)
    return result


def eval_detection_voc(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_voc_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh
    )
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)
    # print(rec)
    return {"ap": ap, "map": np.nanmean(ap)}


def calc_detection_voc_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap
