import logging
import os
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
# from albumentations import BboxParams, Compose, HorizontalFlip, LongestMaxSize
# from albumentations.pytorch.transforms import ToTensor
from chainercv.evaluations import eval_detection_voc
from PIL import Image
from torchvision import transforms
from torchvision.ops import nms
from tqdm import tqdm
import pdb

# this is duplicate
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_aug(aug):
    return Compose(
        aug, bbox_params=BboxParams(format="pascal_voc", label_fields=["gt_labels"])
    )


def prepare(img, boxes, max_dim=None, xflip=False, gt_boxes=None, gt_labels=None):
    aug = get_aug(
        [
            LongestMaxSize(max_size=max_dim),
            HorizontalFlip(p=float(xflip)),
            ToTensor(
                normalize=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ),
        ]
    )
    augmented = aug(
        image=img, bboxes=boxes, gt_labels=np.full(len(boxes), fill_value=1)
    )
    augmented_gt = aug(image=img, bboxes=gt_boxes, gt_labels=gt_labels)

    img = augmented["image"].numpy().astype(np.float32)
    boxes = np.asarray(augmented["bboxes"]).astype(np.float32)
    gt_boxes = np.asarray(augmented_gt["bboxes"]).astype(np.float32)

    return img, boxes, gt_boxes


# def evaluate(net, dataloader):
#     """Evaluates network."""
#     with torch.no_grad():
#         net.eval()

#         total_pred_boxes = []
#         total_pred_scores = []
#         total_pred_labels = []
#         total_gt_boxes = []
#         total_gt_labels = []

#         for (
#             img_id,
#             img,  # is it necessary
#             boxes,
#             scaled_imgs,
#             scaled_boxes,
#             scores,
#             gt_boxes,
#             gt_labels,
#         ) in tqdm(dataloader, "Evaluation"):

#             combined_scores = torch.zeros(len(boxes), 20, dtype=torch.float32)
#             batch_scores = np2gpu(scores.numpy(), DEVICE)

#             # total_pred_boxes = []
#             # total_pred_scores = []
#             # total_pred_labels = []
#             # total_gt_boxes = []
#             # total_gt_labels = []


#             for i, scaled_img in enumerate(scaled_imgs):
#                 # scaled_img = scaled_img.numpy()
#                 # tmp_scaled_boxes = scaled_boxes[i].numpy()

#                 # batch_imgs = np2gpu(scaled_img, DEVICE)
#                 # batch_boxes = np2gpu(tmp_scaled_boxes, DEVICE)

#                 batch_imgs = scaled_img.unsqueeze(0).cuda()
#                 batch_boxes = scaled_boxes[i].unsqueeze(0).cuda()

#                 tmp_combined_scores  = net.test_2(batch_imgs, batch_boxes, batch_scores, gt_labels)
#                 # _,tmp_combined_scores  = net(batch_imgs, batch_boxes, batch_scores, gt_labels)

#                 # pdb.set_trace()
#                 combined_scores += tmp_combined_scores.cpu()

#             combined_scores /= 10

#             # gt_boxes = gt_boxes.numpy()
#             # gt_labels = gt_labels.numpy()

#             # batch_gt_boxes = np2gpu(gt_boxes, DEVICE)
#             # batch_gt_labels = np2gpu(gt_labels, DEVICE)

#             # pdb.set_trace()
#             batch_gt_labels = gt_labels.unsqueeze(0).cuda()
#             batch_gt_boxes = gt_boxes.unsqueeze(0).cuda()

#             batch_pred_boxes = []
#             batch_pred_scores = []
#             batch_pred_labels = []

#             for i in range(len(gt_labels)):

#                 # if i == 14

#                 pdb.set_trace()

#                 region_scores = combined_scores[:, gt_labels[i]]
#                 score_mask = region_scores > 0

#                 selected_scores = region_scores[score_mask]
#                 selected_boxes = boxes[score_mask]

#                 nms_mask = nms(selected_boxes, selected_scores, 0.4)
#                 # pdb.set_trace()

#                 selected_boxes = selected_boxes[nms_mask]
#                 selected_scores = selected_scores[nms_mask]


#                 # for j in range(10):


#                 # iou = bbox_iou([selected_boxes], [gt_boxes[i].unsqueeze(0)])

#                 # aa, bb = torch.sort(selected_scores, descending=True)



#                 # batch_pred_boxes.append(selected_boxes[nms_mask].cpu().numpy())
#                 # batch_pred_scores.append(selected_scores[nms_mask].cpu().numpy())
#                 # batch_pred_labels.append(np.full(len(nms_mask), i, dtype=np.int32))

#         #     total_pred_boxes.append(np.concatenate(batch_pred_boxes, axis=0))
#         #     total_pred_scores.append(np.concatenate(batch_pred_scores, axis=0))
#         #     total_pred_labels.append(np.concatenate(batch_pred_labels, axis=0))
#         #     total_gt_boxes.append(batch_gt_boxes[0].cpu().numpy())
#         #     total_gt_labels.append(batch_gt_labels[0].cpu().numpy())

#         #     # pdb.set_trace()
#         #     result = eval_detection_voc(total_pred_boxes, total_pred_labels, total_pred_scores, total_gt_boxes, total_gt_labels, iou_thresh=0.5, use_07_metric=True)
#         #     print(result['map'])

#         # tqdm.write(f"Avg AP: {result['ap']}")
#         # tqdm.write(f"Avg mAP: {result['map']}")

#         # net.train()




def evaluate(net, dataloader, numADL, epoch):
    """Evaluates network."""
    with torch.no_grad():
        net.eval()

        total_pred_boxes = []
        total_pred_scores = []
        total_pred_labels = []
        total_gt_boxes = []
        total_gt_labels = []
        count = 0
        for (
            img_id,
            img,  # is it necessary
            boxes,
            scaled_imgs,
            scaled_boxes,
            scores,
            gt_boxes,
            gt_labels,
        ) in tqdm(dataloader, "Evaluation"):

            count += 1

            # pdb.set_trace()

            # img_id = dataloader.dataset[336][0]
            # img = dataloader.dataset[336][1]
            # boxes = dataloader.dataset[336][2]
            # scaled_imgs = dataloader.dataset[336][3]
            # scaled_boxes = dataloader.dataset[336][4]
            # scores = dataloader.dataset[336][5]
            # gt_boxes = dataloader.dataset[336][0]
            # gt_labels = dataloader.dataset[336][0]

            combined_scores = torch.zeros(len(boxes), 20, dtype=torch.float32)
            batch_scores = np2gpu(scores.numpy(), DEVICE)

            # total_pred_boxes = []
            # total_pred_scores = []
            # total_pred_labels = []
            # total_gt_boxes = []
            # total_gt_labels = []

            # if count >= 336:
            #     print('stop')
            #     pdb.set_trace()

            for i, scaled_img in enumerate(scaled_imgs):
                # print(i)
                # scaled_img = scaled_img.numpy()
                # tmp_scaled_boxes = scaled_boxes[i].numpy()

                # batch_imgs = np2gpu(scaled_img, DEVICE)
                # batch_boxes = np2gpu(tmp_scaled_boxes, DEVICE)

                batch_imgs = scaled_img.unsqueeze(0).cuda()
                batch_boxes = scaled_boxes[i].unsqueeze(0).cuda()


                # tmp_combined_scores  = net(batch_imgs, batch_boxes, batch_scores, gt_labels)
                tmp_combined_scores  = net.test(batch_imgs, batch_boxes, batch_scores, gt_labels)

                # tmp_combined_scores, _  = net(batch_imgs, batch_boxes, batch_scores, gt_labels, numADL, epoch)
                # tmp_combined_scores, _  = net.test(batch_imgs, batch_boxes, batch_scores, gt_labels, numADL, epoch)
                # tmp_combined_scores, _  = net.test_withD(batch_imgs, batch_boxes, batch_scores, gt_labels, numADL, epoch)
                # _, _, tmp_combined_scores = net(batch_imgs, batch_boxes, batch_scores, gt_labels.unsqueeze(0))



                # pdb.set_trace()
                combined_scores += tmp_combined_scores.cpu()

            combined_scores /= 10

            # gt_boxes = gt_boxes.numpy()
            # gt_labels = gt_labels.numpy()

            # batch_gt_boxes = np2gpu(gt_boxes, DEVICE)
            # batch_gt_labels = np2gpu(gt_labels, DEVICE)

            # pdb.set_trace()
            batch_gt_labels = gt_labels.unsqueeze(0).cuda()
            batch_gt_boxes = gt_boxes.unsqueeze(0).cuda()

            batch_pred_boxes = []
            batch_pred_scores = []
            batch_pred_labels = []

            for i in range(20):

                # if i == 14

                region_scores = combined_scores[:, i]
                score_mask = region_scores > 0

                selected_scores = region_scores[score_mask]
                selected_boxes = boxes[score_mask]

                nms_mask = nms(selected_boxes, selected_scores, 0.4)
                # pdb.set_trace()

                batch_pred_boxes.append(selected_boxes[nms_mask].cpu().numpy())
                batch_pred_scores.append(selected_scores[nms_mask].cpu().numpy())
                batch_pred_labels.append(np.full(len(nms_mask), i, dtype=np.int32))

            total_pred_boxes.append(np.concatenate(batch_pred_boxes, axis=0))
            total_pred_scores.append(np.concatenate(batch_pred_scores, axis=0))
            total_pred_labels.append(np.concatenate(batch_pred_labels, axis=0))
            total_gt_boxes.append(batch_gt_boxes[0].cpu().numpy())
            total_gt_labels.append(batch_gt_labels[0].cpu().numpy())

            # pdb.set_trace()
        result = eval_detection_voc(total_pred_boxes, total_pred_labels, total_pred_scores, total_gt_boxes, total_gt_labels, iou_thresh=0.5, use_07_metric=True)
            # print(result['map'])
            # print(result['ap'])

        tqdm.write(f"Avg AP: {result['ap']}")
        tqdm.write(f"Avg mAP: {result['map']}")

        net.train()


def unique_boxes(boxes, scale=1.0):
    """Returns indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def filter_small_boxes(boxes, min_size):
    """Filters out small boxes."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    mask = (w >= min_size) & (h >= min_size)
    return mask


def swap_axes(boxes):
    """Swaps x and y axes."""
    boxes = boxes.copy()
    boxes = np.stack((boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]), axis=1)
    return boxes


def np2gpu(arr, device):
    """Creates torch array from numpy one."""
    arr = np.expand_dims(arr, axis=0)
    return torch.from_numpy(arr).to(device)




# ##################################
# # from __future__ import division

# from collections import defaultdict
# import itertools
# import numpy as np
# import six

# from chainercv.utils.bbox.bbox_iou import bbox_iou


# def eval_detection_voc(
#         pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
#         gt_difficults=None,
#         iou_thresh=0.5, use_07_metric=False):


#     prec, rec = calc_detection_voc_prec_rec(
#         pred_bboxes, pred_labels, pred_scores,
#         gt_bboxes, gt_labels, gt_difficults,
#         iou_thresh=iou_thresh)

#     ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

#     return {'ap': ap, 'map': np.nanmean(ap)}


# def calc_detection_voc_prec_rec(
#         pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
#         gt_difficults=None,
#         iou_thresh=0.5):


#     pred_bboxes = iter(pred_bboxes)
#     pred_labels = iter(pred_labels)
#     pred_scores = iter(pred_scores)
#     gt_bboxes = iter(gt_bboxes)
#     gt_labels = iter(gt_labels)
#     if gt_difficults is None:
#         gt_difficults = itertools.repeat(None)
#     else:
#         gt_difficults = iter(gt_difficults)

#     n_pos = defaultdict(int)
#     score = defaultdict(list)
#     match = defaultdict(list)

#     # c = 1
#     for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
#         six.moves.zip(
#             pred_bboxes, pred_labels, pred_scores,
#             gt_bboxes, gt_labels, gt_difficults):

#         # print(c)

#         if gt_difficult is None:
#             gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

#         for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
#             pred_mask_l = pred_label == l
#             pred_bbox_l = pred_bbox[pred_mask_l]
#             pred_score_l = pred_score[pred_mask_l]
#             # sort by score
#             order = pred_score_l.argsort()[::-1]
#             pred_bbox_l = pred_bbox_l[order]
#             pred_score_l = pred_score_l[order]

#             gt_mask_l = gt_label == l
#             gt_bbox_l = gt_bbox[gt_mask_l]
#             gt_difficult_l = gt_difficult[gt_mask_l]

#             n_pos[l] += np.logical_not(gt_difficult_l).sum()
#             score[l].extend(pred_score_l)

#             if len(pred_bbox_l) == 0:
#                 continue
#             if len(gt_bbox_l) == 0:
#                 match[l].extend((0,) * pred_bbox_l.shape[0])
#                 continue

#             # VOC evaluation follows integer typed bounding boxes.
#             pred_bbox_l = pred_bbox_l.copy()
#             pred_bbox_l[:, 2:] += 1
#             gt_bbox_l = gt_bbox_l.copy()
#             gt_bbox_l[:, 2:] += 1


#             iou = bbox_iou(pred_bbox_l, gt_bbox_l)
#             # pdb.set_trace()
#             # aa, bb = torch.sort(torch.tensor(iou)[:,0], descending=True)
#             # print(aa[0:10])
#             # torch.sort(iou, descending=True)
#             # pdb.set_trace()
#             gt_index = iou.argmax(axis=1)
#             # set -1 if there is no matching ground truth
#             gt_index[iou.max(axis=1) < iou_thresh] = -1
#             del iou

#             selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
#             for gt_idx in gt_index:
#                 if gt_idx >= 0:
#                     if gt_difficult_l[gt_idx]:
#                         match[l].append(-1)
#                     else:
#                         if not selec[gt_idx]:
#                             match[l].append(1)
#                         else:
#                             match[l].append(0)
#                     selec[gt_idx] = True
#                 else:
#                     match[l].append(0)

#             # pdb.set_trace()

#         # c += 1




#     for iter_ in (
#             pred_bboxes, pred_labels, pred_scores,
#             gt_bboxes, gt_labels, gt_difficults):
#         if next(iter_, None) is not None:
#             raise ValueError('Length of input iterables need to be same.')

#     n_fg_class = max(n_pos.keys()) + 1
#     prec = [None] * n_fg_class
#     rec = [None] * n_fg_class

#     for l in n_pos.keys():            #dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
#         score_l = np.array(score[l])
#         match_l = np.array(match[l], dtype=np.int8)

#         order = score_l.argsort()[::-1]
#         match_l = match_l[order]

#         tp = np.cumsum(match_l == 1)
#         fp = np.cumsum(match_l == 0)

#         # If an element of fp + tp is 0,
#         # the corresponding element of prec[l] is nan.
#         prec[l] = tp / (fp + tp)
#         # If n_pos[l] is 0, rec[l] is None.
#         if n_pos[l] > 0:
#             rec[l] = tp / n_pos[l]
#     # pdb.set_trace()
#     return prec, rec


# def calc_detection_voc_ap(prec, rec, use_07_metric=False):


#     n_fg_class = len(prec)
#     ap = np.empty(n_fg_class)
#     for l in six.moves.range(n_fg_class):
#         if prec[l] is None or rec[l] is None:
#             ap[l] = np.nan
#             continue

#         if use_07_metric:
#             # 11 point metric
#             ap[l] = 0
#             for t in np.arange(0., 1.1, 0.1):
#                 if np.sum(rec[l] >= t) == 0:
#                     p = 0
#                 else:
#                     p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
#                 ap[l] += p / 11
#         else:
#             # correct AP calculation
#             # first append sentinel values at the end
#             mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
#             mrec = np.concatenate(([0], rec[l], [1]))

#             mpre = np.maximum.accumulate(mpre[::-1])[::-1]

#             # to calculate area under PR curve, look for points
#             # where X axis (recall) changes value
#             i = np.where(mrec[1:] != mrec[:-1])[0]

#             # and sum (\Delta recall) * prec
#             ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

#     return ap
