import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet, vgg16, vgg16_bn
from torchvision.ops import roi_pool

# from utils import BASE_DIR
import pdb
from torchvision.utils import save_image
import cv2
import numpy as np
from tqdm import tqdm

import itertools
from chainercv.utils.bbox.bbox_iou import bbox_iou

from network_general import resnet50, CLUB, resnet50_cvpr, mobilenet_v2, resnet50_i2c
from network_general import initialize_weights, mobilenet_v1
from sklearn.metrics import auc
from torch.autograd import Variable







class Minmaxcam_resnet(nn.Module):
    def __init__(self, base_net="vgg", set_size = 5, numclass=200):
        super().__init__()

        assert base_net in {"alexnet", "vgg"}, "`base_net` should be in {alexnet, vgg}"

        self.base_net = base_net
        self.numclass = numclass


        self.base = resnet50_cvpr(architecture_type='cam', pretrained=True)

        self.pred = nn.Linear(2048, self.numclass)


        self.set_size = set_size

        self.aa = list(range(0, self.set_size))
        self.bb = list(itertools.combinations(self.aa, 2))
        self.cc = np.zeros([len(self.bb),2])
        for i in range(len(self.bb)):
            self.cc[i,0] = self.bb[i][0]
            self.cc[i,1] = self.bb[i][1]

        self.mse = nn.MSELoss()



    def show_tsne(self, batch_imgs, index, gt_label, gt_bbox, ori_size):
        # pdb.set_trace()
        out_p5 = self.base(batch_imgs)
        # out_p5 = torch.relu(self.extra_conv(out_p5))
        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_p5, dim=1).unsqueeze(1)

        min_tmp = torch.min(hm, dim=2)[0]
        min_tmp = torch.min(min_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)
        max_tmp = torch.max(hm, dim=2)[0]
        max_tmp = torch.max(max_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)


        hm = (hm - min_tmp)/(max_tmp - min_tmp)

        ################################################

        hm = F.interpolate(hm, size=(224,224), mode='bilinear')
        out_p5_hm = self.base(batch_imgs*hm)

        repre_cam = torch.mean(out_p5_hm, dim=(2,3))
        repre = torch.mean(out_p5, dim=(2,3))

        return repre_cam, repre



    def loss_common_part_interclass(self, repre_1, repre_2):
        # print('hi')

        # repre_1 = torch.zeros([len(self.cc), 4096]).cuda()
        # repre_2 = torch.zeros([len(self.cc), 4096]).cuda()
        c_loss = torch.tensor([0.]).cuda()
        for i in range(len(self.cc)):

            # pdb.set_trace()
            c_loss += self.mse(repre_1[int(self.cc[i,0])].unsqueeze(0), repre_2[int(self.cc[i,1])].unsqueeze(0))

        return c_loss/len(self.cc)


    def update_classification(self, batch_imgs, label, ss, bs):

        for param in self.base.parameters():
            param.requires_grad = True

        for param in self.pred.parameters():
            param.requires_grad = True

        
        out_extra = self.base(batch_imgs)




        pred = self.pred(torch.mean(out_extra, dim=(2,3)))
               
        # pdb.set_trace()
        #################################   

        # repre_set = torch.mean(repre_masked.reshape([bs,ss,1024]), dim=1)
         
        # loss_set = F.cross_entropy(self.set_pred(repre_set) , label[0])
        # pdb.set_trace()
        # print(torch.transpose(label.repeat(1,ss).view(ss,bs),1,0).reshape(1,ss*bs)[0])

        if len(label[0]) != ss*bs:


            loss_img = F.cross_entropy(pred, torch.transpose(label.repeat(1,ss).view(ss,bs),1,0).reshape(1,ss*bs)[0], reduction='mean')

        else:

            # pdb.set_trace()

            loss_img = F.cross_entropy(pred, label[0], reduction='mean')


        return loss_img




    def get_hms(self, batch_imgs, label, ss, bs):




        # pdb.set_trace()
        # out_extra = out_extra.detach()
        # self.base.eval()
        out_extra = self.base(batch_imgs)
        # self.base.train()

        # pdb.set_trace()
        # pred = self.pred(torch.mean(out_extra, dim=(2,3)))

        predict_cls = torch.transpose(label.repeat(1,ss).view(ss,bs),1,0).reshape(1,ss*bs)[0]
        # pdb.set_trace()
        # np.where((predict_cls != 200) and (predict_cls != 201) and(predict_cls != 202))


        for i in range(ss*bs):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)

        # pdb.set_trace()

        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)


        min_tmp = torch.min(hm, dim=2)[0]
        min_tmp = torch.min(min_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)
        max_tmp = torch.max(hm, dim=2)[0]
        max_tmp = torch.max(max_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)


        hm = (hm - min_tmp)/(max_tmp - min_tmp)

        ################################################
        # out_extra_masked = out_extra*hm
        # repre_cam = self.gap(out_extra_masked).squeeze(2).squeeze(2)
        # repre = self.gap(out_extra).squeeze(2).squeeze(2)
        # ################################################

        hm = F.interpolate(hm, size=(224,224), mode='bilinear')

        return hm


    def top1_loc_top15(self, batch_imgs, gt_bbox, gt, ori_size, bprime):

        out_p5 = self.base(batch_imgs)
        # out_extra = torch.relu(self.extra_conv(out_p5))

        # predict_cls = torch.argmax(pred, dim=1)

        predict_cls, bb = bprime.predict_acc(batch_imgs)
        # pdb.set_trace()

        predict_cls_ = gt[0]-1

        for i in range(predict_cls_.shape[0]):
            if i == 0:

                W = self.pred.weight[int(predict_cls_[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls_[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        # W = W/torch.sum(W)
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_p5, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        # threshold_list = np.arange(0,1,0.01)
        threshold_list = np.array([0.1])
        counter_03 = np.zeros(len(threshold_list))
        counter_05 = np.zeros(len(threshold_list))
        counter_07 = np.zeros(len(threshold_list))

        # pdb.set_trace()
        # return torch.sum(predict_cls == gt[0]-1)

        for ii in range(batch_imgs.shape[0]):
            # pdb.set_trace()
            if (predict_cls[ii] == gt[0][ii]-1):
                # print('yes')

            # save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            # ref_1 = cv2.imread('./temp_1.png')
                c_hm = hm[ii].unsqueeze(0)
                c_hm = F.interpolate(c_hm, size=(ori_size[ii,0],ori_size[ii,1]), mode='bilinear')
                c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))
                c_gt_bbox =  np.zeros([1,4])

                c_gt_bbox[0,0] = gt_bbox[ii,:][0]-1
                c_gt_bbox[0,1] = gt_bbox[ii,:][1]-1
                c_gt_bbox[0,2] = gt_bbox[ii,:][0]-1 + gt_bbox[ii,:][2]
                c_gt_bbox[0,3] = gt_bbox[ii,:][1]-1 + gt_bbox[ii,:][3]

                # iouu = np.zeros([100])

                for k in range(len(counter_03)):

                    cm_ = 255*c_hm.cpu().numpy()[0][0]
                    cm_ = cm_.astype('uint8')
                    threshold_value = int(np.max(cm_) * threshold_list[k])
                    _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


                    contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cc = max(contours, key=cv2.contourArea)
                    xx, yy, ww, hh = cv2.boundingRect(cc)
    
                    estimated_bbox = np.zeros([1,4])
                    estimated_bbox[0,1] = yy
                    estimated_bbox[0,3] = yy+hh
                    estimated_bbox[0,0] = xx
                    estimated_bbox[0,2] = xx+ww




                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.5:
                        counter_05[k] += 1

                    counter_03[k] += 1

            if gt[0][ii]-1 in bb[ii]:

                c_hm = hm[ii].unsqueeze(0)
                c_hm = F.interpolate(c_hm, size=(ori_size[ii,0],ori_size[ii,1]), mode='bilinear')
                c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))
                c_gt_bbox =  np.zeros([1,4])

                c_gt_bbox[0,0] = gt_bbox[ii,:][0]-1
                c_gt_bbox[0,1] = gt_bbox[ii,:][1]-1
                c_gt_bbox[0,2] = gt_bbox[ii,:][0]-1 + gt_bbox[ii,:][2]
                c_gt_bbox[0,3] = gt_bbox[ii,:][1]-1 + gt_bbox[ii,:][3]

                # iouu = np.zeros([100])

                for k in range(len(counter_03)):

                    cm_ = 255*c_hm.cpu().numpy()[0][0]
                    cm_ = cm_.astype('uint8')
                    threshold_value = int(np.max(cm_) * threshold_list[k])
                    _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


                    contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cc = max(contours, key=cv2.contourArea)
                    xx, yy, ww, hh = cv2.boundingRect(cc)
    
                    estimated_bbox = np.zeros([1,4])
                    estimated_bbox[0,1] = yy
                    estimated_bbox[0,3] = yy+hh
                    estimated_bbox[0,0] = xx
                    estimated_bbox[0,2] = xx+ww




                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.5:
                        counter_07[k] += 1

                    # counter_03[k] += 1



        return counter_03, counter_05, counter_07

    def update_pwnn(self, batch_imgs, label, ss, bs):


        for param in self.base.parameters():
            param.requires_grad = False

        for param in self.pred.parameters():
            param.requires_grad = True


        out_extra = self.base(batch_imgs)


        if len(label[0]) != ss*bs:

            predict_cls = torch.transpose(label.repeat(1,ss).view(ss,bs),1,0).reshape(1,ss*bs)[0]

        else:
            predict_cls = label[0]






        for i in range(ss*bs):
            if i == 0:

                W = self.pred.weight[predict_cls[i]].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)

        # pdb.set_trace()

        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)


        min_tmp = torch.min(hm, dim=2)[0]
        min_tmp = torch.min(min_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)
        max_tmp = torch.max(hm, dim=2)[0]
        max_tmp = torch.max(max_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)


        hm = (hm - min_tmp)/(max_tmp - min_tmp)

        ################################################
        # out_extra_masked = out_extra*hm
        # repre_cam = self.gap(out_extra_masked).squeeze(2).squeeze(2)
        # repre = self.gap(out_extra).squeeze(2).squeeze(2)
        # ################################################

        hm = F.interpolate(hm, size=(224,224), mode='bilinear')

        # self.base.eval()
        out_extra_hm = self.base(batch_imgs*hm)


        repre_cam = torch.mean(out_extra_hm, dim=(2,3))
        repre = torch.mean(out_extra, dim=(2,3))
        ################################################

        # pdb.set_trace()

        for i in range(bs):
            c_loss_common = self.loss_common_part(repre_cam[i*ss:ss*(i+1)]) 
            c_loss_ori = self.loss_ori_img(repre_cam[ss*i:ss*(i+1)], repre[ss*i:ss*(i+1)]) 
            if i == 0:
                loss_common = c_loss_common
                loss_ori = c_loss_ori
            else:
                loss_common += c_loss_common
                loss_ori += c_loss_ori

        loss_common /= bs
        loss_ori /= bs
        
        return loss_common, loss_ori









 
    def top1_loc_imagenet(self, batch_imgs, gt_bbox, gt, ori_size):
        # pdb.set_trace()


        out_extra = self.base(batch_imgs)



        predict_cls = gt[0]
        # pdb.set_trace()
        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        # W = W/torch.sum(W)
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        threshold_list = np.arange(0,1,0.01)
        # threshold_list = np.array([0.2])
        counter_03 = np.zeros(len(threshold_list))
        counter_05 = np.zeros(len(threshold_list))
        counter_07 = np.zeros(len(threshold_list))

        if predict_cls[0] == gt[0][0]:

            for ii in range(batch_imgs.shape[0]):
            # save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            # ref_1 = cv2.imread('./temp_1.png')
                # pdb.set_trace()
                c_hm = hm[ii].unsqueeze(0)
                c_hm = F.interpolate(c_hm, size=(ori_size[ii,1],ori_size[ii,0]), mode='bilinear')
                c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))
                


                for p in range(gt_bbox.shape[1]):



                    if gt_bbox[ii, p].shape[1] != 0:

                        c_gt_bbox =  np.zeros([1,4])


                        c_gt_bbox[0,0] = gt_bbox[ii,p][0][0]-1
                        c_gt_bbox[0,1] = gt_bbox[ii,p][0][1]-1
                        c_gt_bbox[0,2] = gt_bbox[ii,p][0][0]-1 + gt_bbox[ii,p][0][2]
                        c_gt_bbox[0,3] = gt_bbox[ii,p][0][1]-1 + gt_bbox[ii,p][0][3]


                        for k in range(len(counter_03)):

                            if counter_05[k] * counter_03[k] * counter_07[k] == 1:

                                continue

                            else:

                                cm_ = 255*c_hm.cpu().numpy()[0][0]
                                cm_ = cm_.astype('uint8')
                                threshold_value = int(np.max(cm_) * threshold_list[k])
                                _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


                                contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                                # for kk in range
                                # pdb.set_trace()
                                for kk in range(len(contours)):
                                # for kk in range(1):

                                    # cc = max(contours, key=cv2.contourArea)
                                    cc = contours[kk]
                                    xx, yy, ww, hh = cv2.boundingRect(cc)
    
                                    estimated_bbox = np.zeros([1,4])
                                    estimated_bbox[0,1] = yy
                                    estimated_bbox[0,3] = yy+hh
                                    estimated_bbox[0,0] = xx
                                    estimated_bbox[0,2] = xx+ww


                                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.5:

                                        if counter_05[k] == 0:
                                            counter_05[k] += 1

                                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.3:
                                        if counter_03[k] == 0:
                                            counter_03[k] += 1
                                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.7:
                                        if counter_07[k] == 0:
                                            counter_07[k] += 1

                    else:

                        break



            return counter_03, counter_05, counter_07

        else:
            return 0




    def acc(self, batch_imgs, index, gt_bbox, gt, ori_size):
        # pdb.set_trace()

        out_extra = self.base(batch_imgs)
        pred = self.pred(torch.mean(out_extra, dim=(2,3)))
        # pred = self.pred(out_p3)


        predict_cls = torch.argmax(pred, dim=1)
        # predict_cls = gt[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        # W = W/torch.sum(W)
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        counter = 0

        if predict_cls[0] == gt[0][0]-1:

            for ii in range(batch_imgs.shape[0]):

            

                counter = counter+1


            return counter

        else:
            return 0


    def top1_loc_auc(self, batch_imgs, gt, mask_path):

        out_extra = self.base(batch_imgs)

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        # W = W/torch.sum(W)
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        threshold_list = np.arange(0,1,0.001)
        # threshold_list = np.array([0.2])
        counter_03 = np.zeros(len(threshold_list))
        counter_05 = np.zeros(len(threshold_list))
        counter_07 = np.zeros(len(threshold_list))

        num_bins = len(threshold_list) + 2
        threshold_list_right_edge = np.append(threshold_list,
                                                   [1.0, 2.0, 3.0])


        gt_true_score_hist = np.zeros(num_bins, dtype=np.float)
        gt_false_score_hist = np.zeros(num_bins, dtype=np.float)


        if predict_cls[0] == gt[0][0]-1:

            auc_ = 0

            for ii in range(batch_imgs.shape[0]):

                c_hm = hm[ii].unsqueeze(0)
                c_hm = F.interpolate(c_hm, size=(224,224), mode='bilinear')
                c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))

                precision = np.zeros([threshold_list.shape[0]])
                recall = np.zeros([threshold_list.shape[0]])

                c_mask_path = mask_path[ii]
                mask_path_ = []

                for kk in range(len(c_mask_path)):
                    cc_path = c_mask_path[kk]
                    # pdb.set_trace()
                    if cc_path.split('_')[-1] == 'ignore.png':
                        ignore_path_ = cc_path
                    else:
                        mask_path_.append(cc_path)

                c_gt_mask = get_mask(mask_path_, ignore_path_)



                c_hm = c_hm[0,0].detach().cpu().numpy()
                gt_true_scores = c_hm[c_gt_mask == 1]
                gt_false_scores = c_hm[c_gt_mask == 0]

                gt_true_hist, _ = np.histogram(gt_true_scores, bins=threshold_list_right_edge)
                gt_true_score_hist += gt_true_hist.astype(np.float)
                
                gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=threshold_list_right_edge)
                gt_false_score_hist += gt_false_hist.astype(np.float)

                # pdb.set_trace()



                


            return gt_true_score_hist, gt_false_score_hist

        else:
            return 0

    def top1_loc_auc_2(self, gt_true_score_hist, gt_false_score_hist):


        # pdb.set_trace()

        num_gt_true = gt_true_score_hist.sum()
        tp = gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = gt_false_score_hist.sum()
        fp = gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        # auc *= 100

        # print("Mask AUC on split {}: {}".format(self.split, auc))
        return auc


    def top1_loc(self, batch_imgs, gt_bbox, gt, ori_size):

        out_extra = self.base(batch_imgs)
        pred = self.pred(torch.mean(out_extra, dim=(2,3)))
        # pred = self.pred(out_p3)

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)



        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)


        threshold_list = np.arange(0,1,0.01)
        counter_03 = np.zeros(len(threshold_list))
        counter_05 = np.zeros(len(threshold_list))
        counter_07 = np.zeros(len(threshold_list))

        if predict_cls[0] == gt[0][0]-1:

            for ii in range(batch_imgs.shape[0]):

                c_hm = hm[ii].unsqueeze(0)
                c_hm = F.interpolate(c_hm, size=(ori_size[ii,0],ori_size[ii,1]), mode='bilinear')

                c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))
                # c_hm = F.sigmoid(20*(c_hm - 0.5))

                c_gt_bbox =  np.zeros([1,4])

                c_gt_bbox[0,0] = gt_bbox[ii,:][0]-0
                c_gt_bbox[0,1] = gt_bbox[ii,:][1]-0
                c_gt_bbox[0,2] = gt_bbox[ii,:][0]-0 + gt_bbox[ii,:][2]
                c_gt_bbox[0,3] = gt_bbox[ii,:][1]-0 + gt_bbox[ii,:][3]

                # iouu = np.zeros([100])

                for k in range(len(counter_03)):

                    cm_ = 255*c_hm.cpu().numpy()[0][0]
                    cm_ = cm_.astype('uint8')
                    threshold_value = int(np.max(cm_) * threshold_list[k])
                    _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


                    contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cc = max(contours, key=cv2.contourArea)
                    xx, yy, ww, hh = cv2.boundingRect(cc)
    
                    estimated_bbox = np.zeros([1,4])
                    estimated_bbox[0,1] = yy
                    estimated_bbox[0,3] = yy+hh
                    estimated_bbox[0,0] = xx
                    estimated_bbox[0,2] = xx+ww


                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.5:
                        counter_05[k] += 1
                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.3:
                        counter_03[k] += 1
                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.7:
                        counter_07[k] += 1



            return counter_03, counter_05, counter_07

        else:
            return 0



    def loss_common_part(self, repre):
        # print('hi')

        # repre_1 = torch.zeros([len(self.cc), 4096]).cuda()
        # repre_2 = torch.zeros([len(self.cc), 4096]).cuda()
        c_loss = torch.tensor([0.]).cuda()
        for i in range(len(self.cc)):

            # pdb.set_trace()
            c_loss += self.mse(repre[int(self.cc[i,0])].unsqueeze(0), repre[int(self.cc[i,1])].unsqueeze(0))

        return c_loss/len(self.cc)
        # pdb.set_trace()

    def loss_ori_img(self, repre, repre_ori):
        # print('hi')

        loss = torch.mean(self.mse(repre, repre_ori))
        return loss



    def show_hm(self, batch_imgs, index, gt_label, gt_bbox, ori_size):
        # pdb.set_trace()
        out_extra = self.base(batch_imgs)
        pred = self.pred(torch.mean(out_extra, dim=(2,3)))

        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')

        # pdb.set_trace()
        for i in range(hm.shape[0]):
            c_hm = hm[i].unsqueeze(0)
            c_hm = (c_hm - torch.min(c_hm)) /(torch.max(c_hm) - torch.min(c_hm))
            c_hm =  F.interpolate(c_hm, size=(224,224), mode='bilinear')

            c_hm = c_hm.cpu().numpy()

            c_hm = c_hm[0][0]
            # pdb.set_trace()

            cv2.imwrite('test_{}.png'.format(i), 255*c_hm)




    def show_hm____(self, batch_imgs, index, gt_label, gt_bbox, ori_size):
        # pdb.set_trace()
        out_extra = self.base(batch_imgs)
        pred = self.pred(torch.mean(out_extra, dim=(2,3)))

        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')

        # pdb.set_trace()
        # ccc = 0
        for i in range(batch_imgs.shape[0]):

            heatmap = np.zeros([ori_size[i,0],ori_size[i,1]])

            save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)
            c_gt_bbox =  np.zeros([1,4])
            ref_1 = cv2.imread('./temp_1.png')
            ref_1 = cv2.resize(ref_1, (ori_size[i,1],ori_size[i,0]))
            c_hm = F.interpolate(hm[i].unsqueeze(0), size=(ori_size[i,0],ori_size[i,1]), mode='bilinear')

            c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # # c_hm = c_hm >= 0.13
            # c_hm = c_hm >= 0.18




            cm_ = 255*c_hm.cpu().numpy()[0][0]
            cm_ = cm_.astype('uint8')
            threshold_value = int(np.max(cm_) * 0.13)
            _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


            contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cc = max(contours, key=cv2.contourArea)
            xx, yy, ww, hh = cv2.boundingRect(cc)

            pdb.set_trace()
    
            # for iii in range(cc.shape[0]):
            #     heatmap[cc[iii][0][0], cc[iii][0][1]] = 1

            estimated_bbox = np.zeros([1,4])
            estimated_bbox[0,1] = yy
            estimated_bbox[0,3] = yy+hh
            estimated_bbox[0,0] = xx
            estimated_bbox[0,2] = xx+ww


                # c_gt_bbox = np.array([gt_bbox[i,:]-1])
            # pdb.set_trace()    
            # gt_bbox
            c_gt_bbox[0,0] = gt_bbox[i,:][0]-1
            c_gt_bbox[0,1] = gt_bbox[i,:][1]-1
            c_gt_bbox[0,2] = gt_bbox[i,:][0]-1 + gt_bbox[i,:][2]
            c_gt_bbox[0,3] = gt_bbox[i,:][1]-1 + gt_bbox[i,:][3]

            iou = bbox_iou(c_gt_bbox,estimated_bbox)[0]
            # pdb.set_trace()

            heatmap = np.uint8(255 * c_hm[0][0].cpu().detach().numpy())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img_1 = heatmap * 0.7 + ref_1 *0.5

            if gt_label[0][i]-1 == predict_cls[i]:

                cv2.imwrite('./{}_{}_{}_T_.png'.format(index, i, iou), superimposed_img_1)
            else:
                cv2.imwrite('./{}_{}_{}_F_.png'.format(index, i, iou), superimposed_img_1)

    def show_hm_openimage(self, batch_imgs, index, gt_label):
        # pdb.set_trace()
        out_extra = self.base(batch_imgs)
        pred = self.pred(torch.mean(out_extra, dim=(2,3)))

        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')

        # pdb.set_trace()

        for i in range(batch_imgs.shape[0]):
            save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            ref_1 = cv2.imread('./temp_1.png')
            ref_1 = cv2.resize(ref_1, (224,224))
            c_hm = F.interpolate(hm[i].unsqueeze(0), size=(224,224), mode='bilinear')
            # c_hm = w_scale[i].unsqueeze(0)
            # pdb.set_trace()
            c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.2

            heatmap = np.uint8(255 * c_hm[0][0].cpu().detach().numpy())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img_1 = heatmap * 0.7 + ref_1 *0.5

            cv2.imwrite('./{}_{}.png'.format(index, i), superimposed_img_1)








class Minmaxcam_mobilenet(nn.Module):
    def __init__(self, base_net="vgg", set_size = 5, numclass=200):
        super().__init__()


        self.numclass = numclass


        self.base = mobilenet_v2(pretrained=True)

        self.features = self.base.features

        ################################################################
        self.pred = nn.Linear(1280, self.numclass)
        self.gap = nn.AvgPool2d(28, stride=28)
        ################################################################

    
        self.set_size = set_size

        self.aa = list(range(0, self.set_size))
        self.bb = list(itertools.combinations(self.aa, 2))
        self.cc = np.zeros([len(self.bb),2])
        for i in range(len(self.bb)):
            self.cc[i,0] = self.bb[i][0]
            self.cc[i,1] = self.bb[i][1]
        # self.cc = int(self.cc)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse = nn.MSELoss()




    def show_tsne(self, batch_imgs, index, gt_label, gt_bbox, ori_size):
        # pdb.set_trace()
        out_p5 = self.features(batch_imgs)
        # out_p5 = torch.relu(self.extra_conv(out_p5))
        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_p5, dim=1).unsqueeze(1)

        min_tmp = torch.min(hm, dim=2)[0]
        min_tmp = torch.min(min_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)
        max_tmp = torch.max(hm, dim=2)[0]
        max_tmp = torch.max(max_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)


        hm = (hm - min_tmp)/(max_tmp - min_tmp)

        ################################################

        hm = F.interpolate(hm, size=(224,224), mode='bilinear')
        out_p5_hm = self.features(batch_imgs*hm)

        repre_cam = torch.mean(out_p5_hm, dim=(2,3))
        repre = torch.mean(out_p5, dim=(2,3))

        return repre_cam, repre




    def update_classification(self, batch_imgs, label, ss, bs):

        for param in self.features.parameters():
            param.requires_grad = True

        # for param in self.extra_conv.parameters():
        #     param.requires_grad = True

        for param in self.pred.parameters():
            param.requires_grad = True

        # out_p1 = self.features_p1(batch_imgs)
        # out_p2 = self.features_p2(out_p1)
        # out_p3 = self.features_p3(out_p2)
        # out_p4 = self.features_p4(out_p3)
        # out_p5 = self.features_p5(out_p4)

        out_p5 = self.features(batch_imgs)
        # out_p5 = torch.relu(self.extra_conv(out_p5))
        pred = self.pred(self.gap(out_p5).squeeze(2).squeeze(2))
       
        # pdb.set_trace()
        #################################   

        # repre_set = torch.mean(repre_masked.reshape([bs,ss,1024]), dim=1)
         
        # loss_set = F.cross_entropy(self.set_pred(repre_set) , label[0])
        # pdb.set_trace()
        # print(torch.transpose(label.repeat(1,ss).view(ss,bs),1,0).reshape(1,ss*bs)[0])

        loss_img = F.cross_entropy(pred, torch.transpose(label.repeat(1,ss).view(ss,bs),1,0).reshape(1,ss*bs)[0], reduction='mean')


        return loss_img






    def show_hm_bbox_imagenet(self, batch_imgs, index, gt_label, gt_bbox, ori_size):

        out_extra = self.features(batch_imgs)
        # out_extra = torch.relu(self.extra_conv(out_p5))
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)



        for ii in range(batch_imgs.shape[0]):
            save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            ref_1 = cv2.imread('./temp_1.png')
            ref_1 = cv2.resize(ref_1, (ori_size[i,0],ori_size[i,1]))
            c_hm = F.interpolate(hm[i].unsqueeze(0), size=(ori_size[i,1],ori_size[i,0]), mode='bilinear')
            # c_hm = w_scale[i].unsqueeze(0)
            # pdb.set_trace()
            c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.36

            heatmap = np.uint8(255 * c_hm[0][0].cpu().detach().numpy())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img_1 = heatmap * 0.7 + ref_1 *0.5


            for p in range(gt_bbox.shape[1]):





                if gt_bbox[ii, p].shape[1] != 0:

                    c_gt_bbox =  np.zeros([1,4], dtype=int)


                    c_gt_bbox[0,0] = gt_bbox[ii,p][0][0]-1
                    c_gt_bbox[0,1] = gt_bbox[ii,p][0][1]-1
                    c_gt_bbox[0,2] = gt_bbox[ii,p][0][0]-1 + gt_bbox[ii,p][0][2]
                    c_gt_bbox[0,3] = gt_bbox[ii,p][0][1]-1 + gt_bbox[ii,p][0][3]

                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 0] = 0
                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 1] = 255
                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 2] = 0

                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 0] = 0
                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 1] = 255
                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 2] = 0

                    superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 0] = 0
                    superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 1] = 255
                    superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 2] = 0

                    superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 0] = 0
                    superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 1] = 255
                    superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 2] = 0


                    cm_ = 255*c_hm.cpu().numpy()[0][0]
                    cm_ = cm_.astype('uint8')
                    # threshold_value = int(np.max(cm_) * 0.39)
                    # threshold_value = int(np.max(cm_) * 0.33)
                    threshold_value = int(np.max(cm_) * 0.28)

                    _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


                    contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            


                    for kk in range(1):

                        cc = max(contours, key=cv2.contourArea)
                        xx, yy, ww, hh = cv2.boundingRect(cc)
    
                        estimated_bbox = np.zeros([1,4],dtype=int)
                        estimated_bbox[0,1] = yy
                        estimated_bbox[0,3] = yy+hh
                        estimated_bbox[0,0] = xx
                        estimated_bbox[0,2] = xx+ww
    
        
                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 0] = 0
                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 1] = 0
                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 2] = 255

                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 0] = 0
                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 1] = 0
                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 2] = 255

                        superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 0] = 0
                        superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 1] = 0
                        superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 2] = 255

                        superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 0] = 0
                        superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 1] = 0
                        superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 2] = 255




            cv2.imwrite('./{}_{}_T.png'.format(index, i), superimposed_img_1)
      


    def update_pwnn(self, batch_imgs, label, ss, bs):


        for param in self.features.parameters():
            param.requires_grad = False


        for param in self.pred.parameters():
            param.requires_grad = True


        out_p5 = self.features(batch_imgs)
        pred = self.pred(self.gap(out_p5).squeeze(2).squeeze(2))


        for i in range(ss*bs):
            if i == 0:

                W = self.pred.weight[predict_cls[i]].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)

        # pdb.set_trace()

        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_p5, dim=1).unsqueeze(1)


        min_tmp = torch.min(hm, dim=2)[0]
        min_tmp = torch.min(min_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)
        max_tmp = torch.max(hm, dim=2)[0]
        max_tmp = torch.max(max_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)


        hm = (hm - min_tmp)/(max_tmp - min_tmp)

        ################################################

        hm = F.interpolate(hm, size=(224,224), mode='bilinear')
        out_p5_hm = self.features(batch_imgs*hm)

        repre_cam = self.gap(out_p5_hm).squeeze(2).squeeze(2)
        repre = self.gap(out_p5).squeeze(2).squeeze(2)
        ################################################

        # repre_cam = self.gap(out_p5*hm).squeeze(2).squeeze(2)
        # repre = self.gap(out_p5).squeeze(2).squeeze(2)

        ################################################

        # pdb.set_trace()


        for i in range(bs):
            c_loss_common = self.loss_common_part(repre_cam[i*ss:ss*(i+1)]) 
            c_loss_ori = self.loss_ori_img(repre_cam[ss*i:ss*(i+1)], repre[ss*i:ss*(i+1)]) 
            if i == 0:
                loss_common = c_loss_common
                loss_ori = c_loss_ori
            else:
                loss_common += c_loss_common
                loss_ori += c_loss_ori

        loss_common /= bs
        loss_ori /= bs

        # pdb.set_trace()
        return loss_common, loss_ori


    def get_hms(self, batch_imgs, label, ss, bs):

        # pdb.set_trace()

        out_p5 = self.features(batch_imgs)
        # out_extra = torch.relu(self.extra_conv(out_p5))
        pred = self.pred(self.gap(out_p5).squeeze(2).squeeze(2))

        predict_cls = torch.transpose(label.repeat(1,ss).view(ss,bs),1,0).reshape(1,ss*bs)[0]
        # predict_cls = torch.argmax(pred, dim=0)

        for i in range(ss*bs):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)

        # pdb.set_trace()

        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_p5, dim=1).unsqueeze(1)


        min_tmp = torch.min(hm, dim=2)[0]
        min_tmp = torch.min(min_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)
        max_tmp = torch.max(hm, dim=2)[0]
        max_tmp = torch.max(max_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)


        hm = (hm - min_tmp)/(max_tmp - min_tmp)

        ################################################

        hm = F.interpolate(hm, size=(224,224), mode='bilinear')

        return hm











    def show_hm_bbox(self, batch_imgs, index, gt_label, gt_bbox, ori_size):
        # pdb.set_trace()
        out_extra = self.features(batch_imgs)        
        # out_extra = torch.relu(self.extra_conv(out_p5))
        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')





        for i in range(batch_imgs.shape[0]):
            save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            ref_1 = cv2.imread('./temp_1.png')
            ref_1 = cv2.resize(ref_1, (ori_size[i,1],ori_size[i,0]))
            c_hm = F.interpolate(hm[i].unsqueeze(0), size=(ori_size[i,0],ori_size[i,1]), mode='bilinear')
            # c_hm = w_scale[i].unsqueeze(0)
            # pdb.set_trace()
            c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.36

            heatmap = np.uint8(255 * c_hm[0][0].cpu().detach().numpy())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img_1 = heatmap * 0.7 + ref_1 *0.5

            c_gt_bbox =  np.zeros([1,4], dtype=int)

            # pdb.set_trace()

            c_gt_bbox[0,0] = int(gt_bbox[0,:][0]-1) #x1
            c_gt_bbox[0,1] = gt_bbox[0,:][1]-1 #y1
            c_gt_bbox[0,2] = gt_bbox[0,:][0]-1 + gt_bbox[0,:][2] #x2
            c_gt_bbox[0,3] = gt_bbox[0,:][1]-1 + gt_bbox[0,:][3] #y2



            cm_ = 255*c_hm.cpu().numpy()[0][0]
            cm_ = cm_.astype('uint8')
            threshold_value = int(np.max(cm_) * 0.17)
            _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


            contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cc = max(contours, key=cv2.contourArea)
            xx, yy, ww, hh = cv2.boundingRect(cc)
    
            estimated_bbox = np.zeros([1,4],dtype=int)
            estimated_bbox[0,1] = yy
            estimated_bbox[0,3] = yy+hh
            estimated_bbox[0,0] = xx
            estimated_bbox[0,2] = xx+ww


            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 0] = 0
            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 1] = 255
            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 2] = 0

            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 0] = 0
            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 1] = 255
            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 2] = 0

            superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 0] = 0
            superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 1] = 255
            superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 2] = 0

            superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 0] = 0
            superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 1] = 255
            superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 2] = 0



            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 0] = 0
            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 1] = 0
            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 2] = 255

            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 0] = 0
            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 1] = 0
            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 2] = 255

            superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 0] = 0
            superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 1] = 0
            superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 2] = 255

            superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 0] = 0
            superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 1] = 0
            superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 2] = 255



            if gt_label[0][i]-1 == predict_cls[i]:

                cv2.imwrite('./{}_{}_T.png'.format(index, i), superimposed_img_1)
            else:
                cv2.imwrite('./{}_{}_F.png'.format(index, i), superimposed_img_1)

            # pdb.set_trace()



    def top1_loc_imagenet(self, batch_imgs, gt_bbox, gt, ori_size):


        out_extra = self.features(batch_imgs)



        predict_cls = gt[0]
        # pdb.set_trace()
        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        # W = W/torch.sum(W)
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        threshold_list = np.arange(0,1,0.01)
        # threshold_list = np.array([0.2])
        counter_03 = np.zeros(len(threshold_list))
        counter_05 = np.zeros(len(threshold_list))
        counter_07 = np.zeros(len(threshold_list))

        if predict_cls[0] == gt[0][0]:

            for ii in range(batch_imgs.shape[0]):
            # save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            # ref_1 = cv2.imread('./temp_1.png')
                # pdb.set_trace()
                c_hm = hm[ii].unsqueeze(0)
                c_hm = F.interpolate(c_hm, size=(ori_size[ii,1],ori_size[ii,0]), mode='bilinear')
                c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))
                


                for p in range(gt_bbox.shape[1]):



                    if gt_bbox[ii, p].shape[1] != 0:

                        c_gt_bbox =  np.zeros([1,4])


                        c_gt_bbox[0,0] = gt_bbox[ii,p][0][0]-1
                        c_gt_bbox[0,1] = gt_bbox[ii,p][0][1]-1
                        c_gt_bbox[0,2] = gt_bbox[ii,p][0][0]-1 + gt_bbox[ii,p][0][2]
                        c_gt_bbox[0,3] = gt_bbox[ii,p][0][1]-1 + gt_bbox[ii,p][0][3]


                        for k in range(len(counter_03)):

                            if counter_05[k] * counter_03[k] * counter_07[k] == 1:

                                continue

                            else:

                                cm_ = 255*c_hm.cpu().numpy()[0][0]
                                cm_ = cm_.astype('uint8')
                                threshold_value = int(np.max(cm_) * threshold_list[k])
                                _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


                                contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                                # for kk in range
                                # pdb.set_trace()
                                for kk in range(len(contours)):
                                # for kk in range(1):

                                    # cc = max(contours, key=cv2.contourArea)
                                    cc = contours[kk]
                                    xx, yy, ww, hh = cv2.boundingRect(cc)
    
                                    estimated_bbox = np.zeros([1,4])
                                    estimated_bbox[0,1] = yy
                                    estimated_bbox[0,3] = yy+hh
                                    estimated_bbox[0,0] = xx
                                    estimated_bbox[0,2] = xx+ww


                                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.5:

                                        if counter_05[k] == 0:
                                            counter_05[k] += 1

                                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.3:
                                        if counter_03[k] == 0:
                                            counter_03[k] += 1
                                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.7:
                                        if counter_07[k] == 0:
                                            counter_07[k] += 1

                    else:

                        break



            return counter_03, counter_05, counter_07

        else:
            return 0



    def top1_loc(self, batch_imgs, gt_bbox, gt, ori_size):

        out_extra = self.features(batch_imgs)
        predict_cls = gt[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        # W = W/torch.sum(W)
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        threshold_list = np.arange(0,1,0.01)
        # threshold_list = np.array([0.2])
        counter_03 = np.zeros(len(threshold_list))
        counter_05 = np.zeros(len(threshold_list))
        counter_07 = np.zeros(len(threshold_list))

        if predict_cls[0] == gt[0][0]-1:

            for ii in range(batch_imgs.shape[0]):
            # save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            # ref_1 = cv2.imread('./temp_1.png')
                c_hm = hm[ii].unsqueeze(0)
                c_hm = F.interpolate(c_hm, size=(ori_size[ii,0],ori_size[ii,1]), mode='bilinear')
                c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))
                c_gt_bbox =  np.zeros([1,4])

                c_gt_bbox[0,0] = gt_bbox[ii,:][0]
                c_gt_bbox[0,1] = gt_bbox[ii,:][1]
                c_gt_bbox[0,2] = gt_bbox[ii,:][0] + gt_bbox[ii,:][2]
                c_gt_bbox[0,3] = gt_bbox[ii,:][1] + gt_bbox[ii,:][3]

                # iouu = np.zeros([100])

                for k in range(len(counter_03)):

                    cm_ = 255*c_hm.cpu().numpy()[0][0]
                    cm_ = cm_.astype('uint8')
                    threshold_value = int(np.max(cm_) * threshold_list[k])
                    _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


                    contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cc = max(contours, key=cv2.contourArea)
                    xx, yy, ww, hh = cv2.boundingRect(cc)
    
                    estimated_bbox = np.zeros([1,4])
                    estimated_bbox[0,1] = yy
                    estimated_bbox[0,3] = yy+hh
                    estimated_bbox[0,0] = xx
                    estimated_bbox[0,2] = xx+ww

                    # c_hm_ = c_hm >= (torch.max(c_hm)*threshold_list[k])
                    # c_hm_ = c_hm_[0,0,:,:]
                    # c_hm_ = c_hm_.cpu().numpy()
                    # yy, xx = np.where(c_hm_==True)

                    # estimated_bbox = np.zeros([1,4])
                    # estimated_bbox[0,1] = np.min(yy)
                    # estimated_bbox[0,3] = np.max(yy)
                    # estimated_bbox[0,0] = np.min(xx)
                    # estimated_bbox[0,2] = np.max(xx)



                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.5:
                        counter_05[k] += 1
                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.3:
                        counter_03[k] += 1
                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.7:
                        counter_07[k] += 1

                # c_gt_bbox = np.array([gt_bbox[i,:]-1])
                # pdb.set_trace()
                # if np.max(iouu) > 0.5:
                #     counter += 1


                # if bbox_iou(c_gt_bbox,estimated_bbox)[0]>0.5:
                #     counter += 1

                # counter = counter+1


            return counter_03, counter_05, counter_07

        else:
            return 0

 

    def top1_loc_auc(self, batch_imgs, gt, mask_path):

        out_extra = self.features(batch_imgs)

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        # W = W/torch.sum(W)
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        threshold_list = np.arange(0,1,0.001)
        # threshold_list = np.array([0.2])
        counter_03 = np.zeros(len(threshold_list))
        counter_05 = np.zeros(len(threshold_list))
        counter_07 = np.zeros(len(threshold_list))

        num_bins = len(threshold_list) + 2
        threshold_list_right_edge = np.append(threshold_list,
                                                   [1.0, 2.0, 3.0])


        gt_true_score_hist = np.zeros(num_bins, dtype=np.float)
        gt_false_score_hist = np.zeros(num_bins, dtype=np.float)


        if predict_cls[0] == gt[0][0]-1:

            auc_ = 0

            for ii in range(batch_imgs.shape[0]):

                c_hm = hm[ii].unsqueeze(0)
                c_hm = F.interpolate(c_hm, size=(224,224), mode='bilinear')
                c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))

                precision = np.zeros([threshold_list.shape[0]])
                recall = np.zeros([threshold_list.shape[0]])

                c_mask_path = mask_path[ii]
                mask_path_ = []

                for kk in range(len(c_mask_path)):
                    cc_path = c_mask_path[kk]
                    # pdb.set_trace()
                    if cc_path.split('_')[-1] == 'ignore.png':
                        ignore_path_ = cc_path
                    else:
                        mask_path_.append(cc_path)

                c_gt_mask = get_mask(mask_path_, ignore_path_)
                # print(ignore_path_)
                # print(mask_path_)
                # print('\n')
                # pdb.set_trace()

                # for k in range(len(counter_03)):


                #     c_hm_ = c_hm >= threshold_list[k]

                #     # gt_true_scores = c_hm_[gt_mask == 1]
                #     pdb.set_trace()
                #     c_hm_ = c_hm_[0,0].detach().cpu().numpy()

                #     gt_true_scores = c_hm_[c_gt_mask == 1]





                    

                #     c_gt_mask = c_gt_mask.astype('bool')

                #     precision[k] = np.sum(c_hm_ * c_gt_mask)/np.sum(c_hm_)
                #     recall[k] = np.sum(c_hm_ * c_gt_mask)/np.sum(c_gt_mask)



                c_hm = c_hm[0,0].detach().cpu().numpy()
                gt_true_scores = c_hm[c_gt_mask == 1]
                gt_false_scores = c_hm[c_gt_mask == 0]

                gt_true_hist, _ = np.histogram(gt_true_scores, bins=threshold_list_right_edge)
                gt_true_score_hist += gt_true_hist.astype(np.float)
                
                gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=threshold_list_right_edge)
                gt_false_score_hist += gt_false_hist.astype(np.float)

                # pdb.set_trace()



                


            return gt_true_score_hist, gt_false_score_hist

        else:
            return 0

    def top1_loc_auc_2(self, gt_true_score_hist, gt_false_score_hist):


        # pdb.set_trace()

        num_gt_true = gt_true_score_hist.sum()
        tp = gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = gt_false_score_hist.sum()
        fp = gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2*precision*recall / (recall + precision)



        np.save('./mobilenet_prec.py', precision)
        np.save('./mobilenet_recall.py', recall)

        pdb.set_trace()

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        # auc *= 100

        # print("Mask AUC on split {}: {}".format(self.split, auc))
        return auc

    def loss_common_part(self, repre):
        # print('hi')

        # repre_1 = torch.zeros([len(self.cc), 4096]).cuda()
        # repre_2 = torch.zeros([len(self.cc), 4096]).cuda()
        c_loss = torch.tensor([0.]).cuda()
        for i in range(len(self.cc)):

            # pdb.set_trace()
            c_loss += self.mse(repre[int(self.cc[i,0])].unsqueeze(0), repre[int(self.cc[i,1])].unsqueeze(0))

        return c_loss/len(self.cc)
        # pdb.set_trace()


    def loss_ori_img(self, repre, repre_ori):
        # print('hi')

        # pdb.set_trace()
        loss = torch.mean(self.mse(repre, repre_ori))
        return loss



    def show_hm(self, batch_imgs, index, gt_label, gt_bbox, ori_size):
        # pdb.set_trace()
        out_p5 = self.features(batch_imgs)
        # out_p5 = torch.relu(self.extra_conv(out_p5))
        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_p5, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        for i in range(batch_imgs.shape[0]):
            save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            ref_1 = cv2.imread('./temp_1.png')
            ref_1 = cv2.resize(ref_1, (ori_size[i,1],ori_size[i,0]))
            c_hm = F.interpolate(hm[i].unsqueeze(0), size=(ori_size[i,0],ori_size[i,1]), mode='bilinear')
            # c_hm = w_scale[i].unsqueeze(0)
            # pdb.set_trace()
            c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # pdb.set_trace
            # c_hm = c_hm >= 0.23
            # c_hm = c_hm >= 0.32


            heatmap = np.uint8(255 * c_hm[0][0].cpu().detach().numpy())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img_1 = heatmap * 0.7 + ref_1 *0.5

            # c_hm = w_scale[i].unsqueeze(0)
            # c_hm = F.interpolate(c_hm, size=(ori_size[i,0],ori_size[i,1]), mode='bilinear')
            # # pdb.set_trace()
            # c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.2

            c_hm = c_hm[0,0,:,:]
            c_hm = c_hm.cpu().numpy()
            yy, xx = np.where(c_hm==True)

            estimated_bbox = np.zeros([1,4])
            c_gt_bbox =  np.zeros([1,4])
            estimated_bbox[0,1] = np.min(yy)
            estimated_bbox[0,3] = np.max(yy)
            estimated_bbox[0,0] = np.min(xx)
            estimated_bbox[0,2] = np.max(xx)



                # c_gt_bbox = np.array([gt_bbox[i,:]-1])
            # pdb.set_trace()    
            # gt_bbox
            c_gt_bbox[0,0] = gt_bbox[i,:][0]-1
            c_gt_bbox[0,1] = gt_bbox[i,:][1]-1
            c_gt_bbox[0,2] = gt_bbox[i,:][0]-1 + gt_bbox[i,:][2]
            c_gt_bbox[0,3] = gt_bbox[i,:][1]-1 + gt_bbox[i,:][3]

            iou = bbox_iou(c_gt_bbox,estimated_bbox)[0]
            # pdb.set_trace()

            if gt_label[0][i]-1 == predict_cls[i]:

                cv2.imwrite('./{}_{}_{}_T.png'.format(index, i, iou), superimposed_img_1)
            else:
                cv2.imwrite('./{}_{}_{}_F.png'.format(index, i, iou), superimposed_img_1)
            # counter += 1
        # pdb.set_trace()





    def show_hm_openimage(self, batch_imgs, index, gt_label):
        # pdb.set_trace()
        out_extra = self.base(batch_imgs)
        pred = self.pred(torch.mean(out_extra, dim=(2,3)))

        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')

        # pdb.set_trace()

        for i in range(batch_imgs.shape[0]):
            save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            ref_1 = cv2.imread('./temp_1.png')
            ref_1 = cv2.resize(ref_1, (224,224))
            c_hm = F.interpolate(hm[i].unsqueeze(0), size=(224,224), mode='bilinear')
            # c_hm = w_scale[i].unsqueeze(0)
            # pdb.set_trace()
            c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.2

            heatmap = np.uint8(255 * c_hm[0][0].cpu().detach().numpy())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img_1 = heatmap * 0.7 + ref_1 *0.5

            cv2.imwrite('./{}_{}.png'.format(index, i), superimposed_img_1)




class Minmaxcam_VGG(nn.Module):
    def __init__(self, base_net="vgg", set_size = 5, numclass=200):
        super().__init__()



        self.numclass = numclass


        self.base = vgg16(pretrained=True)  
        self.features = self.base.features[:-1]
        self.extra_conv = nn.Conv2d(512, 1024, 3, 1, 1)
        self.pred = nn.Linear(1024, self.numclass)


        self.gap = nn.AvgPool2d(14, stride=14)

        self.set_size = set_size

        self.aa = list(range(0, self.set_size))
        self.bb = list(itertools.combinations(self.aa, 2))
        self.cc = np.zeros([len(self.bb),2])
        for i in range(len(self.bb)):
            self.cc[i,0] = self.bb[i][0]
            self.cc[i,1] = self.bb[i][1]
        # self.cc = int(self.cc)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.mse = nn.MSELoss()







    def update_classification(self, batch_imgs, label, ss, bs):

        for param in self.features.parameters():
            param.requires_grad = True

        for param in self.extra_conv.parameters():
            param.requires_grad = True

        for param in self.pred.parameters():
            param.requires_grad = True



        out_p5 = self.features(batch_imgs)
        out_extra = torch.relu(self.extra_conv(out_p5))
        pred = self.pred(torch.mean(out_extra, dim=(2,3)))
       

        #################################   

        loss_img = F.cross_entropy(pred, torch.transpose(label.repeat(1,ss).view(ss,bs),1,0).reshape(1,ss*bs)[0], reduction='mean')
        # pdb.set_trace()
        # loss_img = F.binary_cross_entropy(F.sigmoid(pred), 1.*label[0], reduction="mean")
        return loss_img





    def get_hms(self, batch_imgs, label, ss, bs):




        out_p5 = self.features(batch_imgs)
        out_extra = torch.relu(self.extra_conv(out_p5))
        pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        predict_cls = torch.transpose(label.repeat(1,ss).view(ss,bs),1,0).reshape(1,ss*bs)[0]
        # predict_cls = torch.argmax(pred, dim=0)

        for i in range(ss*bs):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)

        # pdb.set_trace()

        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)


        min_tmp = torch.min(hm, dim=2)[0]
        min_tmp = torch.min(min_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)
        max_tmp = torch.max(hm, dim=2)[0]
        max_tmp = torch.max(max_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)


        hm = (hm - min_tmp)/(max_tmp - min_tmp)



        hm = F.interpolate(hm, size=(224,224), mode='bilinear')

        pdb.set_trace()

        return hm



    def update_pwnn(self, batch_imgs, label, ss, bs):


        for param in self.features.parameters():
            param.requires_grad = False

        for param in self.extra_conv.parameters():
            param.requires_grad = False

        for param in self.pred.parameters():
            param.requires_grad = True


        out_p5 = self.features(batch_imgs)
        out_extra = torch.relu(self.extra_conv(out_p5))
        pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        predict_cls = torch.transpose(label.repeat(1,ss).view(ss,bs),1,0).reshape(1,ss*bs)[0]
        # predict_cls = torch.argmax(pred, dim=0)

        for i in range(ss*bs):
            if i == 0:

                W = self.pred.weight[predict_cls[i]].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)

        # pdb.set_trace()

        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)


        min_tmp = torch.min(hm, dim=2)[0]
        min_tmp = torch.min(min_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)
        max_tmp = torch.max(hm, dim=2)[0]
        max_tmp = torch.max(max_tmp, dim=2)[0].unsqueeze(2).unsqueeze(2)


        hm = (hm - min_tmp)/(max_tmp - min_tmp)

        ################################################
        # out_extra_masked = out_extra*hm
        # repre_cam = self.gap(out_extra_masked).squeeze(2).squeeze(2)
        # repre = self.gap(out_extra).squeeze(2).squeeze(2)
        # ################################################
        hm = F.interpolate(hm, size=(224,224), mode='bilinear')
        out_p5_hm = self.features(batch_imgs*hm)
        out_extra_masked = torch.relu(self.extra_conv(out_p5_hm))

        repre_cam = self.gap(out_extra_masked).squeeze(2).squeeze(2)
        repre = self.gap(out_extra).squeeze(2).squeeze(2)
        ################################################

        # pdb.set_trace()


        for i in range(bs):
            c_loss_common = self.loss_common_part(repre_cam[i*ss:ss*(i+1)]) 
            c_loss_ori = self.loss_ori_img(repre_cam[ss*i:ss*(i+1)], repre[ss*i:ss*(i+1)]) 
            if i == 0:
                loss_common = c_loss_common
                loss_ori = c_loss_ori
            else:
                loss_common += c_loss_common
                loss_ori += c_loss_ori

        loss_common /= bs
        loss_ori /= bs

        # pdb.set_trace()

        return loss_common, loss_ori



 

    def show_hm_openimage(self, batch_imgs, index, gt_label):
        # pdb.set_trace()
        out_extra = self.features(batch_imgs)
        out_extra = torch.relu(self.extra_conv(out_extra))

        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')

        # pdb.set_trace()

        for i in range(batch_imgs.shape[0]):
            save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            ref_1 = cv2.imread('./temp_1.png')
            ref_1 = cv2.resize(ref_1, (224,224))
            c_hm = F.interpolate(hm[i].unsqueeze(0), size=(224,224), mode='bilinear')
            # c_hm = w_scale[i].unsqueeze(0)
            # pdb.set_trace()
            c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.2

            heatmap = np.uint8(255 * c_hm[0][0].cpu().detach().numpy())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img_1 = heatmap * 0.7 + ref_1 *0.5

            cv2.imwrite('./{}_{}.png'.format(index, i), superimposed_img_1)





    def top1_loc(self, batch_imgs, gt_bbox, gt, ori_size):

        out_p5 = self.features(batch_imgs)
        out_extra = torch.relu(self.extra_conv(out_p5))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                # pdb.set_trace()
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        # W = W/torch.sum(W)
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        threshold_list = np.arange(0,1,0.01)
        # threshold_list = np.array([0.2])
        counter_03 = np.zeros(len(threshold_list))
        counter_05 = np.zeros(len(threshold_list))
        counter_07 = np.zeros(len(threshold_list))

        if predict_cls[0] == gt[0][0]-1:

            for ii in range(batch_imgs.shape[0]):
            # save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            # ref_1 = cv2.imread('./temp_1.png')
                c_hm = hm[ii].unsqueeze(0)
                c_hm = F.interpolate(c_hm, size=(ori_size[ii,0],ori_size[ii,1]), mode='bilinear')
                c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))
                c_gt_bbox =  np.zeros([1,4])

                c_gt_bbox[0,0] = gt_bbox[ii,:][0]
                c_gt_bbox[0,1] = gt_bbox[ii,:][1]
                c_gt_bbox[0,2] = gt_bbox[ii,:][0] + gt_bbox[ii,:][2]
                c_gt_bbox[0,3] = gt_bbox[ii,:][1] + gt_bbox[ii,:][3]

                # iouu = np.zeros([100])

                for k in range(len(counter_03)):

                    cm_ = 255*c_hm.cpu().numpy()[0][0]
                    cm_ = cm_.astype('uint8')
                    threshold_value = int(np.max(cm_) * threshold_list[k])
                    _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


                    contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cc = max(contours, key=cv2.contourArea)
                    xx, yy, ww, hh = cv2.boundingRect(cc)
    
                    estimated_bbox = np.zeros([1,4])
                    estimated_bbox[0,1] = yy
                    estimated_bbox[0,3] = yy+hh
                    estimated_bbox[0,0] = xx
                    estimated_bbox[0,2] = xx+ww

                    # c_hm_ = c_hm >= (torch.max(c_hm)*threshold_list[k])
                    # c_hm_ = c_hm_[0,0,:,:]
                    # c_hm_ = c_hm_.cpu().numpy()
                    # yy, xx = np.where(c_hm_==True)

                    # estimated_bbox = np.zeros([1,4])
                    # estimated_bbox[0,1] = np.min(yy)
                    # estimated_bbox[0,3] = np.max(yy)
                    # estimated_bbox[0,0] = np.min(xx)
                    # estimated_bbox[0,2] = np.max(xx)



                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.5:
                        counter_05[k] += 1
                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.3:
                        counter_03[k] += 1
                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.7:
                        counter_07[k] += 1

                # c_gt_bbox = np.array([gt_bbox[i,:]-1])
                # pdb.set_trace()
                # if np.max(iouu) > 0.5:
                #     counter += 1


                # if bbox_iou(c_gt_bbox,estimated_bbox)[0]>0.5:
                #     counter += 1

                # counter = counter+1


            return counter_03, counter_05, counter_07

        else:
            return 0


  

    def top1_loc_imagenet(self, batch_imgs, gt_bbox, gt, ori_size):
        # pdb.set_trace()

        out_p5 = self.features(batch_imgs)
        out_extra = torch.relu(self.extra_conv(out_p5))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt[0]
        # pdb.set_trace()
        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        # W = W/torch.sum(W)
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        threshold_list = np.arange(0,1,0.01)
        # threshold_list = np.array([0.2])
        counter_03 = np.zeros(len(threshold_list))
        counter_05 = np.zeros(len(threshold_list))
        counter_07 = np.zeros(len(threshold_list))

        if predict_cls[0] == gt[0][0]:

            for ii in range(batch_imgs.shape[0]):
            # save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            # ref_1 = cv2.imread('./temp_1.png')
                # pdb.set_trace()
                c_hm = hm[ii].unsqueeze(0)
                c_hm = F.interpolate(c_hm, size=(ori_size[ii,1],ori_size[ii,0]), mode='bilinear')
                c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))
                


                for p in range(gt_bbox.shape[1]):



                    if gt_bbox[ii, p].shape[1] != 0:

                        c_gt_bbox =  np.zeros([1,4])


                        c_gt_bbox[0,0] = gt_bbox[ii,p][0][0]-1
                        c_gt_bbox[0,1] = gt_bbox[ii,p][0][1]-1
                        c_gt_bbox[0,2] = gt_bbox[ii,p][0][0]-1 + gt_bbox[ii,p][0][2]
                        c_gt_bbox[0,3] = gt_bbox[ii,p][0][1]-1 + gt_bbox[ii,p][0][3]


                        for k in range(len(counter_03)):

                            if counter_05[k] * counter_03[k] * counter_07[k] == 1:

                                continue

                            else:

                                cm_ = 255*c_hm.cpu().numpy()[0][0]
                                cm_ = cm_.astype('uint8')
                                threshold_value = int(np.max(cm_) * threshold_list[k])
                                _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


                                contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                                # for kk in range
                                # pdb.set_trace()
                                # for kk in range(len(contours)):
                                for kk in range(1):

                                    cc = max(contours, key=cv2.contourArea)
                                    # cc = contours[kk]
                                    xx, yy, ww, hh = cv2.boundingRect(cc)
    
                                    estimated_bbox = np.zeros([1,4])
                                    estimated_bbox[0,1] = yy
                                    estimated_bbox[0,3] = yy+hh
                                    estimated_bbox[0,0] = xx
                                    estimated_bbox[0,2] = xx+ww


                                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.5:

                                        if counter_05[k] == 0:
                                            counter_05[k] += 1

                                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.3:
                                        if counter_03[k] == 0:
                                            counter_03[k] += 1
                                    if bbox_iou(c_gt_bbox,estimated_bbox)[0] > 0.7:
                                        if counter_07[k] == 0:
                                            counter_07[k] += 1

                    else:

                        break



            return counter_03, counter_05, counter_07

        else:
            return 0





    def top1_loc_auc(self, batch_imgs, gt, mask_path):

        out_p5 = self.features(batch_imgs)
        out_extra = torch.relu(self.extra_conv(out_p5))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[predict_cls[i]].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        # W = W/torch.sum(W)
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        threshold_list = np.arange(0,1,0.001)
        # threshold_list = np.array([0.2])
        counter_03 = np.zeros(len(threshold_list))
        counter_05 = np.zeros(len(threshold_list))
        counter_07 = np.zeros(len(threshold_list))

        num_bins = len(threshold_list) + 2
        threshold_list_right_edge = np.append(threshold_list,
                                                   [1.0, 2.0, 3.0])


        gt_true_score_hist = np.zeros(num_bins, dtype=np.float)
        gt_false_score_hist = np.zeros(num_bins, dtype=np.float)


        if predict_cls[0] == gt[0][0]-1:

            auc_ = 0

            for ii in range(batch_imgs.shape[0]):

                c_hm = hm[ii].unsqueeze(0)
                c_hm = F.interpolate(c_hm, size=(batch_imgs.shape[2],batch_imgs.shape[2]), mode='bilinear')
                c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))

                precision = np.zeros([threshold_list.shape[0]])
                recall = np.zeros([threshold_list.shape[0]])

                c_mask_path = mask_path[ii]
                mask_path_ = []

                for kk in range(len(c_mask_path)):
                    cc_path = c_mask_path[kk]
                    # pdb.set_trace()
                    if cc_path.split('_')[-1] == 'ignore.png':
                        ignore_path_ = cc_path
                    else:
                        mask_path_.append(cc_path)

                # pdb.set_trace()
                c_gt_mask = get_mask(mask_path_, ignore_path_)



                c_hm = c_hm[0,0].detach().cpu().numpy()
                gt_true_scores = c_hm[c_gt_mask == 1]
                gt_false_scores = c_hm[c_gt_mask == 0]

                gt_true_hist, _ = np.histogram(gt_true_scores, bins=threshold_list_right_edge)
                gt_true_score_hist += gt_true_hist.astype(np.float)
                
                gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=threshold_list_right_edge)
                gt_false_score_hist += gt_false_hist.astype(np.float)

                # pdb.set_trace()



                


            return gt_true_score_hist, gt_false_score_hist

        else:
            return 0



    def top1_loc_auc_2(self, gt_true_score_hist, gt_false_score_hist):



        num_gt_true = gt_true_score_hist.sum()
        tp = gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = gt_false_score_hist.sum()
        fp = gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        np.save('./vgg_ours_prec.py', precision)
        np.save('./vgg_ours_recall.py', recall)


        pdb.set_trace()


        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        # auc *= 100

        # print("Mask AUC on split {}: {}".format(self.split, auc))
        return auc



    def loss_common_part(self, repre):
        # print('hi')

        # repre_1 = torch.zeros([len(self.cc), 4096]).cuda()
        # repre_2 = torch.zeros([len(self.cc), 4096]).cuda()
        c_loss = torch.tensor([0.]).cuda()
        for i in range(len(self.cc)):

            # pdb.set_trace()
            c_loss += self.mse(repre[int(self.cc[i,0])].unsqueeze(0), repre[int(self.cc[i,1])].unsqueeze(0))

        return c_loss/len(self.cc)
        # pdb.set_trace()

    def loss_ori_img(self, repre, repre_ori):
        # print('hi')

        # pdb.set_trace()
        loss = torch.mean(self.mse(repre, repre_ori))
        return loss



    def show_hm(self, batch_imgs, index, gt_label, gt_bbox, ori_size):
        # pdb.set_trace()
        out_p5 = self.features(batch_imgs)
        out_extra = torch.relu(self.extra_conv(out_p5))
        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')


        # for i in range(hm.shape[0]):
        #     c_hm = hm[i].unsqueeze(0)
        #     c_hm = (c_hm - torch.min(c_hm)) /(torch.max(c_hm) - torch.min(c_hm))
        #     c_hm =  F.interpolate(c_hm, size=(224,224), mode='bilinear')

        #     c_hm = c_hm.cpu().numpy()

        #     c_hm = c_hm[0][0]
        #     # pdb.set_trace()

        #     cv2.imwrite('test_{}.png'.format(i), 255*c_hm)

        # pdb.set_trace()

        for i in range(batch_imgs.shape[0]):
            save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            ref_1 = cv2.imread('./temp_1.png')
            ref_1 = cv2.resize(ref_1, (ori_size[i,1],ori_size[i,0]))
            c_hm = F.interpolate(hm[i].unsqueeze(0), size=(ori_size[i,0],ori_size[i,1]), mode='bilinear')
            # c_hm = w_scale[i].unsqueeze(0)
            # pdb.set_trace()
            c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.36

            heatmap = np.uint8(255 * c_hm[0][0].cpu().detach().numpy())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img_1 = heatmap * 0.7 + ref_1 *0.5

            # c_hm = w_scale[i].unsqueeze(0)
            # c_hm = F.interpolate(c_hm, size=(ori_size[i,0],ori_size[i,1]), mode='bilinear')
            # # pdb.set_trace()
            # c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.36

            c_hm = c_hm[0,0,:,:]
            c_hm = c_hm.cpu().numpy()
            yy, xx = np.where(c_hm==True)

            estimated_bbox = np.zeros([1,4])
            c_gt_bbox =  np.zeros([1,4])
            estimated_bbox[0,1] = np.min(yy)
            estimated_bbox[0,3] = np.max(yy)
            estimated_bbox[0,0] = np.min(xx)
            estimated_bbox[0,2] = np.max(xx)



                # c_gt_bbox = np.array([gt_bbox[i,:]-1])
            # pdb.set_trace()    
            # gt_bbox
            c_gt_bbox[0,0] = gt_bbox[i,:][0]-1
            c_gt_bbox[0,1] = gt_bbox[i,:][1]-1
            c_gt_bbox[0,2] = gt_bbox[i,:][0]-1 + gt_bbox[i,:][2]
            c_gt_bbox[0,3] = gt_bbox[i,:][1]-1 + gt_bbox[i,:][3]

            iou = bbox_iou(c_gt_bbox,estimated_bbox)[0]
            # pdb.set_trace()

            if gt_label[0][i]-1 == predict_cls[i]:

                cv2.imwrite('./{}_{}_{}_T.png'.format(index, i, iou), superimposed_img_1)
            else:
                cv2.imwrite('./{}_{}_{}_F.png'.format(index, i, iou), superimposed_img_1)




    def show_hm_bbox(self, batch_imgs, index, gt_label, gt_bbox, ori_size):
        # pdb.set_trace()
        out_p5 = self.features(batch_imgs)
        out_extra = torch.relu(self.extra_conv(out_p5))
        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')

                # c_hm = hm[ii].unsqueeze(0)
                # c_hm = F.interpolate(c_hm, size=(ori_size[ii,0],ori_size[ii,1]), mode='bilinear')
                # c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm) - torch.min(c_hm))


                # # iouu = np.zeros([100])

                # for k in range(len(counter_03)):





        for i in range(batch_imgs.shape[0]):
            save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            ref_1 = cv2.imread('./temp_1.png')
            ref_1 = cv2.resize(ref_1, (ori_size[i,1],ori_size[i,0]))
            c_hm = F.interpolate(hm[i].unsqueeze(0), size=(ori_size[i,0],ori_size[i,1]), mode='bilinear')
            # c_hm = w_scale[i].unsqueeze(0)
            # pdb.set_trace()
            c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.36

            heatmap = np.uint8(255 * c_hm[0][0].cpu().detach().numpy())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img_1 = heatmap * 0.7 + ref_1 *0.5

            c_gt_bbox =  np.zeros([1,4], dtype=int)

            # pdb.set_trace()

            c_gt_bbox[0,0] = int(gt_bbox[0,:][0]-1) #x1
            c_gt_bbox[0,1] = gt_bbox[0,:][1]-1 #y1
            c_gt_bbox[0,2] = gt_bbox[0,:][0]-1 + gt_bbox[0,:][2] #x2
            c_gt_bbox[0,3] = gt_bbox[0,:][1]-1 + gt_bbox[0,:][3] #y2



            cm_ = 255*c_hm.cpu().numpy()[0][0]
            cm_ = cm_.astype('uint8')
            threshold_value = int(np.max(cm_) * 0.36)
            _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


            contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cc = max(contours, key=cv2.contourArea)
            xx, yy, ww, hh = cv2.boundingRect(cc)
    
            estimated_bbox = np.zeros([1,4],dtype=int)
            estimated_bbox[0,1] = yy
            estimated_bbox[0,3] = yy+hh
            estimated_bbox[0,0] = xx
            estimated_bbox[0,2] = xx+ww


            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 0] = 0
            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 1] = 255
            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 2] = 0

            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 0] = 0
            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 1] = 255
            superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 2] = 0

            superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 0] = 0
            superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 1] = 255
            superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 2] = 0

            superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 0] = 0
            superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 1] = 255
            superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 2] = 0



            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 0] = 0
            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 1] = 0
            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 2] = 255

            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 0] = 0
            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 1] = 0
            superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 2] = 255

            superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 0] = 0
            superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 1] = 0
            superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 2] = 255

            superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 0] = 0
            superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 1] = 0
            superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 2] = 255



            if gt_label[0][i]-1 == predict_cls[i]:

                cv2.imwrite('./{}_{}_T.png'.format(index, i), superimposed_img_1)
            else:
                cv2.imwrite('./{}_{}_F.png'.format(index, i), superimposed_img_1)

            # pdb.set_trace()




    def show_hm_bbox_imagenet(self, batch_imgs, index, gt_label, gt_bbox, ori_size):

        out_p5 = self.features(batch_imgs)
        out_extra = torch.relu(self.extra_conv(out_p5))
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)



        for ii in range(batch_imgs.shape[0]):
            save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            ref_1 = cv2.imread('./temp_1.png')
            ref_1 = cv2.resize(ref_1, (ori_size[i,0],ori_size[i,1]))
            c_hm = F.interpolate(hm[i].unsqueeze(0), size=(ori_size[i,1],ori_size[i,0]), mode='bilinear')
            # c_hm = w_scale[i].unsqueeze(0)
            # pdb.set_trace()
            c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.36

            heatmap = np.uint8(255 * c_hm[0][0].cpu().detach().numpy())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img_1 = heatmap * 0.7 + ref_1 *0.5


            for p in range(gt_bbox.shape[1]):





                if gt_bbox[ii, p].shape[1] != 0:

                    c_gt_bbox =  np.zeros([1,4], dtype=int)


                    c_gt_bbox[0,0] = gt_bbox[ii,p][0][0]-1
                    c_gt_bbox[0,1] = gt_bbox[ii,p][0][1]-1
                    c_gt_bbox[0,2] = gt_bbox[ii,p][0][0]-1 + gt_bbox[ii,p][0][2]
                    c_gt_bbox[0,3] = gt_bbox[ii,p][0][1]-1 + gt_bbox[ii,p][0][3]

                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 0] = 0
                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 1] = 255
                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,0]:c_gt_bbox[0,0]+3, 2] = 0

                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 0] = 0
                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 1] = 255
                    superimposed_img_1[c_gt_bbox[0,1] : c_gt_bbox[0,3], c_gt_bbox[0,2]:c_gt_bbox[0,2]+3, 2] = 0

                    superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 0] = 0
                    superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 1] = 255
                    superimposed_img_1[c_gt_bbox[0,1]:c_gt_bbox[0,1]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 2] = 0

                    superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 0] = 0
                    superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 1] = 255
                    superimposed_img_1[c_gt_bbox[0,3]:c_gt_bbox[0,3]+3, c_gt_bbox[0,0]: c_gt_bbox[0,2], 2] = 0


                    cm_ = 255*c_hm.cpu().numpy()[0][0]
                    cm_ = cm_.astype('uint8')
                    threshold_value = int(np.max(cm_) * 0.19)
                    # threshold_value = int(np.max(cm_) * 0.28)
                    # threshold_value = int(np.max(cm_) * 0.27)

                    _, thresholded_gray_heatmap = cv2.threshold(cm_, threshold_value, 255, cv2.THRESH_TOZERO)


                    contours, _ = cv2.findContours(thresholded_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            

                    # for kk in range(len(contours)):
                    for kk in range(1):

                        # cc = contours[kk]
                        cc = max(contours, key=cv2.contourArea)
                        xx, yy, ww, hh = cv2.boundingRect(cc)

                        # xx, yy, ww, hh = cv2.boundingRect(cc)
    
                        estimated_bbox = np.zeros([1,4],dtype=int)
                        estimated_bbox[0,1] = yy
                        estimated_bbox[0,3] = yy+hh
                        estimated_bbox[0,0] = xx
                        estimated_bbox[0,2] = xx+ww
    
        
                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 0] = 0
                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 1] = 0
                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,0]:estimated_bbox[0,0]+3, 2] = 255

                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 0] = 0
                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 1] = 0
                        superimposed_img_1[estimated_bbox[0,1] : estimated_bbox[0,3], estimated_bbox[0,2]:estimated_bbox[0,2]+3, 2] = 255

                        superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 0] = 0
                        superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 1] = 0
                        superimposed_img_1[estimated_bbox[0,1]:estimated_bbox[0,1]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 2] = 255

                        superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 0] = 0
                        superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 1] = 0
                        superimposed_img_1[estimated_bbox[0,3]:estimated_bbox[0,3]+3, estimated_bbox[0,0]: estimated_bbox[0,2], 2] = 255




            cv2.imwrite('./{}_{}_T.png'.format(index, i), superimposed_img_1)
      
            # pdb.set_trace()





    def show_hm_imagenet(self, batch_imgs, index, gt_label, gt_bbox, ori_size):
        # pdb.set_trace()
        out_p5 = self.features(batch_imgs)
        out_extra = torch.relu(self.extra_conv(out_p5))
        # pred = self.pred(self.gap(out_extra).squeeze(2).squeeze(2))

        # predict_cls = torch.argmax(pred, dim=1)
        predict_cls = gt_label[0]-1

        for i in range(1):
            if i == 0:

                W = self.pred.weight[int(predict_cls[i])].unsqueeze(0)
            else:
                W = torch.cat((W, self.pred.weight[int(predict_cls[i])].unsqueeze(0)),dim=0)


        # pdb.set_trace()
        hm = torch.sum(W.unsqueeze(2).unsqueeze(2) * out_extra, dim=1).unsqueeze(1)
        # w_scale = F.interpolate(hm, size=(224,224), mode='bilinear')

        pdb.set_trace()

        for i in range(batch_imgs.shape[0]):
            save_image(batch_imgs[i],'./temp_1.png',normalize=True, nrow=1, pad_value=0, padding=0)

            ref_1 = cv2.imread('./temp_1.png')
            # pdb.set_trace()
            ref_1 = cv2.resize(ref_1, (ori_size[i,0],ori_size[i,1]))
            c_hm = F.interpolate(hm[i].unsqueeze(0), size=(ori_size[i,1],ori_size[i,0]), mode='bilinear')
            # c_hm = w_scale[i].unsqueeze(0)
            # pdb.set_trace()
            c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.2

            heatmap = np.uint8(255 * c_hm[0][0].cpu().detach().numpy())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img_1 = heatmap * 0.7 + ref_1 *0.5

            # c_hm = w_scale[i].unsqueeze(0)
            # c_hm = F.interpolate(c_hm, size=(ori_size[i,0],ori_size[i,1]), mode='bilinear')
            # # pdb.set_trace()
            # c_hm = (c_hm - torch.min(c_hm))/(torch.max(c_hm)-torch.min(c_hm))
            # c_hm = c_hm >= 0.2

            c_hm = c_hm[0,0,:,:]
            c_hm = c_hm.cpu().numpy()
            yy, xx = np.where(c_hm==True)

            estimated_bbox = np.zeros([1,4])
            c_gt_bbox =  np.zeros([1,4])
            estimated_bbox[0,1] = np.min(yy)
            estimated_bbox[0,3] = np.max(yy)
            estimated_bbox[0,0] = np.min(xx)
            estimated_bbox[0,2] = np.max(xx)


            cv2.imwrite('./{}_{}_T.png'.format(index, i), superimposed_img_1)

            # c_gt_bbox[0,0] = gt_bbox[i,:][0]-1
            # c_gt_bbox[0,1] = gt_bbox[i,:][1]-1
            # c_gt_bbox[0,2] = gt_bbox[i,:][0]-1 + gt_bbox[i,:][2]
            # c_gt_bbox[0,3] = gt_bbox[i,:][1]-1 + gt_bbox[i,:][3]

            # iou = bbox_iou(c_gt_bbox,estimated_bbox)[0]
            # # pdb.set_trace()

            # if gt_label[0][i]-1 == predict_cls[i]:

            #     cv2.imwrite('./{}_{}_{}_T.png'.format(index, i, iou), superimposed_img_1)
            # else:
            #     cv2.imwrite('./{}_{}_{}_F.png'.format(index, i, iou), superimposed_img_1)
            # # counter += 1
        # pdb.set_trace()

 







