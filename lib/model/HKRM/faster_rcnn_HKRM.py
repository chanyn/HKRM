import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn_region import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade_region import _ProposalTargetLayer
#import time
#import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta



## Use region_feature to compute A
class A_region_compute(nn.Module):
    def __init__(self, input_features):
        super(A_region_compute, self).__init__()
        self.conv2d_1 = nn.Conv2d(input_features, input_features, 1, stride=1)
        self.conv2d_2 = nn.Conv2d(input_features, 1, 1, stride=1)
    def forward(self, cat_feature):
        W1 = cat_feature.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2)
#        W_new = torch.abs(W1 - W2)
        W_new = W1 - W2
        W_new = torch.transpose(W_new, 1, 3)
        W_new = self.conv2d_1(W_new)
        W_new = F.relu(W_new)
        W_new = self.conv2d_2(W_new)
        # Use softmax
        W_new = W_new.contiguous()
        W_new=W_new.squeeze(1)
        W_new = F.softmax(W_new,2)
        Adj_M=W_new      
        return Adj_M

class Know_Rout_mod_im(nn.Module):
    def __init__(self, num_A, input_features, output_features):
        super(Know_Rout_mod_im, self).__init__()
        self.num_A = num_A
        for i in range(self.num_A):
            module_A = A_region_compute(5)
            self.add_module('compute_A{}'.format(i), module_A)
        self.region_squeeze = nn.Linear(input_features, output_features)

    def forward(self, reigon_feature, x):
        no_node = reigon_feature.shape[1]
        bs = reigon_feature.shape[0]
        A = Variable(torch.zeros(bs,no_node,no_node)).cuda()
        Iden = Variable(torch.eye(no_node).unsqueeze(0).repeat(bs, 1, 1), requires_grad=False).cuda()
        for i in range(self.num_A):
            A = self._modules['compute_A{}'.format(i)](reigon_feature) + A + Iden
        # Row sum to one
        A = A / A.sum(1).unsqueeze(1)
        x = torch.bmm(A, x)
        x = self.region_squeeze(x)
        return x

class A_compute(nn.Module):
    def __init__(self, input_features, nf=64, ratio=[4, 2, 1]):
        super(A_compute, self).__init__()
        self.num_features = nf
        self.conv2d_1 = nn.Conv2d(input_features, int(nf * ratio[0]), 1, stride=1)
        #        self.bn_1 = nn.BatchNorm2d(int(nf * ratio[0]))
        self.conv2d_2 = nn.Conv2d(int(nf * ratio[0]), int(nf * ratio[1]), 1, stride=1)
        self.conv2d_3 = nn.Conv2d(int(nf * ratio[1]), int(nf * ratio[2]), 1, stride=1)
        self.conv2d_4 = nn.Conv2d(int(nf * ratio[2]), 1, 1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, cat_feature):
        W1 = cat_feature.unsqueeze(2)
        W2 = torch.transpose(W1, 1, 2)
        W_new = torch.abs(W1 - W2)
        W_new = torch.transpose(W_new, 1, 3)
        W_new = self.conv2d_1(W_new)
        W_new = self.relu(W_new)
        W_new = self.conv2d_2(W_new)
        W_new = self.relu(W_new)
        W_new = self.conv2d_3(W_new)
        W_new = self.relu(W_new)
        W_new = self.conv2d_4(W_new)
        W_new = W_new.contiguous()
        W_new = W_new.squeeze(1)
        Adj_M = W_new
        return Adj_M


class Know_Rout_mod(nn.Module):
    def __init__(self, input_features, output_features):
        super(Know_Rout_mod, self).__init__()
        self.input_features = input_features
        self.lay_1_compute_A = A_compute(input_features)
        self.transferW = nn.Linear(input_features, output_features)

    def forward(self, cat_feature):
        cat_feature_stop = Variable(cat_feature.data)
        Adj_M1 = self.lay_1_compute_A(cat_feature_stop)
        # batch matrix-matrix product
        W_M1 = F.softmax(Adj_M1, 2)

        # batch matrix-matrix product
        cat_feature = torch.bmm(W_M1, cat_feature)
        cat_feature = self.transferW(cat_feature)

        return cat_feature, Adj_M1


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic, cls_a_prob, cls_r_prob, modules_size):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        # modules size and exist
        self.modules_size = modules_size
        self.module_exist = [x != 0 for x in modules_size]

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        # define three modules
        if self.module_exist[0]:
            self.one_Know_Rout_mod_a = Know_Rout_mod(2048, modules_size[0])
            self.gt_adj_a = nn.Parameter(torch.from_numpy(cls_a_prob), requires_grad=False)
        if self.module_exist[1]:
            self.one_Know_Rout_mod_r = Know_Rout_mod(2048, modules_size[1])
            self.gt_adj_r = nn.Parameter(torch.from_numpy(cls_r_prob), requires_grad=False)
        if self.module_exist[2]:
            self.one_Know_Rout_mod_s = Know_Rout_mod_im(10, 2048, modules_size[2])


    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, output_cls_score, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes, output_cls_score)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws, output_bg_score = roi_data

            # record rois_label
            index_ = rois_label.long()
            if self.module_exist[0]:
                gt_adj_a = Variable(rois_target.new(batch_size, index_.size(1), index_.size(1)).zero_()).detach()
            if self.module_exist[1]:
                gt_adj_r = Variable(rois_target.new(batch_size, index_.size(1), index_.size(1)).zero_()).detach()

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            
            if self.module_exist[0]:
                for b in range(batch_size):
                    temp = self.gt_adj_a[index_[b], :]
                    temp = temp.transpose(0,1)[index_[b], :]
                    gt_adj_a[b] = temp.transpose(0,1)
            if self.module_exist[1]:
                for b in range(batch_size):
                    temp = self.gt_adj_r[index_[b], :]
                    temp = temp.transpose(0,1)[index_[b], :]
                    gt_adj_r[b] = temp.transpose(0,1)


        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
            output_bg_score = output_cls_score

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # pooled_feat [bs*128,1024,7,7]

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        # pooled_feat [bs*128,2048]

        # Region feature
        if self.module_exist[2]:
            norm_rois = rois.data[:, :, 1:] / im_info[:, [1, 0]].repeat(1, 2).unsqueeze(1)
            region_feature = torch.cat([norm_rois, output_bg_score[:, :, 1].contiguous().unsqueeze(2)], 2)
            region_feature = Variable(region_feature)

        # visual attribute
        visual_feat = pooled_feat.view(batch_size, -1, 2048)
        # three modules
        pooled_feat = visual_feat.clone()
        if self.module_exist[0]:
            transfer_feat_a, adja = self.one_Know_Rout_mod_a(visual_feat)
            pooled_feat = torch.cat((pooled_feat, transfer_feat_a), -1)
        if self.module_exist[1]:
            transfer_feat_r, adjr = self.one_Know_Rout_mod_r(visual_feat)
            pooled_feat = torch.cat((pooled_feat, transfer_feat_r), -1)
        if self.module_exist[2]:
            transfer_feat_s = self.one_Know_Rout_mod_s(region_feature, visual_feat)
            pooled_feat = torch.cat((pooled_feat, transfer_feat_s), -1)
        
        # Concate the modules' output
        pooled_feat = pooled_feat.contiguous().view(-1, 2048 + int(sum(self.modules_size)))

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred_hkrm(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.RCNN_cls_score_hkrm(pooled_feat)
        cls_prob = F.softmax(cls_score, dim=1)

        RCNN_loss_cls = 0.
        RCNN_loss_bbox = 0.
        adja_loss = Variable(torch.zeros(1), requires_grad = False).cuda()
        adjr_loss = Variable(torch.zeros(1), requires_grad = False).cuda()

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            if self.module_exist[0]:
                gt_adja_nograd = gt_adj_a.detach()
                adja_loss = F.mse_loss(adja, gt_adja_nograd)
            if self.module_exist[1]:
                gt_adjr_nograd = gt_adj_r.detach()
                adjr_loss = F.mse_loss(adjr, gt_adjr_nograd)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        if self.training:
            return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, adja_loss, adjr_loss

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score_hkrm, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred_hkrm, 0, 0.001, cfg.TRAIN.TRUNCATED)


    def create_architecture(self):
        self._init_modules()
        self._init_weights()
