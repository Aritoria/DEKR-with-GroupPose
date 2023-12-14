# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (zigang@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import cfg
from .config import update_config
from .conv_module import HighResolutionModule
from .conv_block import BasicBlock, Bottleneck, AdaptBlock

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
    'ADAPTIVE': AdaptBlock
}


class PoseHigherResolutionNet(nn.Module):
    def __init__(self, **kwargs):
        super(PoseHigherResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # build stage
        self.spec = cfg.MODEL.SPEC
        self.cfg = cfg
        update_config(cfg)
        self.stages_spec = self.spec.STAGES
        self.num_stages = self.spec.STAGES.NUM_STAGES
        num_channels_last = [256]
        for i in range(self.num_stages):
            num_channels = self.stages_spec.NUM_CHANNELS[i]
            transition_layer = \
                self._make_transition_layer(num_channels_last, num_channels)
            setattr(self, 'transition{}'.format(i + 1), transition_layer)

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, True
            )
            setattr(self, 'stage{}'.format(i + 2), stage)

        # build head net
        inp_channels = int(sum(self.stages_spec.NUM_CHANNELS[-1]))
        config_heatmap = self.spec.HEAD_HEATMAP
        config_offset = self.spec.HEAD_OFFSET
        # self.num_joints = cfg.DATASET.NUM_JOINTS
        self.num_joints = 17
        self.num_offset = self.num_joints * 2
        self.num_joints_with_center = self.num_joints + 1
        self.offset_prekpt = config_offset['NUM_CHANNELS_PERKPT']

        offset_channels = self.num_joints * self.offset_prekpt
        self.transition_heatmap = self._make_transition_for_head(
            inp_channels, config_heatmap['NUM_CHANNELS'])
        self.transition_offset = self._make_transition_for_head(
            inp_channels, offset_channels)
        self.head_heatmap = self._make_heatmap_head(config_heatmap)
        self.offset_feature_layers, self.offset_final_layer = \
            self._make_separete_regression_head(config_offset)

        self.pretrained_layers = self.spec.PRETRAINED_LAYERS

    def _make_transition_for_head(self, inplanes, outplanes):
        transition_layer = [
            nn.Conv2d(inplanes, outplanes, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(True)
        ]
        return nn.Sequential(*transition_layer)

    def _make_heatmap_head(self, layer_config):
        heatmap_head_layers = []

        feature_conv = self._make_layer(
            blocks_dict[layer_config['BLOCK']],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_CHANNELS'],
            layer_config['NUM_BLOCKS'],
            dilation=layer_config['DILATION_RATE']
        )
        heatmap_head_layers.append(feature_conv)

        heatmap_conv = nn.Conv2d(
            in_channels=layer_config['NUM_CHANNELS'],
            out_channels=self.num_joints_with_center,
            kernel_size=self.spec.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
        )
        heatmap_head_layers.append(heatmap_conv)

        return nn.ModuleList(heatmap_head_layers)

    def _make_separete_regression_head(self, layer_config):
        offset_feature_layers = []
        offset_final_layer = []

        for _ in range(self.num_joints):
            feature_conv = self._make_layer(
                blocks_dict[layer_config['BLOCK']],
                layer_config['NUM_CHANNELS_PERKPT'],
                layer_config['NUM_CHANNELS_PERKPT'],
                layer_config['NUM_BLOCKS'],
                dilation=layer_config['DILATION_RATE']
            )
            offset_feature_layers.append(feature_conv)

            offset_conv = nn.Conv2d(
                in_channels=layer_config['NUM_CHANNELS_PERKPT'],
                out_channels=2,
                kernel_size=self.spec.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if self.spec.FINAL_CONV_KERNEL == 3 else 0
            )
            offset_final_layer.append(offset_conv)

        return nn.ModuleList(offset_feature_layers), nn.ModuleList(offset_final_layer)

    def _make_layer(
            self, block, inplanes, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes,
                            stride, downsample, dilation=dilation))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, stages_spec, stage_index, num_inchannels,
                    multi_scale_output=True):
        num_modules = stages_spec.NUM_MODULES[stage_index]
        num_branches = stages_spec.NUM_BRANCHES[stage_index]
        num_blocks = stages_spec.NUM_BLOCKS[stage_index]
        num_channels = stages_spec.NUM_CHANNELS[stage_index]
        block = blocks_dict[stages_spec['BLOCK'][stage_index]]
        fuse_method = stages_spec.FUSE_METHOD[stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward_op(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, 'transition{}'.format(i + 1))
            for j in range(self.stages_spec['NUM_BRANCHES'][i]):
                if transition[j]:
                    x_list.append(transition[j](y_list[-1]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, 'stage{}'.format(i + 2))(x_list)

        x0_h, x0_w = y_list[0].size(2), y_list[0].size(3)
        x = torch.cat([y_list[0], \
                       F.upsample(y_list[1], size=(x0_h, x0_w), mode='bilinear'), \
                       F.upsample(y_list[2], size=(x0_h, x0_w), mode='bilinear'), \
                       F.upsample(y_list[3], size=(x0_h, x0_w), mode='bilinear')], 1)

        heatmap = self.head_heatmap[1](
            self.head_heatmap[0](self.transition_heatmap(x)))

        final_offset = []
        offset_feature = self.transition_offset(x)

        for j in range(self.num_joints):
            final_offset.append(
                self.offset_final_layer[j](
                    self.offset_feature_layers[j](
                        offset_feature[:, j * self.offset_prekpt:(j + 1) * self.offset_prekpt])))

        offset = torch.cat(final_offset, dim=1)
        return heatmap, offset

    def forward(self, image):
        with torch.no_grad():
            image = F.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)
            # print(image.shape)
            heatmap, offset = self.forward_op(image)
            # print(heatmap.shape)

            posemap = self.offset_to_pose(offset, flip=False)
            # print(posemap.shape)
            flip_index_heat = FLIP_CONFIG['COCO_WITH_CENTER']
            flip_index_offset = FLIP_CONFIG['COCO']
            image = torch.flip(image, [3])
            image[:, :, :, :-3] = image[:, :, :, 3:]
            heatmap_flip, offset_flip = self.forward_op(image)
            # print(heatmap_flip.shape)
            heatmap_flip = torch.flip(heatmap_flip, [3])
            heatmap = (heatmap + heatmap_flip[:, flip_index_heat, :, :]) / 2.0
            posemap_flip = self.offset_to_pose(offset_flip, flip_index=flip_index_offset)
            posemap = (posemap + torch.flip(posemap_flip, [3])) / 2.0
            # print(posemap_flip.shape)
            poses_all = []
            for i in range(len(heatmap)):
                heatmap_sum = 0
                poses = []
                heatmap_sum, poses = self.aggregate_results(
                    heatmap_sum, poses, heatmap[i].unsqueeze(0), [posemap[i]], 1
                )
                heatmap_avg = heatmap_sum / 1
                poses, scores = self.pose_nms(heatmap_avg, poses)
                poses = torch.Tensor(poses)
                poses_all.append(poses)

            poses_all = poses_all

            # scores = rescore_valid(self.cfg, poses, scores)
            # all_reg_preds.append(poses)
            # all_reg_scores.append(scores)
            # print(poses)
            # print(scores)
            # 输出的pose是以512为尺度的，没有使用归一化坐标
        return heatmap, offset, poses_all, torch.Tensor(scores)

        # return heatmap, offset


    def init_weights(self, pretrained='pose_dekr_hrnetw32_coco.pth', verbose=True):
        logger.info('=> init weights from normal distribution pose_dekr_hrnetw32_coco.pth')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.transform_matrix_conv.bias, 0)
            if hasattr(m, 'translation_conv'):
                nn.init.constant_(m.translation_conv.weight, 0)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.translation_conv.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained,
                                               map_location=lambda storage, loc: storage)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        print("Have Load The DEKR MODEL")

    def aggregate_results(
            self, heatmap_sum, poses, heatmap, posemap, scale
    ):
        """
        Get initial pose proposals and aggregate the results of all scale.

        Args:
            heatmap (Tensor): Heatmap at this scale (1, 1+num_joints, w, h)
            posemap (Tensor): Posemap at this scale (1, 2*num_joints, w, h)
            heatmap_sum (Tensor): Sum of the heatmaps (1, 1+num_joints, w, h)
            poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
        """
        dataset_input = 512
        dataset_output = 128
        ratio = dataset_input * 1.0 / dataset_output
        reverse_scale = ratio / scale
        h, w = heatmap[0].size(-1), heatmap[0].size(-2)


        heatmap_sum += self.up_interpolate(
            heatmap,
            size=(int(reverse_scale * w), int(reverse_scale * h)),
            mode='bilinear'
        )

        center_heatmap = heatmap[0, -1:]
        pose_ind, ctr_score = self.get_maximum_from_heatmap(center_heatmap)
        posemap = posemap[0].permute(1, 2, 0).view(h * w, -1, 2)
        pose = reverse_scale * posemap[pose_ind]
        ctr_score = ctr_score[:, None].expand(-1, pose.shape[-2])[:, :, None]
        poses.append(torch.cat([pose, ctr_score], dim=2))

        return heatmap_sum, poses

    def get_multi_stage_outputs(self, heatmap, offset):
        # forward
        posemap = self.offset_to_pose(offset, flip=False)
        return heatmap, posemap

    def offset_to_pose(self, offset, flip=True, flip_index=None):
        bs , num_offset, h, w = offset.shape[0:]
        num_joints = int(num_offset / 2)

        reg_poses_all = []

        for i in range(bs):
            reg_poses = self.get_reg_poses(offset[i], num_joints)

            if flip:
                reg_poses = reg_poses[:, flip_index, :]
                reg_poses[:, :, 0] = w - reg_poses[:, :, 0] - 1

            reg_poses = reg_poses.contiguous().view(h * w, 2 * num_joints).permute(1, 0)
            reg_poses = reg_poses.contiguous().view(1, -1, h, w).contiguous()
            reg_poses_all.append(reg_poses)
        reg_poses_all = torch.cat(reg_poses_all, dim=0)
        return reg_poses_all

    def get_reg_poses(self, offset, num_joints):
        _, h, w = offset.shape
        offset = offset.permute(1, 2, 0).reshape(h * w, num_joints, 2)
        locations = self.get_locations(h, w, offset.device)
        locations = locations[:, None, :].expand(-1, num_joints, -1)
        poses = locations - offset

        return poses

    def get_locations(self, output_h, output_w, device):
        shifts_x = torch.arange(
            0, output_w, step=1,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, output_h, step=1,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)

        return locations

    def up_interpolate(self, x, size, mode='bilinear'):
        H = x.size()[2]
        W = x.size()[3]
        scale_h = int(size[0] / H)
        scale_w = int(size[1] / W)
        inter_x = torch.nn.functional.interpolate(x, size=[size[0] - scale_h + 1, size[1] - scale_w + 1],
                                                  align_corners=True, mode='bilinear')
        padd = torch.nn.ReplicationPad2d((0, scale_w - 1, 0, scale_h - 1))
        return padd(inter_x)

    def get_maximum_from_heatmap(self, heatmap):
        maxm = self.hierarchical_pool(heatmap)
        maxm = torch.eq(maxm, heatmap).float()
        heatmap = heatmap * maxm
        scores = heatmap.view(-1)
        dataset_max_num_people = 30
        scores, pos_ind = scores.topk(dataset_max_num_people)
        test_KEYPOINT_THRESHOLD = 0.01
        select_ind = (scores > (test_KEYPOINT_THRESHOLD)).nonzero()
        scores = scores[select_ind][:, 0]
        pos_ind = pos_ind[select_ind][:, 0]

        return pos_ind, scores

    def hierarchical_pool(self, heatmap):
        pool1 = torch.nn.MaxPool2d(3, 1, 1)
        pool2 = torch.nn.MaxPool2d(5, 1, 2)
        pool3 = torch.nn.MaxPool2d(7, 1, 3)
        map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
        test_pool_threshold1 = 300
        test_pool_threshold2 = 200
        if map_size > test_pool_threshold1:
            maxm = pool3(heatmap[None, :, :, :])
        elif map_size > test_pool_threshold2:
            maxm = pool2(heatmap[None, :, :, :])
        else:
            maxm = pool1(heatmap[None, :, :, :])

        return maxm

    def pose_nms(self, heatmap_avg, poses):
        """
        NMS for the regressed poses results.

        Args:
            heatmap_avg (Tensor): Avg of the heatmaps at all scales (1, 1+num_joints, w, h)
            poses (List): Gather of the pose proposals [(num_people, num_joints, 3)]
        """
        scale1_index = sorted([1], reverse=True).index(1.0)
        pose_norm = poses[scale1_index]
        max_score = pose_norm[:, :, 2].max() if pose_norm.shape[0] else 1

        for i, pose in enumerate(poses):
            if i != scale1_index:
                max_score_scale = pose[:, :, 2].max() if pose.shape[0] else 1
                pose[:, :, 2] = pose[:, :, 2] / max_score_scale * max_score * 1.0

        pose_score = torch.cat([pose[:, :, 2:] for pose in poses], dim=0)
        pose_coord = torch.cat([pose[:, :, :2] for pose in poses], dim=0)

        if pose_coord.shape[0] == 0:
            return [], []

        num_people, num_joints, _ = pose_coord.shape
        heatval = self.get_heat_value(pose_coord, heatmap_avg[0])
        heat_score = (torch.sum(heatval, dim=1) / num_joints)[:, 0]

        pose_score = pose_score * heatval
        poses = torch.cat([pose_coord.cpu(), pose_score.cpu()], dim=2)

        keep_pose_inds = self.nms_core(pose_coord, heat_score)
        poses = poses[keep_pose_inds]
        heat_score = heat_score[keep_pose_inds]
        DATASET_MAX_NUM_PEOPLE = 30
        if len(keep_pose_inds) > DATASET_MAX_NUM_PEOPLE:
            heat_score, topk_inds = torch.topk(heat_score, DATASET_MAX_NUM_PEOPLE)
            poses = poses[topk_inds]

        poses = [poses.numpy()]
        scores = [i[:, 2].mean() for i in poses[0]]

        return poses, scores

    def cal_area_2_torch(self, v):
        w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
        h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
        return w * w + h * h

    def nms_core(self, pose_coord, heat_score):
        num_people, num_joints, _ = pose_coord.shape
        pose_area = self.cal_area_2_torch(pose_coord)[:, None].repeat(1, num_people * num_joints)
        pose_area = pose_area.reshape(num_people, num_people, num_joints)

        pose_diff = pose_coord[:, None, :, :] - pose_coord
        pose_diff.pow_(2)
        pose_dist = pose_diff.sum(3)
        pose_dist.sqrt_()
        TEST_NMS_THRE = 0.15
        TEST_NMS_NUM_THRE = 10
        pose_thre = TEST_NMS_THRE * torch.sqrt(pose_area)
        pose_dist = (pose_dist < pose_thre).sum(2)
        nms_pose = pose_dist > TEST_NMS_NUM_THRE

        ignored_pose_inds = []
        keep_pose_inds = []
        for i in range(nms_pose.shape[0]):
            if i in ignored_pose_inds:
                continue
            keep_inds = nms_pose[i].nonzero().cpu().numpy()
            keep_inds = [list(kind)[0] for kind in keep_inds]
            keep_scores = heat_score[keep_inds]
            ind = torch.argmax(keep_scores)
            keep_ind = keep_inds[ind]
            if keep_ind in ignored_pose_inds:
                continue
            keep_pose_inds += [keep_ind]
            ignored_pose_inds += list(set(keep_inds) - set(ignored_pose_inds))

        return keep_pose_inds

    def get_heat_value(self, pose_coord, heatmap):
        _, h, w = heatmap.shape
        heatmap_nocenter = heatmap[:-1].flatten(1, 2).transpose(0, 1)

        y_b = torch.clamp(torch.floor(pose_coord[:, :, 1]), 0, h - 1).long()
        x_l = torch.clamp(torch.floor(pose_coord[:, :, 0]), 0, w - 1).long()
        heatval = torch.gather(heatmap_nocenter, 0, y_b * w + x_l).unsqueeze(-1)
        return heatval

    # def get_final_preds(self,grouped_joints, center, scale, heatmap_size):
    #     final_results = []
    #     for person in grouped_joints[0]:
    #         joints = np.zeros((person.shape[0], 3))
    #         joints = self.transform_preds(person, center, scale, heatmap_size)
    #         final_results.append(joints)
    #
    #     return final_results
    #
    # def transform_preds(self,coords, center, scale, output_size):
    #     # target_coords = np.zeros(coords.shape)
    #     target_coords = coords.copy()
    #     trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    #     for p in range(coords.shape[0]):
    #         target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    #     return target_coords


FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
    'CROWDPOSE': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
    ],
    'CROWDPOSE_WITH_CENTER': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
    ]
}


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model
