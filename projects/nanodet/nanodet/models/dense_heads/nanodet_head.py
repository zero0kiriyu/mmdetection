import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, MultiConfig, OptConfigType)
from mmdet.models.dense_heads.gfl_head import GFLHead
from mmdet.models.utils import multi_apply


@MODELS.register_module()
class NanoDetHead(GFLHead):
    """
    Modified from GFL, use same loss functions but much lightweight convolution heads
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        stacked_convs: int = 2,
        conv_cfg: OptConfigType = None,
        use_depthwise: bool = True,
        norm_cfg: ConfigType = dict(
            type='BN'),
        loss_dfl: ConfigType = dict(
            type='DistributionFocalLoss', loss_weight=0.25),
        bbox_coder: ConfigType = dict(type='DistancePointBBoxCoder'),
        reg_max: int = 16,
        share_cls_reg=False,
        act_cfg=dict(type="LeakyReLU"),
        init_cfg: MultiConfig = dict(
            type='Normal',
            layer='Conv2d',
            std=0.01,
            override=dict(
                type='Normal',
                name='gfl_cls',
                std=0.01,
                bias_prob=0.01)),
        **kwargs
    ) -> None:
        
        self.share_cls_reg = share_cls_reg
        self.act_cfg = act_cfg
        self.ConvModule = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        super(NanoDetHead, self).__init__(
            num_classes,
            in_channels,
            stacked_convs,
            conv_cfg,
            norm_cfg,
            loss_dfl,
            bbox_coder,
            reg_max,
            init_cfg,
            **kwargs
        )

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in self.anchor_generator.strides:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.cls_out_channels + 4 * (self.reg_max + 1)
                    if self.share_cls_reg
                    else self.cls_out_channels,
                    1,
                    padding=0,
                )
                for _ in self.anchor_generator.strides
            ]
        )
        # TODO: if
        self.gfl_reg = nn.ModuleList(
            [
                nn.Conv2d(self.feat_channels, 4 * (self.reg_max + 1), 1, padding=0)
                for _ in self.anchor_generator.strides
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    act_cfg=self.act_cfg,
                )
            )
            if not self.share_cls_reg:
                reg_convs.append(
                    self.ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None,
                        act_cfg=self.act_cfg,
                    )
                )

        return cls_convs, reg_convs

    def forward(self, feats):
        return multi_apply(self.forward_single,
                           feats,
                           self.cls_convs,
                           self.reg_convs,
                           self.gfl_cls,
                           self.gfl_reg,
                           )

    def forward_single(self,feat,cls_conv,reg_conv,gfl_cls,gfl_reg):
        cls_feat = feat
        reg_feat = feat
        for cls_conv in cls_conv:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_conv:
            reg_feat = reg_conv(reg_feat)
        if self.share_cls_reg:
            feat = gfl_cls(cls_feat)
            cls_score, bbox_pred = torch.split(feat, [self.cls_out_channels, 4 * (self.reg_max + 1)], dim=1)
        else:
            cls_score = gfl_cls(cls_feat)
            bbox_pred = gfl_reg(reg_feat)
            
        if torch.onnx.is_in_onnx_export():
            cls_score = torch.sigmoid(cls_score).reshape(1, self.num_classes, -1).permute(0, 2, 1)
            bbox_pred = bbox_pred.reshape(1, (self.reg_max + 1) * 4, -1).permute(0, 2, 1)
        return cls_score, bbox_pred