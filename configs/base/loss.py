"""WildDet3D loss configuration."""

from __future__ import annotations

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.config.typing import ExperimentParameters
from vis4d.engine.loss_module import LossModule

from wilddet3d.loss.wilddet3d_loss import WildDet3DLoss, WildDet3DLossConfig
from wilddet3d.head import Det3DCoder
from configs.base.connector import WildDet3DLossConnector


def get_wilddet3d_loss_cfg(
    params: ExperimentParameters,
    box_coder: ConfigDict,
    # Global scale factors (for balancing 2D:3D:Geom ratio)
    loss_2d_scale: float = 1.0,
    loss_3d_scale: float = 1.0,
    loss_geom_scale: float = 5.0,
    # 2D Loss weights
    loss_cls_weight: float = 20.0,  # SAM3 original
    loss_bbox_weight: float = 5.0,
    loss_giou_weight: float = 2.0,
    # 3D Loss weights
    loss_delta_2d_weight: float = 1.0,  # same as GDino3D
    loss_depth_weight: float = 1.0,
    loss_dim_weight: float = 1.0,
    loss_rot_weight: float = 1.0,
    # Geometry loss weights
    loss_silog_weight: float = 1.0,
    loss_phi_weight: float = 0.1,
    loss_theta_weight: float = 0.1,
    loss_opt_ssi_weight: float = 0.5,  # SSI loss weight (UniDepthV2)
    # Auxiliary loss
    aux_loss_weight: float = 1.0,
    # 3D Confidence Head (positive + negative)
    use_3d_conf: bool = False,
    loss_3d_conf_weight: float = 20.0,
    conf_depth_weight: float = 0.7,
    conf_iou_3d_weight: float = 0.3,
    # Presence loss
    presence_loss_weight: float = 20.0,
    # O2M matching
    use_o2m: bool = True,
    # Ignore box negative loss suppression
    use_ignore_suppress: bool = False,
    ignore_iou_threshold: float = 0.5,
) -> ConfigDict:
    """Get WildDet3D loss configuration.

    Args:
        params: Experiment parameters.
        box_coder: Box coder configuration.
        loss_2d_scale: Global scale for 2D losses (cls, bbox, giou).
        loss_3d_scale: Global scale for 3D losses (delta, depth, dim, rot).
        loss_geom_scale: Global scale for geometry backend losses (silog, etc.).
        loss_cls_weight: Classification loss weight.
        loss_bbox_weight: L1 box loss weight.
        loss_giou_weight: GIoU loss weight.
        loss_delta_2d_weight: Delta 2D center loss weight.
        loss_depth_weight: Depth loss weight.
        loss_dim_weight: Dimension loss weight.
        loss_rot_weight: Rotation loss weight.
        loss_silog_weight: SILog depth loss weight.
        loss_phi_weight: Phi angle loss weight.
        loss_theta_weight: Theta angle loss weight.
        aux_loss_weight: Auxiliary output loss weight.

    Returns:
        Loss configuration.
    """
    loss_config = class_config(
        WildDet3DLossConfig,
        # Global scale factors
        loss_2d_scale=loss_2d_scale,
        loss_3d_scale=loss_3d_scale,
        loss_geom_scale=loss_geom_scale,
        # 2D Loss weights
        loss_cls_weight=loss_cls_weight,
        loss_bbox_weight=loss_bbox_weight,
        loss_giou_weight=loss_giou_weight,
        # 3D Loss weights
        loss_delta_2d_weight=loss_delta_2d_weight,
        loss_depth_weight=loss_depth_weight,
        loss_dim_weight=loss_dim_weight,
        loss_rot_weight=loss_rot_weight,
        # Geometry loss weights
        loss_silog_weight=loss_silog_weight,
        loss_phi_weight=loss_phi_weight,
        loss_theta_weight=loss_theta_weight,
        loss_opt_ssi_weight=loss_opt_ssi_weight,
        # Auxiliary loss
        aux_loss_weight=aux_loss_weight,
        # Presence loss
        presence_loss_weight=presence_loss_weight,
        # 3D Confidence Head
        use_3d_conf=use_3d_conf,
        loss_3d_conf_weight=loss_3d_conf_weight,
        conf_depth_weight=conf_depth_weight,
        conf_iou_3d_weight=conf_iou_3d_weight,
        # O2M matching
        use_o2m=use_o2m,
        # Ignore box negative loss suppression
        use_ignore_suppress=use_ignore_suppress,
        ignore_iou_threshold=ignore_iou_threshold,
    )
    
    loss = class_config(
        WildDet3DLoss,
        config=loss_config,
        box_coder=box_coder,
    )

    # Wrap in LossModule for proper wandb logging of individual loss components
    # (loss_cls, loss_bbox, loss_giou, loss_depth, etc.)
    # WildDet3DLossConnector passes structured objects (WildDet3DOut, batch) to loss
    loss_dict = {
        "loss": loss,
        "connector": class_config(WildDet3DLossConnector),
    }

    # Metrics (metric_* keys) are logged to wandb but excluded from
    # total_loss so they don't affect optimization or loss scale.
    return class_config(
        LossModule,
        losses=[loss_dict],
        exclude_attributes=[
            "WildDet3DLoss.metric_ce_f1",
            "WildDet3DLoss.metric_presence_acc",
            "WildDet3DLoss.metric_fusion_delta",
        ],
    )

