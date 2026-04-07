"""WildDet3D on Stereo4D-Bench evaluation - box prompt (oracle), monocular depth."""

from __future__ import annotations

import os

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.io import FileBackend
from vis4d.zoo.base import get_default_cfg

from configs.base.callback import get_stereo4d_eval_callbacks
from configs.base.dataset.stereo4d import get_stereo4d_test_cfg
from configs.base.pl import get_pl_cfg

from configs.base.optim import get_wilddet3d_optim_cfg
from configs.base.model import (
    get_wilddet3d_cfg,
    get_wilddet3d_hyperparams_cfg,
)
from configs.base.loss import get_wilddet3d_loss_cfg
from configs.base.connector import (
    get_wilddet3d_data_connector_cfg,
    WildDet3DDetect3DEvalConnector,
    WildDet3DVisConnector,
)
from configs.base.data import get_wilddet3d_data_cfg


def get_config() -> ExperimentConfig:
    """Returns WildDet3D box prompt (oracle) eval for Stereo4D-Bench."""
    config = get_default_cfg(
        exp_name="wilddet3d_stereo4d_box_prompt"
    )

    config.use_checkpoint = True

    params = get_wilddet3d_hyperparams_cfg(
        num_epochs=12,
        samples_per_gpu=4,
        workers_per_gpu=4,
        base_lr=1e-4,
    )

    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_backend = class_config(FileBackend)
    sam3_image_shape = (1008, 1008)
    data_root = "data/in_the_wild"

    stereo4d_test_cfg = get_stereo4d_test_cfg(
        data_root=data_root,
        test_dataset="Stereo4D_val",
        data_backend=data_backend,
        with_depth=False,
        shape=sam3_image_shape,
    )

    config.data = get_wilddet3d_data_cfg(
        train_datasets=stereo4d_test_cfg,
        test_datasets=[stereo4d_test_cfg],
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        max_prompts_per_image=50,
        use_text_prompts=True,
        use_geometry_prompts=True,
        oracle_eval=True,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model, box_coder = get_wilddet3d_cfg(
        params=params,
        sam3_checkpoint="pretrained/sam3/sam3_detector.pt",
        geometry_backend_type="lingbot_depth",
        lingbot_encoder_freeze_blocks=21,
        backbone_freeze_blocks=28,
        canonical_rotation=True,
        oracle_eval=True,
    )

    config.loss = get_wilddet3d_loss_cfg(params, box_coder)

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = get_wilddet3d_optim_cfg(
        params,
        freeze_backbone=params.freeze_backbone,
        freeze_all_pretrained=params.freeze_all_pretrained,
    )

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector, config.test_data_connector = (
        get_wilddet3d_data_connector_cfg()
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    from vis4d.data.const import AxisMode
    from vis4d.engine.callbacks import VisualizerCallback
    from vis4d.vis.image.bbox3d_visualizer import BoundingBox3DVisualizer
    from vis4d.vis.image.canvas import PillowCanvasBackend
    from vis4d.zoo.base import get_default_callbacks_cfg

    callbacks = get_default_callbacks_cfg()

    callbacks.extend(
        get_stereo4d_eval_callbacks(
            data_root=data_root,
            output_dir=config.output_dir,
            test_connector=class_config(WildDet3DDetect3DEvalConnector),
        )
    )

    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(
                BoundingBox3DVisualizer,
                axis_mode=AxisMode.OPENCV,
                width=4,
                camera_near_clip=0.01,
                plot_heading=False,
                vis_freq=50,
                plot_trajectory=False,
                canvas=class_config(PillowCanvasBackend, font_size=16),
                save_boxes3d=True,
            ),
            output_dir=config.output_dir,
            save_prefix="box3d",
            test_connector=class_config(
                WildDet3DVisConnector, score_threshold=0.0
            ),
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
