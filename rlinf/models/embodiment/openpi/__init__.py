# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# openpi model configs

import os
import pathlib

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.gse import (
        configure_openpi_gse,
        is_gse_enabled,
        state_dict_contains_gse,
        state_dict_contains_vlm_gse,
    )
    from rlinf.models.embodiment.openpi.lora import (
        configure_openpi_action_lora,
        is_action_lora_enabled,
        state_dict_contains_action_lora,
    )
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    gse_config = cfg.get("gse", None)
    gse_enabled = is_gse_enabled(gse_config)
    action_lora_enabled = is_action_lora_enabled(cfg)
    if gse_enabled and cfg.get("is_lora", False):
        raise ValueError("OpenPI GSE and LoRA cannot be enabled at the same time")

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Read weights first so a resumed GSE checkpoint can be identified before
    # the model structure is changed. Original SFT checkpoints are loaded before
    # injection, while GSE checkpoints require the wrapped structure first.
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        model_state_dict = {}
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            model_state_dict.update(state_dict)

    checkpoint_has_gse = state_dict_contains_gse(model_state_dict)
    checkpoint_has_vlm_gse = state_dict_contains_vlm_gse(model_state_dict)
    checkpoint_has_action_lora = state_dict_contains_action_lora(model_state_dict)
    if checkpoint_has_gse and checkpoint_has_action_lora:
        raise ValueError("The checkpoint cannot contain both GSE and action LoRA")
    if checkpoint_has_action_lora and not action_lora_enabled:
        raise ValueError(
            "The checkpoint contains action LoRA parameters, but "
            "is_lora=true and lora_target=action_expert are not configured"
        )
    vlm_gse_enabled = is_gse_enabled(
        gse_config.get("vlm", None) if gse_config is not None else None
    )
    upgrade_action_checkpoint_with_vlm = (
        checkpoint_has_gse and vlm_gse_enabled and not checkpoint_has_vlm_gse
    )
    if checkpoint_has_gse and not gse_enabled:
        raise ValueError(
            "The checkpoint contains GSE parameters, but model.gse.enabled is false"
        )
    if checkpoint_has_vlm_gse and not vlm_gse_enabled:
        raise ValueError(
            "The checkpoint contains VLM GSE parameters, but model.gse.vlm.enabled "
            "is false"
        )
    if checkpoint_has_gse:
        if upgrade_action_checkpoint_with_vlm:
            action_only_config = dict(gse_config)
            action_only_config["vlm"] = {"enabled": False}
            action_only_config["train_action_adapters"] = True
            configure_openpi_gse(model, action_only_config)
        else:
            configure_openpi_gse(model, gse_config)
    elif checkpoint_has_action_lora:
        configure_openpi_action_lora(model, cfg)

    model.load_state_dict(model_state_dict, strict=False)
    del model_state_dict

    if upgrade_action_checkpoint_with_vlm:
        configure_openpi_gse(
            model,
            gse_config,
            action_already_injected=True,
        )

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    if gse_enabled and not checkpoint_has_gse:
        configure_openpi_gse(model, gse_config)
    if action_lora_enabled and not checkpoint_has_action_lora:
        configure_openpi_action_lora(model, cfg)
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats_path = (
        data_kwargs.get("norm_stats_path") if data_kwargs is not None else None
    )
    if norm_stats_path is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            norm_dir = pathlib.Path(norm_stats_path).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = _checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)
    else:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
