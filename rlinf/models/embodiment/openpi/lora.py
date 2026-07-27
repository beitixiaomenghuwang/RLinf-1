"""OpenPI-specific integration for action-expert LoRA adapters."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from torch import nn

DEFAULT_ACTION_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class ActionLoRAInjectionReport:
    """Summarize action-expert LoRA injection and trainable parameters."""

    injected_module_names: tuple[str, ...]
    adapter_parameters: int
    trainable_parameters: int


def is_action_lora_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether plain LoRA should target the OpenPI action expert."""
    return (
        bool(config.get("is_lora", False))
        and str(config.get("lora_target", "vlm")) == "action_expert"
    )


def state_dict_contains_action_lora(state_dict: Mapping[str, Any]) -> bool:
    """Identify a full checkpoint containing action-expert LoRA parameters."""
    action_prefix = ".paligemma_with_expert.gemma_expert.model."
    markers = (".lora_A.", ".lora_B.")
    return any(
        action_prefix in f".{name}" and any(marker in f".{name}." for marker in markers)
        for name in state_dict
    )


def _get_action_expert_transformer(model: nn.Module) -> nn.Module:
    try:
        return model.paligemma_with_expert.gemma_expert.model
    except AttributeError as error:
        raise ValueError(
            "OpenPI action LoRA requires model.paligemma_with_expert.gemma_expert.model"
        ) from error


def _tag_lora_subtree(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        setattr(module, "_to_lora", enabled)


def configure_openpi_action_lora(
    model: nn.Module, config: Mapping[str, Any]
) -> ActionLoRAInjectionReport:
    """Inject plain PEFT LoRA into the Pi0.5 action transformer only."""
    if not is_action_lora_enabled(config):
        raise ValueError(
            "configure_openpi_action_lora requires is_lora=true and "
            "lora_target=action_expert"
        )
    if bool(config.get("lora_require_pi05", True)) and not bool(
        getattr(model, "pi05", False)
    ):
        raise ValueError("OpenPI action LoRA requires a pi0.5 model")

    from peft import LoraConfig, inject_adapter_in_model

    rank = int(config.get("lora_rank", 64))
    if rank <= 0:
        raise ValueError("lora_rank must be positive")
    target_modules = tuple(
        config.get("lora_target_modules", DEFAULT_ACTION_LORA_TARGETS)
    )
    if not target_modules:
        raise ValueError("lora_target_modules must not be empty")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=float(config.get("lora_alpha", rank)),
        lora_dropout=float(config.get("lora_dropout", 0.0)),
        target_modules=list(target_modules),
        init_lora_weights=config.get("lora_init", "gaussian"),
        bias="none",
    )

    model.requires_grad_(False)
    action_expert = _get_action_expert_transformer(model)
    action_expert = inject_adapter_in_model(lora_config, action_expert)
    model.paligemma_with_expert.gemma_expert.model = action_expert

    _tag_lora_subtree(model, False)
    _tag_lora_subtree(action_expert, True)
    if bool(config.get("lora_train_value_head", True)) and hasattr(model, "value_head"):
        model.value_head.requires_grad_(True)

    injected_module_names = tuple(
        name
        for name, module in action_expert.named_modules()
        if hasattr(module, "lora_A")
    )
    adapter_parameters = sum(
        parameter.numel()
        for name, parameter in action_expert.named_parameters()
        if "lora_" in name
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    report = ActionLoRAInjectionReport(
        injected_module_names=injected_module_names,
        adapter_parameters=adapter_parameters,
        trainable_parameters=trainable_parameters,
    )
    model.action_lora_injection_report = report
    return report
