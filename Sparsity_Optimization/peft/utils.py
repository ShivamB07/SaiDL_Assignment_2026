import torch
from peft.SoRA import SoRALinear
from peft.LoRA import LoRALinear
from peft.AdaLoRA import AdaLoRALinear
 
 
# FIX 4: Added missing `rank` and `alpha` parameters.
# Original signature: inject_peft_layers(model, peft_type="lora")
# Called in train.py as: inject_peft_layers(model, cfg.peft.type, rank=..., alpha=...)
# → TypeError: inject_peft_layers() got unexpected keyword arguments 'rank', 'alpha'
def inject_peft_layers(model, peft_type="lora", rank=8, alpha=16):
    for name, module in model.named_modules():
        # FIX 5: `module.requires_grad = False` just sets an attribute — it does NOT
        # freeze parameters. The correct method is requires_grad_(False), which
        # recursively sets requires_grad=False on all parameters of the module.
        module.requires_grad_(False)
 
        # Target Attention Query and Value projection layers in DeBERTa
        if "attention" in name and ("query_proj" in name or "value_proj" in name):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = model.get_submodule(parent_name)
 
            if peft_type == "lora":
                new_layer = LoRALinear(module, rank=rank, alpha=alpha)
            elif peft_type == "adalora":
                new_layer = AdaLoRALinear(module, init_rank=rank, alpha=alpha)
            elif peft_type == "sora":
                new_layer = SoRALinear(module, init_rank=rank, alpha=alpha)
            else:
                raise ValueError(f"Unsupported PEFT type: {peft_type}")
 
            setattr(parent, child_name, new_layer)
    return model
 
 
def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} || All params: {all_params} || Trainable %: {100 * trainable_params / all_params:.4f}")
 
 
def compute_effective_rank(model, threshold=1e-5):
    effective_ranks = []
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, AdaLoRALinear, SoRALinear)):
            with torch.no_grad():
                if isinstance(module, LoRALinear):
                    delta_W = module.lora_A @ module.lora_B
                elif isinstance(module, AdaLoRALinear):
                    delta_W = (module.lora_P * module.lora_E) @ module.lora_Q
                elif isinstance(module, SoRALinear):
                    delta_W = (module.lora_A * module.gate) @ module.lora_B
 
                # Use torch.linalg.svd (torch.svd is deprecated in PyTorch >= 1.9)
                _, S, _ = torch.linalg.svd(delta_W, full_matrices=False)
                rank = (S > threshold).sum().item()
                effective_ranks.append(rank)
 
    avg_rank = sum(effective_ranks) / len(effective_ranks) if effective_ranks else 0
    return avg_rank