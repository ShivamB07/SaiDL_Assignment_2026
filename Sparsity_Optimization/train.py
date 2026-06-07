import time
import wandb  # FIX 1: wandb was used but never imported
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import matthews_corrcoef
from data.cola_dataset import get_cola_dataloader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
 
from peft.utils import inject_peft_layers, print_trainable_parameters, compute_effective_rank
from peft.AdaLoRA import AdaLoRALinear
from peft.SoRA import SoRALinear

wandb.login(key="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")

 
 
def evaluate(model, eval_loader, device):
    model.eval()
    losses = []
    all_preds, all_labels = [], []
 
    with torch.no_grad():
        for batch in eval_loader:
            # FIX 2: batch is a dict, not a (inputs, labels) tuple.
            # Original code: `inputs, labels = batch` → TypeError at runtime.
            # Also fixes: `outputs = model(inputs)` (missing attention_mask)
            # and `batch["labels"]` after batch was already unpacked to local vars.
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = torch.nn.functional.cross_entropy(outputs.logits, batch["labels"])
            losses.append(loss.item())
 
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
 
    avg_loss = sum(losses) / len(losses)
    mcc_score = matthews_corrcoef(all_labels, all_preds)
    model.train()
    return avg_loss, mcc_score
 
 
@hydra.main(version_base=None, config_path="config", config_name="base_config")
def main(cfg: DictConfig) -> None:
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))
 
    # FIX 1 (cont.): Initialize wandb before any wandb.log() calls
    wandb.init(
        project="sparsity-optimization",
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
 
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        num_labels=cfg.model.num_labels,
    )
    model = inject_peft_layers(model, cfg.peft.type, rank=cfg.peft.rank, alpha=cfg.peft.alpha).to(device)
    print_trainable_parameters(model)
 
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.learning_rate,
    )
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)
 
    train_loader = get_cola_dataloader(
        tokenizer,
        split="train",
        batch_size=cfg.training.batch_size,
        block_size=cfg.training.block_size,
    )
    val_loader = get_cola_dataloader(
        tokenizer,
        split="validation",
        batch_size=cfg.training.eval_batch_size,
        block_size=cfg.training.block_size,
        shuffle=False,
    )
    print(device)
 
    training_start_time = time.time()
 
    for epoch in range(cfg.training.num_epochs):
        epoch_start_time = time.time()
 
        if epoch % cfg.training.eval_interval == 0 or epoch == cfg.training.num_epochs - 1:
            val_loss, val_mcc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}, Val MCC = {val_mcc:.4f}")
            wandb.log({"val_loss": val_loss, "val_mcc": val_mcc, "epoch": epoch})
            scheduler.step(val_loss)
 
        train_loss = 0.0
        for idx, batch in enumerate(train_loader):
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), batch["labels"].reshape(-1))
 
            # AdaLoRA Specific: Add orthogonality penalty
            if cfg.peft.type == "adalora":
                ortho_penalty = sum(m.get_orthogonality_penalty() for m in model.modules() if isinstance(m, AdaLoRALinear))
                loss += cfg.adalora.ortho_lambda * ortho_penalty
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            # SoRA Specific: Proximal Gradient Step
            if cfg.peft.type == "sora":
                for module in model.modules():
                    if isinstance(module, SoRALinear):
                        module.proximal_step(threshold=cfg.sora.proximal_threshold)
 
            # FIX 3: Remove the extra leading space that caused an IndentationError.
            # Original line: `             if idx % ...` (13 spaces instead of 12)
            if idx % cfg.training.log_interval == 0 and idx > 0:
                print(f"epoch {epoch}, step {idx}: train loss {train_loss/cfg.training.log_interval:.4f}")
                wandb.log({"train_loss": train_loss / cfg.training.log_interval, "step": idx})
                train_loss = loss.item()
            else:
                train_loss += loss.item()
 
            if idx >= cfg.training.max_iterations_per_epoch:
                break
 
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        wandb.log({"epoch_time_seconds": epoch_time, "epoch": epoch})
 
    total_training_time = time.time() - training_start_time
    print(f"Total training time: {total_training_time:.2f}s")
 
    effective_rank = compute_effective_rank(model)
    print(f"Average Effective Rank: {effective_rank:.4f}")
 
    wandb.log({
        "total_training_time_seconds": total_training_time,
        "average_effective_rank": effective_rank,
    })
    wandb.finish()
 
 
if __name__ == "__main__":
    main()