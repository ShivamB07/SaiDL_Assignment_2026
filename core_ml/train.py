import torch
import hydra
import time
import math
import wandb
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2TokenizerFast
from torch.optim.lr_scheduler import ReduceLROnPlateau


from data.wikitext2 import get_wikitext2_dataloader
from models import Transformer 

wandb.login(key="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")

@torch.no_grad()
def estimate_evaluation_loss(model, eval_loader, device):
    losses = torch.zeros(len(eval_loader), device=device)
    model.eval()
    
    start_time = time.perf_counter()
    total_tokens = 0
    
    for k, (xb, yb) in enumerate(eval_loader):
        xb, yb = xb.to(device), yb.to(device)
        total_tokens += xb.numel() 
        
        _, loss = model(xb, yb)
        losses[k] = loss.item()
        
    end_time = time.perf_counter()
    eval_time = end_time - start_time
    eval_throughput = total_tokens / eval_time if eval_time > 0 else 0
    
    model.train()
    return losses.mean().item(), eval_throughput


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def main(cfg: DictConfig) -> None:
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, config=OmegaConf.to_container(cfg, resolve=True))
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    model = Transformer(
        vocab_size=vocab_size,
        n_embed=cfg.model.n_embed,
        block_size=cfg.model.block_size,
        num_layers=cfg.model.num_layers,
        num_head=cfg.model.num_head,
        dropout_prob=cfg.model.dropout_prob,
        attention_type=cfg.attention.type,
        sliding_window_mask=cfg.attention.sliding_window_mask,
        window_size=cfg.attention.window_size,
        positional_encoding=cfg.attention.positional_encoding,
        group_size=cfg.attention.group_size,
        use_local_conv=cfg.attention.use_local_conv,
        ngram_size=cfg.attention.ngram_size,
        base=cfg.attention.base
    ).to(device)

    print("Model: ")
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimiser.lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="min", factor = 0.5, patience = 2, min_lr = 1e-5)

    train_loader = get_wikitext2_dataloader(
        tokenizer,
        cfg.model.block_size,
        "train",
        cfg.train.batch_size,
        shuffle=True,
    )
    val_loader = get_wikitext2_dataloader(
        tokenizer,
        cfg.model.block_size,
        "validation",
        cfg.train.eval_batch_size,
        shuffle=False,
    )

    print(device)

    for epoch in range(cfg.train.epochs):
        
        if epoch % cfg.train.eval_interval == 0 or epoch == cfg.train.epochs - 1:
            val_loss, eval_throughput = estimate_evaluation_loss(model, val_loader, device)
            
            val_perplexity = math.exp(val_loss) if val_loss < 20 else float('inf') # Avoid overflow for large losses
            wandb.log({"val/loss": val_loss, "val/perplexity": val_perplexity, "val/throughput": eval_throughput})
            scheduler.step(val_loss)
            print(f"\n--- VALIDATION (Epoch {epoch}) ---")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Perplexity: {val_perplexity:.4f}")
            print(f"Eval Throughput: {eval_throughput:.2f} tokens/sec\n")

        
        train_loss = 0
        epoch_train_tokens = 0
        epoch_start_time = time.perf_counter()
        
        for idx, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            epoch_train_tokens += xb.numel()
            
            _, loss = model(xb, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if idx % cfg.train.log_interval == 0 and idx > 0:
                print(f"epoch {epoch}, step {idx}: train loss {train_loss/cfg.train.log_interval:.4f}")
                wandb.log({"train/loss": train_loss/cfg.train.log_interval})
                train_loss = loss.item()  
            else:
                train_loss += loss.item()

            if idx >= cfg.train.max_iterations_per_epoch:
                break  
                
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        train_throughput = epoch_train_tokens / epoch_duration

        print(f"--- Epoch {epoch} Metrics Summary ---")
        print(f"Epoch Time: {epoch_duration:.2f} seconds")
        print(f"Training Throughput: {train_throughput:.2f} tokens/sec")
        print("-----------------------------------\n")

    print("Training completed.")
    torch.save(model.state_dict(), "transformer_wikitext2.pth")


if __name__ == "__main__":
    main()