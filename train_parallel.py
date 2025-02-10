import math
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
# Use a non-interactive backend for standalone scripts.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_dataset
import transformers.models.gpt2.modeling_gpt2 as gpt2_modeling
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import argparse


# ============================================================
# 1. Define Activation Modules
# ============================================================

# GELU (tanh approximation used in GPT-2)
class GELUActivation(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

# Trainable Erf Activation:
#    f(x) = x * erf(x/(sqrt(2)*sigma)) with sigma trainable.
class TrainableErfActivation(nn.Module):
    def __init__(self, init_sigma=1.0, eps=1e-6):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(init_sigma, dtype=torch.float32))
        self.eps = eps
    def forward(self, x):
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        scale = torch.erf(norm/ (math.sqrt(2) * self.sigma))
        return x * scale


class VectorGELUActivation(nn.Module):
    def forward(self, x):
        # Assume x's last dimension is the vector dimension.
        # Compute the norm along the last dimension (with keepdim=True so that it broadcasts)
        r = torch.norm(x, p=2, dim=-1, keepdim=True)
        # Determine the vector dimensionality (d)
        d = x.size(-1)
        # Create a tensor for a = d/2 on the same device and dtype as x.
        a = torch.tensor(d / 2.0, dtype=x.dtype, device=x.device)
        # Compute the regularized lower incomplete gamma function,
        # which equals (1/Γ(a)) ∫_0^(r^2/2) t^(a-1) e^{-t} dt.
        scale = torch.special.gammainc(a, (r ** 2) / 2.0)
        #print("vecgelu", r.mean().item(), scale.mean().item())
        return x * scale

class ProbabilisticVecGELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
         # Compute the norm of each vector (assumed along the last dimension)
         r = input.norm(p=2, dim=-1, keepdim=True)
         # Determine the vector dimension d (from the last dimension)
         d = input.size(-1)
         # Create a tensor a = d/2 on the same device/dtype as input.
         a = torch.tensor(d / 2.0, dtype=input.dtype, device=input.device)
         # Compute probability p = gammainc(a, (r^2)/2)
         p = torch.special.gammainc(a, (r ** 2) / 2.0)
         # Sample a uniform random tensor of the same shape as r
         rand_sample = torch.rand_like(r)
         # Create a binary mask: 1 where the random number is less than p, else 0.
         mask = (rand_sample < p).float()
         ctx.save_for_backward(p)
         # Multiply input by mask (mask broadcasts along the vector dimension)
         return input * mask

    @staticmethod
    def backward(ctx, grad_output):
         # Retrieve the probability p computed during forward.
         p, = ctx.saved_tensors
         # Use a surrogate gradient that scales grad_output by p.
         grad_input = grad_output * p
         return grad_input

class ProbabilisticVecGELUActivation(nn.Module):
    def forward(self, x):
         return ProbabilisticVecGELUFunction.apply(x)


# ReLU: use the built-in module (nn.ReLU)
# ============================================================
# 2. Activation Mapping Dictionary (five variants, including SiLU)
# ============================================================
activation_modules = {
    "trainable_erf": lambda: TrainableErfActivation(),
    "vecgelu": lambda: VectorGELUActivation(),
    "relu": lambda: nn.ReLU(),
    "gelu": lambda: GELUActivation(),
    "silu": lambda: nn.SiLU(),  # Added SiLU activation function.
    "probvecgelu": lambda: ProbabilisticVecGELUActivation(),  # New probabilistic vector GELU.
}

## ============================================================
## Custom GPT2MLP with an explicit activation submodule
## (Re-implemented without calling the parent __init__ to avoid size issues)
## ============================================================
class CustomGPT2MLP(nn.Module):
    def __init__(self, config=None):
        if config is None:
            config = GPT2Config(
                n_embd=256,
                n_layer=6,
                n_head=4,
                vocab_size=50257,
            )
        # Determine the intermediate size—if config.n_inner is not provided, default to 4 * hidden_size.
        intermediate_size = 4 * config.n_embd
        super().__init__()
        embed_dim = config.hidden_size  # should be set to config.n_embd
        # Initialize the two linear layers (Conv1D) using the GPT2 implementation.
        self.c_fc = gpt2_modeling.Conv1D(intermediate_size, embed_dim)
        self.c_proj = gpt2_modeling.Conv1D(embed_dim, intermediate_size)
        # Set the activation function as an explicit submodule.
        self.act = nn.GELU()  # default activation; will be overridden by override_activation()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

def replace_gpt2_mlp_with_custom(model, config):
    # Replace each transformer block's MLP with our custom version.
    for block in model.transformer.h:
        block.mlp = CustomGPT2MLP(config)

# ============================================================
# 3. Custom Callback to Record Metrics and Write Logs/Plots
# ============================================================
class ActivationMetricsCallback(TrainerCallback):
    def __init__(self, grad_cutoff=1e-5, neuron_threshold=0.05, log_file=None, eval_interval=500):
        super().__init__()
        self.grad_cutoff = grad_cutoff
        self.neuron_threshold = neuron_threshold
        self.log_file = log_file
        self.eval_interval = eval_interval
        self.metrics_log = {}  # global_step -> metrics dict
        self.activation_means = []        # collected from forward hooks
        self.vector_norms = []             # for vector activations: norm of each vector
        self.scales = []                   # for vector activations: scale applied
        self.dead_neuron_fractions = []     # collected from forward hooks
        self.gradients = {}  # to store gradients per parameter

    def on_train_begin(self, args, state, control, **kwargs):
        self.metrics_log = {}
        self.activation_means = []
        self.vector_norms = []
        self.scales = []
        self.dead_neuron_fractions = []
        self.gradients = {}
        self.hooks = []
        self.param_hooks = []
        model = kwargs["model"]
        self.act_module_names = []
        # Register forward hooks only on our custom GPT2MLP modules.
        for name, module in model.named_modules():
            if isinstance(module, CustomGPT2MLP):
                hook = module.act.register_forward_hook(self._forward_hook)
                self.hooks.append(hook)
                self.act_module_names.append(name + ".act")
        print("Registered activation hooks on:", self.act_module_names)

    def _forward_hook(self, module, input, output):
        # output: tensor of shape [batch, seq_length, hidden_dim]
        mean_val = output.mean().item()
        self.activation_means.append(mean_val)
        
        # For vector activations, compute norm and scale.
        if (isinstance(module, VectorGELUActivation) or isinstance(module, ProbabilisticVecGELUActivation)):
            # Compute the norm of each vector (batch, seq_len, 1)
            r = torch.norm(output, p=2, dim=-1, keepdim=False)  # shape: [batch, seq_len]
            self.vector_norms.extend(r.flatten().tolist())
            # Compute the scale applied (batch, seq_len, 1)
            if isinstance(module, VectorGELUActivation):
                d = output.size(-1)
                a = torch.tensor(d / 2.0, dtype=output.dtype, device=output.device)
                scale = torch.special.gammainc(a, (r ** 2) / 2.0)
                self.scales.extend(scale.flatten().tolist())
            elif isinstance(module, ProbabilisticVecGELUActivation):
                # For probabilistic, the scale is the mask (0 or 1)
                self.scales.extend(output.any(dim=-1).float().flatten().tolist())
        
        neuron_mean_abs = output.abs().mean(dim=(0, 1))  # shape: [hidden_dim]
        dead_frac = (neuron_mean_abs < self.neuron_threshold).float().mean().item()
        self.dead_neuron_fractions.append(dead_frac)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # Get the model from kwargs (the Trainer passes the model)
        model = kwargs.get("model")
        if model is None:
            return control

        grad_norms = {}
        total_grad_norm = 0.0
        total_grad_elements = 0
        vanishing_grad_elements = 0
        # Iterate over model parameters
        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                grad_norms[name] = norm
                total_grad_norm += norm
                # Also count grad elements and how many are below our threshold
                grad_data = param.grad.data
                total_grad_elements += grad_data.numel()
                vanishing_grad_elements += (grad_data.abs() < self.grad_cutoff).sum().item()
        
        grad_vanish_fraction = (vanishing_grad_elements / total_grad_elements) if total_grad_elements > 0 else 0.0
        # Debug prints commented out:
        # print(f"[DEBUG] Step {state.global_step} gradient norms: {grad_norms}")
        # print(f"[DEBUG] Total grad norm: {total_grad_norm:.6e}, Grad vanish fraction: {grad_vanish_fraction:.6e}")

        # Save the values in _grad_metrics for later logging in on_step_end.
        self._grad_metrics = {"grad_vanish_fraction": grad_vanish_fraction, "grad_norm": total_grad_norm}
        return control

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        total_weight_elements = 0
        dead_weight_elements = 0
        sum_abs_weight = 0.0
        for param in model.parameters():
            if param.requires_grad:
                weight = param.detach()
                total_weight_elements += weight.numel()
                dead_weight_elements += (weight.abs() < self.neuron_threshold).sum().item()
                sum_abs_weight += weight.abs().sum().item()
        avg_weight = (sum_abs_weight / total_weight_elements) if total_weight_elements > 0 else 0.0
        dead_weight_fraction = (dead_weight_elements / total_weight_elements) if total_weight_elements > 0 else 0.0

        avg_activation = sum(self.activation_means)/len(self.activation_means) if self.activation_means else 0.0
        avg_vector_norm = sum(self.vector_norms)/len(self.vector_norms) if self.vector_norms else 0.0
        avg_scale = sum(self.scales)/len(self.scales) if self.scales else 0.0
        avg_dead_neuron_frac = sum(self.dead_neuron_fractions)/len(self.dead_neuron_fractions) if self.dead_neuron_fractions else 0.0

        grad_metrics = self._grad_metrics if hasattr(self, "_grad_metrics") else {"grad_vanish_fraction": 0.0, "grad_norm": 0.0}

        # For TrainableErfActivation, record the learned sigma.
        learned_sigma = None
        count = 0
        for module in model.modules():
            if isinstance(module, TrainableErfActivation):
                learned_sigma = module.sigma.item() if learned_sigma is None else learned_sigma + module.sigma.item()
                count += 1
        if count > 0:
            learned_sigma /= count

        # Attempt to capture loss and evaluation loss.
        # If 'logs' is not passed in on_step_end, fall back to the latest logs captured in on_log.
        logs = kwargs.get("logs", {}) or getattr(self, "latest_logs", {})
        loss_val = logs.get("loss", None)
        eval_loss_val = logs.get("eval_loss", None)

        step = state.global_step
        self.metrics_log[str(step)] = {
            "grad_vanish_fraction": grad_metrics["grad_vanish_fraction"],
            "grad_norm": grad_metrics["grad_norm"],
            "dead_neuron_fraction": avg_dead_neuron_frac,
            "avg_activation": avg_activation,
            "avg_vector_norm": avg_vector_norm,
            "avg_scale": avg_scale,
            "avg_weight": avg_weight,
            "dead_weight_fraction": dead_weight_fraction,
            "learned_sigma": learned_sigma,
            "loss": loss_val,
            "eval_loss": eval_loss_val,
        }

        # Print progress less often (every 4*eval_interval steps).
        if step % (self.eval_interval * 4) == 0:
            print(f"Step {step}: {self.metrics_log.get(str(step), 'No log recorded')}")

        # Append metrics for the current step to the log file in JSONL format.
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                record = {"step": step}
                record.update(self.metrics_log[str(step)])
                f.write(json.dumps(record) + "\n")

        # Clear temporary lists and gradient metrics.
        self.activation_means = []
        self.vector_norms = []
        self.scales = []
        self.dead_neuron_fractions = []
        self.gradients = {}
        if hasattr(self, "_grad_metrics"):
            del self._grad_metrics

    def on_train_end(self, args, state, control, **kwargs):
        for hook in self.hooks:
            hook.remove()
        # (Final logging is already handled in on_step_end, so no need to overwrite the file here.)

# ============================================================
# 4. Helper function to override activations in the model.
# ============================================================
def override_activation(model, activation_name):
    if activation_name not in activation_modules:
        raise ValueError(f"Unsupported activation: {activation_name}")
    # Only override the activation for our custom GPT2 MLP modules.
    for module in model.modules():
        if isinstance(module, CustomGPT2MLP):
            module.act = activation_modules[activation_name]()
    return model

# ============================================================
# 5. Training function.
# ============================================================
def train_model(activation_name="gelu", run_name="gelu", max_steps=100000, logging_steps=200, eval_interval=50, batch_size=8, num_layers=48):
    # Set up model, tokenizer, and activation replacement.
    config = GPT2Config(
        n_embd=1600,
        n_layer=48,
        n_head=25,
        vocab_size=50257,
    )
    model = GPT2LMHeadModel(config)
    replace_gpt2_mlp_with_custom(model, config)
    override_activation(model, activation_name)
    
    # For vector activations, multiply weights by 3
    if 'vec' in activation_name.lower():
        for param in model.parameters():
            if isinstance(param, torch.nn.Parameter):
                param.data *= 1.1
                
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # Prepare training dataset in streaming mode.
    train_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    # Tokenize each example on the fly.
    def tokenize_function(example):
         return tokenizer(example["text"], truncation=True, max_length=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create a generator that yields properly collated batches.
    def stream_batches(dataset, batch_size, tokenize_fn, collate_fn):
         batch = []
         for example in dataset:
              tokenized = tokenize_fn(example)
              # Skip examples that produce empty sequences.
              if len(tokenized["input_ids"]) == 0:
                   continue
              batch.append(tokenized)
              if len(batch) == batch_size:
                   yield collate_fn(batch)
                   batch = []
         if batch:
              yield collate_fn(batch)

    train_batch_generator = stream_batches(train_dataset, batch_size=batch_size, tokenize_fn=tokenize_function, collate_fn=data_collator)

    # Prepare evaluation dataset.
    eval_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.filter(lambda x: len(x["input_ids"]) > 0)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=data_collator)

    # Set up optimizer.
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize our callback.
    log_file = f"logs/log_{run_name}.jsonl"
    # Clear existing log file if it exists.
    if os.path.exists(log_file):
         os.remove(log_file)
    callback = ActivationMetricsCallback(log_file=log_file, eval_interval=eval_interval)

    # Create a simple state object.
    class TrainState:
         pass
    state = TrainState()
    state.global_step = 0

    # Manually invoke the callback's on_train_begin to register forward hooks.
    callback.on_train_begin(None, state, None, model=model)

    model.train()
    # Use our streaming generator to iterate until max_steps is reached.
    for batch in train_batch_generator:
         state.global_step += 1
         step = state.global_step

         # Move batch to device.
         batch = {k: v.to(device) for k, v in batch.items()}

         # Forward pass (inputs and labels are the same for LM).
         outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
         loss = outputs.loss

         optimizer.zero_grad()
         loss.backward()
         callback.on_pre_optimizer_step(None, state, None, model=model)
         optimizer.step()

         # Every eval_interval steps, do a quick evaluation averaged over eval_interval//10 batches.
         if step % eval_interval == 0:
              model.eval()
              with torch.no_grad():
                   num_eval_batches = max(1, eval_interval // 10)
                   eval_losses = []
                   batch_count = 0
                   for eval_batch in eval_loader:
                        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                        eval_outputs = model(input_ids=eval_batch["input_ids"], labels=eval_batch["input_ids"])
                        eval_losses.append(eval_outputs.loss.item())
                        batch_count += 1
                        if batch_count >= num_eval_batches:
                             break
                   eval_loss = sum(eval_losses) / len(eval_losses)
              model.train()
         else:
              eval_loss = None

         # Create logs dictionary.
         logs = {"loss": loss.item()}
         if eval_loss is not None:
              logs["eval_loss"] = eval_loss

         callback.on_step_end(None, state, None, model=model, logs=logs)

         if step % (eval_interval * 4) == 0:
              print(f"Step {step}: {callback.metrics_log.get(str(step), 'No log recorded')}")

         if step >= max_steps:
              break

    # Manually invoke on_train_end after training execution.
    callback.on_train_end(None, state, None, model=model)
    print(f"Training complete for '{run_name}'.")
    return None, callback.metrics_log

def update_combined_plots_from_files(metrics_dict, metric_names):
    # Define a simple moving average smoothing function.
    def smooth(values, window_size=5):
         if len(values) < window_size:
              return values
         # Use convolution in 'valid' mode so the output length is len(values) - window_size + 1.
         return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

    for metric in metric_names:
        plt.figure(figsize=(12,6))
        steps_counts = []
        for variant, ml in metrics_dict.items():
            variant_steps = sorted([int(s) for s in ml.keys() if ml[s].get(metric) is not None])
            if variant_steps:
                steps_counts.append(len(variant_steps))
        if not steps_counts:
            continue
        min_steps_count = min(steps_counts)
        for variant, ml in metrics_dict.items():
            variant_steps = sorted([int(s) for s in ml.keys() if ml[s].get(metric) is not None])
            if variant_steps:
                truncated_steps = variant_steps[:min_steps_count]
                values = [ml[str(s)][metric] for s in truncated_steps]
                # If the metric is loss or eval_loss, apply smoothing.
                if metric in ["loss", "eval_loss"]:
                    window_size = 20  # Increased smoothing window.
                    # Apply smoothing only if we have enough data.
                    if len(values) >= window_size:
                        smoothed_values = smooth(values, window_size)
                        # Adjust the steps to match the length of smoothed values.
                        smoothed_steps = truncated_steps[window_size - 1:]
                        plt.plot(smoothed_steps, smoothed_values, label=variant)
                    else:
                        plt.plot(truncated_steps, values, label=variant)
                else:
                    plt.plot(truncated_steps, values, label=variant)
        plt.xlabel("Training Step")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} Over Time")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/combined_{metric}.png")
        plt.close()

if __name__ == '__main__':
    # Create directories for logs and plots if they don't exist.
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", type=str, default=None,
                        help="Specify a single activation function to run (e.g. gelu, silu, relu, etc.)")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum training steps (default: 100000)")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Logging steps interval (default: 50)")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="Evaluation interval (default: 100)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--num_layers", type=int, default=48,
                        help="Number of layers in the model (default: 48)")
    parser.add_argument("--plot_only", action="store_true",
                        help="If set, do not train; instead refresh plots using existing log files.")
    parser.add_argument("--smoothing_window", type=int, default=40,
                        help="Smoothing window for loss/eval_loss plots (default: 40)")
    parser.add_argument("--plot_start", type=int, default=0,
                        help="Starting training step to begin plotting from (default: 0)")
    args = parser.parse_args()

    if args.plot_only:
         # Refresh plots from existing log files without training.
         # If --activation is provided, only refresh that variant; otherwise, refresh all logs.
         if args.activation is not None:
              variants = [args.activation]
         else:
              # Use all log files in the "logs" folder.
              variants = [fname[4:-6] for fname in os.listdir("logs") if fname.startswith("log_") and fname.endswith(".jsonl")]
         combined_logs = {}
         for variant in variants:
              log_file = f"logs/log_{variant}.jsonl"
              if os.path.exists(log_file):
                   try:
                        with open(log_file, "r") as f:
                             file_logs = {}
                             for line in f:
                                  if line.strip():
                                       rec = json.loads(line)
                                       file_logs[str(rec["step"])] = rec
                             combined_logs[variant] = file_logs
                   except Exception as e:
                        print(f"Error reading log file for {variant}: {e}")
         metrics_list = ["grad_vanish_fraction", "grad_norm", "dead_neuron_fraction", "avg_activation", "avg_weight", "dead_weight_fraction", "learned_sigma", "loss", "eval_loss"]
         metrics_list += ["avg_vector_norm", "avg_scale"]  # Add new metrics for vector activations
         # Define a local update function that uses the provided smoothing window.
         def update_combined_plots_with_smoothing(metrics_dict, metric_names, smoothing_window, plot_start):
             def smooth(values, window_size=smoothing_window):
                  if len(values) < window_size:
                       return values
                  return np.convolve(values, np.ones(window_size)/window_size, mode='valid')
             for metric in metric_names:
                 plt.figure(figsize=(12,6))
                 steps_counts = []
                 for variant, ml in metrics_dict.items():
                      # Convert steps to ints and filter by plot_start.
                      variant_steps = sorted([int(s) for s in ml.keys() if ml[s].get(metric) is not None and int(s) >= plot_start])
                      if variant_steps:
                           steps_counts.append(len(variant_steps))
                 if not steps_counts:
                      continue
                 min_steps_count = min(steps_counts)
                 for variant, ml in metrics_dict.items():
                      variant_steps = sorted([int(s) for s in ml.keys() if ml[s].get(metric) is not None and int(s) >= plot_start])
                      if variant_steps:
                           truncated_steps = variant_steps[:min_steps_count]
                           values = [ml[str(s)][metric] for s in truncated_steps]
                           # Apply smoothing for loss/eval_loss if sufficient data exists.
                           if metric in ["loss", "eval_loss"]:
                                if len(values) >= smoothing_window:
                                     smoothed_values = smooth(values, smoothing_window)
                                     smoothed_steps = truncated_steps[smoothing_window - 1:]
                                     plt.plot(smoothed_steps, smoothed_values, label=variant)
                                else:
                                     plt.plot(truncated_steps, values, label=variant)
                           else:
                                plt.plot(truncated_steps, values, label=variant)
                 plt.xlabel("Training Step")
                 plt.ylabel(metric.replace("_", " ").title())
                 plt.title(f"{metric.replace('_', ' ').title()} Over Time")
                 plt.legend()
                 plt.grid(True)
                 plt.savefig(f"plots/combined_{metric}.png")
                 plt.close()
         update_combined_plots_with_smoothing(combined_logs, metrics_list, args.smoothing_window, args.plot_start)
         print("Plot refresh complete.")
         exit(0)
    elif args.activation is not None:
         print(f"Running training for activation '{args.activation}' in single-process mode")
         train_model(
             activation_name=args.activation, 
             run_name=args.activation, 
             max_steps=args.max_steps, 
             logging_steps=args.logging_steps, 
             eval_interval=args.eval_interval,
             batch_size=args.batch_size,
             num_layers=args.num_layers
         )
    else:
         # 6. Run training experiments in parallel.
         # "trainable_erf" is commented out by default. To include it, add it to the list.
         selected_variants = ["vecgelu"]
         log_histories = {}
         metrics_logs = {}
         with ProcessPoolExecutor(max_workers=len(selected_variants), mp_context=multiprocessing.get_context("spawn")) as executor:
              futures = {executor.submit(train_model, activation_name=v, run_name=v,
                           max_steps=args.max_steps, logging_steps=args.logging_steps, eval_interval=args.eval_interval,
                           batch_size=args.batch_size, num_layers=args.num_layers): v for v in selected_variants}
              poll_interval = 10  # seconds
              metrics_list = ["grad_vanish_fraction", "grad_norm", "dead_neuron_fraction", "avg_activation", "avg_weight", "dead_weight_fraction", "learned_sigma", "loss", "eval_loss", "avg_vector_norm", "avg_scale"]
   
              # Poll and update combined plots while training is still ongoing.
              while not all(future.done() for future in futures):
                   combined_logs = {}
                   for variant in selected_variants:
                        log_file = f"logs/log_{variant}.jsonl"
                        if os.path.exists(log_file):
                             try:
                                  with open(log_file, "r") as f:
                                       file_logs = {}
                                       for line in f:
                                            if line.strip():
                                                 rec = json.loads(line)
                                                 file_logs[str(rec["step"])] = rec
                                       combined_logs[variant] = file_logs
                             except Exception as e:
                                  print(f"Error reading log file for {variant}: {e}")
                   if combined_logs:
                        update_combined_plots_from_files(combined_logs, metrics_list)
                   time.sleep(poll_interval)
 
              # After training processes are completed, update final metrics.
              for future in as_completed(futures):
                   variant = futures[future]
                   try:
                        _, metrics_log = future.result()
                        metrics_logs[variant] = metrics_log
                        print(f"Completed training for variant: {variant}")
                   except Exception as e:
                        print(f"Error training with activation {variant}: {e}")
 
              # Final combined plot update.
              combined_logs = {}
              for variant in selected_variants:
                   log_file = f"logs/log_{variant}.jsonl"
                   if os.path.exists(log_file):
                        with open(log_file, "r") as f:
                             file_logs = {}
                             for line in f:
                                  if line.strip():
                                       rec = json.loads(line)
                                       file_logs[str(rec["step"])] = rec
                             combined_logs[variant] = file_logs
              if combined_logs:
                   update_combined_plots_from_files(combined_logs, metrics_list)

    # End of main.