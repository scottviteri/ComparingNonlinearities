import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def stream_imagenet_chunk(split, start_idx, num_examples, batch_size):
    """
    Loads a subset (chunk) of the ImageNet dataset in streaming mode and returns batches.

    Parameters:
       split (str): "train" or "validation"
       start_idx (int): Number of examples to skip (to rotate through the dataset).
       num_examples (int): Number of examples to take after skipping.
       batch_size (int): Batch size for the output.

    Yields:
       Tuple (images, labels) where images are tensors and labels are tensors.
    """
    from datasets import load_dataset
    # Load the dataset in streaming mode.
    dataset = load_dataset("imagenet-1k", split=split, streaming=True)
    # Skip a fixed number of examples, then take only a subset
    if start_idx > 0:
         dataset = dataset.skip(start_idx)
    dataset = dataset.take(num_examples)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    batch = []
    counter = 0
    for example in dataset:
         counter += 1
         if counter % (batch_size * 10) == 0:
              print(f"[Streaming] Processed {counter} examples")
         try:
              image = transform(example["image"])
         except Exception as e:
              continue
         label = example["label"]
         batch.append((image, label))
         if len(batch) == batch_size:
              images, labels = zip(*batch)
              images = torch.stack(images)
              labels = torch.tensor(labels)
              yield images, labels
              batch = []
    if batch:
         images, labels = zip(*batch)
         images = torch.stack(images)
         labels = torch.tensor(labels)
         yield images, labels

class VectorGELUActivation(nn.Module):
    def forward(self, x):
        # If input is a 4D tensor, assume shape (B, C, H, W) and process it in 4x4 subblocks.
        assert x.dim() == 4
        B, C, H, W = x.shape
        block_size = 4
        # Ensure H and W are divisible by block_size.
        assert H % block_size == 0 and W % block_size == 0, "H and W must be divisible by block_size"
        # Unfold into subblocks; result shape: (B, C, H//block_size, W//block_size, block_size, block_size)
        x_unfold = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        # Flatten each subblock to a vector; new shape: (B, C, H//block_size, W//block_size, block_size*block_size)
        x_flat = x_unfold.contiguous().view(B, C, H//block_size, W//block_size, -1)
        # Compute the L2 norm over the last dimension with keepdim -> shape: (..., 1)
        r = torch.norm(x_flat, p=2, dim=-1, keepdim=True)
        d = x_flat.size(-1)
        a = torch.tensor(d / 2.0, dtype=x.dtype, device=x.device)
        scale = torch.special.gammainc(a, (r ** 2) / 2.0)
        # Apply the scaling to each subblock vector.
        x_flat = x_flat * scale
        # Reshape back to subblock shape.
        x_sub = x_flat.view(B, C, H//block_size, W//block_size, block_size, block_size)
        # Permute dimensions to interleave the subblock dimensions with the block indexes.
        x_sub = x_sub.permute(0, 1, 2, 4, 3, 5).contiguous()
        # Reshape to recover the original (B, C, H, W) structure.
        x_out = x_sub.view(B, C, H, W)
        return x_out

        #else:
        #    # For non-4D inputs, use the default implementation.
        #    r = torch.norm(x, p=2, dim=-1, keepdim=True)
        #    d = x.size(-1)
        #    a = torch.tensor(d / 2.0, dtype=x.dtype, device=x.device)
        #    scale = torch.special.gammainc(a, (r ** 2) / 2.0)
        #    return x * scale

class ConfigurableCNN(nn.Module):
    """
    A configurable CNN model for ImageNet with a specified number of convolution layers.
    The architecture builds a sequence of convolution + activation layers (default 20 layers),
    followed by an adaptive average pool and a final linear classifier.
    """
    def __init__(self, num_layers=20, base_channels=64, activation_fn=nn.GELU):
        super(ConfigurableCNN, self).__init__()
        layers = []
        in_channels = 3
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1))
            layers.append(activation_fn())
            in_channels = base_channels
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels, 1000)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def train_model(activation_name, activation_fn, num_layers):
    # Training configuration
    batch_size = 32
    learning_rate = 1e-5
    max_steps = 5000
    eval_interval = 100  # evaluate every 100 training steps
    global_step = 0
    chunk_counter = 0
    global_gradients = []  # list of (global_step, average gradients per tracked module)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for activation {activation_name}")
    os.makedirs("imagenet_plots", exist_ok=True)
    
    # Initialize lists to record loss over time.
    global_steps_list = []
    train_losses_list = []
    val_losses_list = []
    
    # Create model using the provided activation function.
    model = ConfigurableCNN(num_layers=num_layers, activation_fn=activation_fn).to(device)
    # If using vecgelu activation, modify the random initialization by multiplying each weight by 1.2.
    if activation_name == "vecgelu":
         for name, param in model.named_parameters():
             if "weight" in name and param.dim() > 1:
                 param.data.mul_(2.0)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # If using VectorGELUActivation, set up activation norm tracking.
    if activation_name == "vecgelu":
         # Extract all activation modules from the features.
         act_modules = [m for m in model.features if isinstance(m, VectorGELUActivation)]
         # Prepare dictionaries to accumulate activation norms and scale values for each module.
         act_norm_hist = {i: [] for i in range(len(act_modules))}
         scale_hist = {i: [] for i in range(len(act_modules))}
         # Create a persistent figure for the activation norm plot.
         fig_act, ax_act = plt.subplots()
         ax_act.set_xlabel("Layer")
         ax_act.set_ylabel("Mean Activation Norm")
         ax_act.set_title("Activation Norm by Layer over Training Steps (vecgelu)")
         ax_act.set_xticks(range(len(act_modules)))
         ax_act.set_xticklabels([f"Layer {i+1}" for i in range(len(act_modules))])

         # Create a persistent figure for the scale plot.
         fig_scale, ax_scale = plt.subplots()
         ax_scale.set_xlabel("Layer")
         ax_scale.set_ylabel("Mean Scale")
         ax_scale.set_title("Scale (mean) by Layer over Training Steps (vecgelu)")
         ax_scale.set_xticks(range(len(act_modules)))
         ax_scale.set_xticklabels([f"Layer {i+1}" for i in range(len(act_modules))])

         # Register a forward hook on each activation module.
         for idx, module in enumerate(act_modules):
             def hook(module, input, output, idx=idx):
                 x = input[0]
                 if x.dim() == 4:
                     block_size = 4
                     B, C, H, W = x.shape
                     # Unfold into 4x4 subblocks: shape (B, C, H//block_size, W//block_size, block_size, block_size)
                     x_unfold = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
                     # Flatten each subblock to a vector: shape (B, C, H//block_size, W//block_size, block_size*block_size)
                     x_flat = x_unfold.contiguous().view(B, C, H//block_size, W//block_size, -1)
                     # Compute the L2 norm on each flattened subblock (keepdim=True)
                     r = torch.norm(x_flat, p=2, dim=-1, keepdim=True)
                     # Also compute scale as in the forward() method.
                     d = x_flat.size(-1)
                     a = torch.tensor(d/2.0, dtype=x.dtype, device=x.device)
                     scale = torch.special.gammainc(a, (r ** 2)/2.0)
                 else:
                     r = torch.norm(x, p=2, dim=-1, keepdim=True)
                     d = x.size(-1)
                     a = torch.tensor(d/2.0, dtype=x.dtype, device=x.device)
                     scale = torch.special.gammainc(a, (r ** 2)/2.0)
                 act_norm = r.mean().item()
                 scale_val = scale.mean().item()
                 act_norm_hist[idx].append(act_norm)
                 scale_hist[idx].append(scale_val)
             module.register_forward_hook(hook)
    
    # For gradient tracking, extract each convolution layer and append fc.
    grad_modules = [m for m in model.features if isinstance(m, nn.Conv2d)]
    grad_modules.append(model.fc)
    grad_labels = [f"Conv {i+1}" for i in range(len(grad_modules)-1)] + ["FC"]
    
    # Create a persistent figure for the gradient plot.
    fig, ax = plt.subplots()
    ax.set_xticks(range(len(grad_labels)))
    ax.set_xticklabels(grad_labels)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average Gradient Magnitude")
    ax.set_title(f"Gradient Magnitude by Layer over Training Steps ({activation_name})")

    while global_step < max_steps:
        # Determine training chunk settings.
        train_chunk_size = 300  # process 100 images in this chunk
        start_idx = (chunk_counter * train_chunk_size) % 1281167  # rotate through training data
        train_stream = stream_imagenet_chunk("train", start_idx, train_chunk_size, batch_size)
        
        running_loss = 0.0
        n_train_batches = 0
        interval_gradients = [[] for _ in range(len(grad_modules))]
        
        for inputs, labels in train_stream:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            for idx, module in enumerate(grad_modules):
                grad_vals = []
                for param in module.parameters():
                    if param.grad is not None:
                        grad_vals.append(param.grad.data.abs().mean().item())
                if grad_vals:
                    interval_gradients[idx].append(sum(grad_vals) / len(grad_vals))
            optimizer.step()
            running_loss += loss.item()
            n_train_batches += 1
            global_step += 1
            if global_step % eval_interval == 0:
                break
        avg_interval_grad = []
        for idx in range(len(grad_modules)):
            if interval_gradients[idx]:
                avg_interval_grad.append(sum(interval_gradients[idx]) / len(interval_gradients[idx]))
            else:
                avg_interval_grad.append(0.0)
        global_gradients.append((global_step, avg_interval_grad))
        
        avg_train_loss = running_loss / n_train_batches if n_train_batches > 0 else running_loss
        print(f"[{activation_name}] Step [{global_step}/{max_steps}], Training Loss: {avg_train_loss:.4f}")
        
        model.eval()
        val_chunk_size = 1000
        val_start_idx = (global_step % 50000)
        val_stream = stream_imagenet_chunk("validation", val_start_idx, val_chunk_size, batch_size)
        val_loss = 0.0
        correct = 0
        total = 0
        n_val_batches = 0
        with torch.no_grad():
            for inputs, labels in val_stream:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                n_val_batches += 1
        avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else val_loss
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"[{activation_name}] Step [{global_step}/{max_steps}], Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        x = list(range(len(grad_labels)))
        y = global_gradients[-1][1]
        ax.plot(x, y, marker='o', label=f"Step {global_step}")
        ax.set_xticks(x)
        ax.set_xticklabels(grad_labels)
        ax.legend(loc='upper right')
        fig.savefig(os.path.join("imagenet_plots", f"gradient_plot_{activation_name}.png"))
        
        # For vecgelu, compute and plot the average activation norm and average scale per layer.
        if activation_name == "vecgelu":
            avg_act_norm = []
            avg_scale = []
            for i in range(len(act_modules)):
                if act_norm_hist[i]:
                    avg_val = sum(act_norm_hist[i]) / len(act_norm_hist[i])
                else:
                    avg_val = 0.0
                avg_act_norm.append(avg_val)
                if scale_hist[i]:
                    scale_avg_val = sum(scale_hist[i]) / len(scale_hist[i])
                else:
                    scale_avg_val = 0.0
                avg_scale.append(scale_avg_val)
            # Append new lines to the respective plots.
            ax_act.plot(range(len(act_modules)), avg_act_norm, marker='o', label=f"Step {global_step}")
            ax_act.legend(loc='upper right')
            fig_act.savefig(os.path.join("imagenet_plots", f"activation_norm_plot_{activation_name}.png"))

            ax_scale.plot(range(len(act_modules)), avg_scale, marker='o', label=f"Step {global_step}")
            ax_scale.legend(loc='upper right')
            fig_scale.savefig(os.path.join("imagenet_plots", f"scale_plot_{activation_name}.png"))

            # Reset both histories for the next interval.
            act_norm_hist = {i: [] for i in range(len(act_modules))}
            scale_hist = {i: [] for i in range(len(act_modules))}
        
        # Record loss values for this evaluation interval.
        global_steps_list.append(global_step)
        train_losses_list.append(avg_train_loss)
        val_losses_list.append(avg_val_loss)

        # Plot train and validation loss over time.
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(global_steps_list, train_losses_list, label='Train Loss')
        ax_loss.plot(global_steps_list, val_losses_list, label='Validation Loss')
        ax_loss.set_xlabel("Global Step")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"Train and Validation Loss Over Time ({activation_name})")
        ax_loss.legend()
        fig_loss.savefig(os.path.join("imagenet_plots", f"loss_plot_{activation_name}.png"))
        plt.close(fig_loss)
        
        model.train()
        chunk_counter += 1

if __name__ == '__main__':
    import argparse
    import multiprocessing
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=20,
                        help="Number of convolution layers in the model (default: 20)")
    args = parser.parse_args()
    
    # Spawn two processes for the two activation types.
    p1 = multiprocessing.Process(target=train_model, args=("gelu", nn.GELU, args.num_layers))
    p2 = multiprocessing.Process(target=train_model, args=("vecgelu", VectorGELUActivation, args.num_layers))
    p1.start()
    p2.start()
    p1.join()
    p2.join() 