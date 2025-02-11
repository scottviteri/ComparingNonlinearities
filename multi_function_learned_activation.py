"""
This script trains a multi-task network to approximate two functions 
simultaneously: a sinusoid (sin(x)) and the Cantor indicator function.
The network uses a learned activation module whose weights are initialized
to (nearly) the identity (with small random epsilon perturbations). 
"""

import argparse
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from contextlib import nullcontext

# -----------------------
# TARGET FUNCTIONS
# -----------------------

def cantor_indicator(x, generation=3):
    """
    Compute the Cantor indicator for x (assumed in [-1,1]).
    The indicator is 1 if x is in the Cantor set (up to a given generation), else 0.
    
    Assumes x is a pytorch tensor.
    """
    # Convert x to its fractional part (x mod 1), making the Cantor indicator periodic over [n, n+1]
    t = x - torch.floor(x)
    indicator = torch.ones_like(t)
    for i in range(generation):
        t = 3 * t
        digit = torch.floor(t + 1e-5)
        # If digit equals 1, mark it as not in the Cantor set.
        indicator[digit == 1] = 0
        t = t - digit
        t = torch.clamp(t, 0, 1)
    return indicator

def target_sin(x):
    return torch.sin(x)

def target_cantor(x, generation=3):
    return cantor_indicator(x, generation=generation)

# -----------------------
# LEARNED ACTIVATION MODULE
# -----------------------

class LearnedActivation(nn.Module):
    def __init__(self, hidden_size=32):
        """
        Learns a nonlinearity via a small network.
        Initialization:
            fc1 copies its input,
            fc2 and fc3 are initially the identity,
            fc4 averages the inputs.
        A small epsilon perturbation (1e-3) is added to break symmetry.
        """
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        # Initialize fc1 to copy input: output = x for each neuron.
        nn.init.constant_(self.fc1.weight, 1.0)
        nn.init.constant_(self.fc1.bias, 0.0)
        with torch.no_grad():
            self.fc1.weight.add_(1e-3 * torch.randn_like(self.fc1.weight))

        # Initialize fc2 as the identity.
        with torch.no_grad():
            self.fc2.weight.copy_(torch.eye(hidden_size))
            self.fc2.weight.add_(1e-3 * torch.randn_like(self.fc2.weight))
        nn.init.constant_(self.fc2.bias, 0.0)

        # Initialize fc3 as the identity.
        with torch.no_grad():
            self.fc3.weight.copy_(torch.eye(hidden_size))
            self.fc3.weight.add_(1e-3 * torch.randn_like(self.fc3.weight))
        nn.init.constant_(self.fc3.bias, 0.0)

        # Initialize fc4 to average the inputs.
        with torch.no_grad():
            self.fc4.weight.copy_(torch.ones(1, hidden_size) / hidden_size)
            self.fc4.weight.add_(1e-3 * torch.randn_like(self.fc4.weight))
        nn.init.constant_(self.fc4.bias, 0.0)

    def forward(self, x):
        # This learned activation applies elementwise transformation.
        orig_shape = x.shape
        # Flatten to (-1, 1) so that each scalar is processed independently.
        x_flat = x.view(-1, 1)
        out = self.fc1(x_flat)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out.view(orig_shape)

# -----------------------
# MULTI-TASK NETWORK
# -----------------------

class MultiFunctionNet(nn.Module):
    def __init__(self, hidden_size=50, num_layers=3):
        """
        A simple feedforward network that shares a learned activation module.
        It takes a scalar input and outputs two values:
            - The first target is sin(x)
            - The second target is the Cantor indicator of x.
        """
        super().__init__()
        self.input_layer = nn.Linear(1, hidden_size)
        # Use LearnedActivation as the nonlinearity; here we set its hidden
        # size equal to the network's hidden_size.
        self.act1 = LearnedActivation(hidden_size=hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.act2 = LearnedActivation(hidden_size=hidden_size)
        self.output_layer = nn.Linear(hidden_size, 2)  # two outputs

    def forward(self, x):
        out = self.input_layer(x)
        out = self.act1(out)
        out = self.hidden_layer(out)
        out = self.act2(out)
        out = self.output_layer(out)
        return out

# -----------------------
# TRAINING & PLOTTING
# -----------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Instantiate the network and optimizer.
    net = MultiFunctionNet(hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Optionally, load previously saved learned activation weights.
    if args.load_learned_activation and os.path.exists(args.learned_activation_file):
        net.load_state_dict(torch.load(args.learned_activation_file))
        print(f"Loaded network weights from {args.learned_activation_file}")
    else:
        if args.load_learned_activation:
            print(f"File {args.learned_activation_file} not found; starting from random initialization.")

    loss_history = []
    
    # For plotting, we also save the network predictions periodically.
    predictions_history = []
    # For the learned activation (act1), store historical snapshots.
    learned_activation_history = []
    x_plot = torch.linspace(args.x_range[0], args.x_range[1], args.num_plot_points).unsqueeze(1).to(device)
    
    profile_context = (torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available(),
                        record_shapes=True) if args.profile else nullcontext())
    with profile_context as prof:
        for step in range(args.steps):
            # Sample training data uniformly in the x_range.
            x_train = torch.rand((args.batch_size, 1), device=device) * (args.x_range[1] - args.x_range[0]) + args.x_range[0]
            # Compute targets.
            y_target_sin = target_sin(x_train)
            y_target_cantor = target_cantor(x_train, generation=args.generation)
            # Combine targets into 2D tensor.
            y_train = torch.cat([y_target_sin, y_target_cantor], dim=1)

            # Forward, loss, and backward.
            optimizer.zero_grad()
            y_pred = net(x_train)
            loss = nn.MSELoss()(y_pred, y_train)
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            # Periodically record predictions.
            if step % args.plot_interval == 0:
                net.eval()
                with torch.no_grad():
                    y_pred_plot = net(x_plot).cpu().numpy()
                predictions_history.append((step, y_pred_plot))
                
                # Update learned activation history (from act1)
                with torch.no_grad():
                    x_act = torch.linspace(args.activation_range[0], args.activation_range[1],
                                             args.num_plot_points).unsqueeze(1).to(device)
                    y_act_current = net.act1(x_act).cpu().detach().numpy()
                learned_activation_history.append((step, y_act_current))
                
                net.train()

                print(f"Step {step}, Loss: {loss.item():.6f}")

                # Save network weights (which include learned activation submodules)
                torch.save(net.state_dict(), args.learned_activation_file)

                # Generate and save an intermediate plot updating predictions vs. ground truth.
                x_plot_np = x_plot.cpu().numpy()
                y_true_sin = target_sin(x_plot).cpu().numpy()
                y_true_cantor = target_cantor(x_plot, generation=args.generation).cpu().numpy()

                fig_live, axs_live = plt.subplots(1, 2, figsize=(12, 5))
                # Sinusoid task plot.
                axs_live[0].plot(x_plot_np, y_true_sin, label="True sin(x)", color="black", linewidth=2)
                axs_live[0].plot(x_plot_np, y_pred_plot[:, 0], label="Predicted sin(x)", color="red", linestyle="--")
                axs_live[0].set_title("Sinusoid Task")
                axs_live[0].legend()

                # Cantor indicator task plot.
                axs_live[1].plot(x_plot_np, y_true_cantor, label="True Cantor indicator", color="black", linewidth=2)
                axs_live[1].plot(x_plot_np, y_pred_plot[:, 1], label="Predicted Cantor indicator", color="blue", linestyle="--")
                axs_live[1].set_title("Cantor Indicator Task")
                axs_live[1].legend()

                fig_live.tight_layout()
                os.makedirs("multi_function_plots", exist_ok=True)
                fig_live.savefig("multi_function_plots/intermediate.png")
                plt.close(fig_live)

                # Also update the learned activation plot periodically (overwrite the same file).
                x_act = torch.linspace(args.activation_range[0], args.activation_range[1],
                                         args.num_plot_points).unsqueeze(1).to(device)
                import matplotlib.cm as cm
                cmap = cm.get_cmap("Greys")
                fig_la, ax_la = plt.subplots(figsize=(6, 4))
                if learned_activation_history:
                     n_events = len(learned_activation_history)
                     for idx, (s, hist_y) in enumerate(learned_activation_history):
                          # Normalize index so older snapshots are lighter.
                          norm_val = (idx + 1) / n_events
                          color_event = cmap(norm_val)
                          ax_la.plot(x_act.cpu().numpy(), hist_y, color=color_event, linestyle="--", linewidth=1)
                with torch.no_grad():
                     y_act_current = net.act1(x_act).cpu().detach().numpy()
                # Plot current activation function with a solid line.
                ax_la.plot(x_act.cpu().numpy(), y_act_current, color="navy", linewidth=2)
                ax_la.axvline(x=0, color="gray", linestyle="dotted")
                ax_la.set_title("Learned Activation (act1)")
                fig_la.tight_layout()
                fig_la.savefig("multi_function_plots/learned_activation_plot.png")
                plt.close(fig_la)

    if args.profile:
        print(prof.key_averages().table(sort_by="self_cuda_time_total"))

    # Plot final predictions vs ground truth.
    x_plot_np = x_plot.cpu().numpy()
    # Ground truth for each function.
    y_true_sin = target_sin(x_plot).cpu().detach().numpy()
    y_true_cantor = target_cantor(x_plot, generation=args.generation).cpu().detach().numpy()

    # Create a figure with two subplots.
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Sinusoid task.
    axs[0].plot(x_plot_np, y_true_sin, label="True sin(x)", color="black", linewidth=2)
    axs[0].plot(x_plot_np, predictions_history[-1][1][:, 0], label="Predicted sin(x)", color="red", linestyle="--")
    axs[0].set_title("Sinusoid Task")
    axs[0].legend()
    # Cantor task.
    axs[1].plot(x_plot_np, y_true_cantor, label="True Cantor indicator", color="black", linewidth=2)
    axs[1].plot(x_plot_np, predictions_history[-1][1][:, 1], label="Predicted Cantor indicator", color="blue", linestyle="--")
    axs[1].set_title("Cantor Indicator Task")
    axs[1].legend()
    fig.tight_layout()
    os.makedirs("multi_function_plots", exist_ok=True)
    fig.savefig("multi_function_plots/multi_function_fit.png")
    plt.close(fig)

    # Additionally, plot the learned activation function using historical snapshots.
    x_act = torch.linspace(args.activation_range[0], args.activation_range[1],
                             args.num_plot_points).unsqueeze(1).to(device)
    import matplotlib.cm as cm
    cmap = cm.get_cmap("Greys")
    fig_la, ax_la = plt.subplots(figsize=(6, 4))
    if learned_activation_history:
         n_events = len(learned_activation_history)
         for idx, (s, hist_y) in enumerate(learned_activation_history):
              # Normalize index so older snapshots are lighter.
              norm_val = (idx + 1) / n_events
              color_event = cmap(norm_val)
              ax_la.plot(x_act.cpu().numpy(), hist_y, color=color_event, linestyle="--", linewidth=1)
    with torch.no_grad():
         y_act_current = net.act1(x_act).cpu().detach().numpy()
    # Plot current activation function with a solid line.
    ax_la.plot(x_act.cpu().numpy(), y_act_current, color="navy", linewidth=2)
    ax_la.axvline(x=0, color="gray", linestyle="dotted")
    ax_la.set_title("Learned Activation (act1)")
    fig_la.tight_layout()
    fig_la.savefig("multi_function_plots/learned_activation_plot.png")
    plt.close(fig_la)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-task network with learned activation to fit sin(x) and Cantor indicator simultaneously."
    )
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Total number of layers (input, hidden, output layers).")
    parser.add_argument("--hidden_size", type=int, default=50,
                        help="Hidden layer size for the network.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of training steps.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--plot_interval", type=int, default=1000,
                        help="Number of steps between plot updates.")
    parser.add_argument("--num_plot_points", type=int, default=1000,
                        help="Number of points for plotting.")
    parser.add_argument("--x_range", type=float, nargs=2, default=[-1.0, 1.0],
                        help="Lower and upper limits for training data fitting interval.")
    parser.add_argument("--activation_range", type=float, nargs=2, default=[-2.0, 2.0],
                        help="Lower and upper limits for plotting the learned activation function.")
    parser.add_argument("--generation", type=int, default=3,
                        help="Fractal generation level for Cantor indicator (digit expansion depth).")
    parser.add_argument("--learned_activation_file", type=str, default="multi_learned_activation_weights.pt",
                        help="File path to save/load the learned activation weights.")
    parser.add_argument("--load_learned_activation", action="store_true",
                        help="If provided, load previously saved learned activation weights before training.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA.")
    parser.add_argument("--profile", action="store_true",
                        help="Profile the training loop using PyTorch's autograd profiler.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug prints during training.")
    args = parser.parse_args()
    main(args) 