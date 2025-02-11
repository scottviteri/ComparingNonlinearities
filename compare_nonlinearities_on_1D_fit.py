import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from contextlib import nullcontext

def build_net(num_layers, hidden_size, activation, use_complex=False):
    layers = []
    if num_layers == 1:
        if use_complex:
            layers.append(ComplexLinear(1, 1))
        else:
            layers.append(nn.Linear(1, 1))
    else:
        # First layer: input -> hidden
        if use_complex:
            layers.append(ComplexLinear(1, hidden_size))
            layers.append(activation())
        else:
            layers.append(nn.Linear(1, hidden_size))
            layers.append(activation())
        # Middle layers (if any): hidden -> hidden
        for _ in range(num_layers - 2):
            if use_complex:
                layers.append(ComplexLinear(hidden_size, hidden_size))
                layers.append(activation())
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(activation())
        # Final layer: hidden -> output
        if use_complex:
            layers.append(ComplexLinear(hidden_size, 1))
        else:
            layers.append(nn.Linear(hidden_size, 1))
    return nn.Sequential(*layers)

def cantor_indicator(x, generation):
    # Map input x from [-1,1] to [0,1]. It is assumed that x is in the range [-1,1].
    t = (x + 1) / 2.0
    # Start with indicator 1 everywhere (meaning "in the Cantor set").
    indicator = torch.ones_like(t)
    for i in range(generation):
        t = 3 * t
        digit = torch.floor(t + 1e-5)
        # If the digit equals 1, mark as not in the Cantor set.
        indicator[digit == 1] = 0
        # Keep only the fractional part and clamp to [0,1] to avoid negative rounding errors.
        t = t - digit
        t = torch.clamp(t, 0, 1)
    return indicator

def target_function(x, generation=3):
    return cantor_indicator(x, generation=generation)

def sample_cantor_set(num_samples, generation, device=torch.device("cpu"), debug=False):
    """
    Generate num_samples points in [-1,1] that lie in the Cantor set by
    generating a ternary expansion using random choices from {0,2}.
    
    The output points are computed as 2*(ternary number) - 1, so they lie in [-1,1].
    This function relies on `cantor_indicator`, which assumes its input is in [-1,1].
    """
    # Each sample will have 'generation' digits: either 0 or 2.
    digits = torch.randint(0, 2, (num_samples, generation), device=device) * 2  # each digit is 0 or 2
    # Cast digits to double for improved numerical precision
    digits = digits.double()
    # Compute the ternary number, i.e. sum(digit/3^i)
    powers = 3 ** torch.arange(1, generation+1, device=device, dtype=torch.float64)  # shape (generation,)
    sample = (digits / powers).sum(dim=1, keepdim=True)  # shape (num_samples, 1)
    # Map from [0,1] to [-1,1]: x = 2*sample - 1
    result = 2 * sample - 1
    if debug:
        # Run the indicator check in double precision for accuracy
        indicator = cantor_indicator(result, generation=generation)
        frac_correct = torch.mean(indicator==1, dtype=torch.float64).item()
        if not torch.all(indicator == 1):
            fail_mask = (indicator == 0)
            failing_results = result[fail_mask]
            failing_t = (failing_results + 1) / 2.0
            print("Debug: Found", fail_mask.sum().item(), "failing samples out of", num_samples)
            fail_indices = torch.nonzero(fail_mask, as_tuple=True)[0]
            for i in range(min(5, len(fail_indices))):
                idx = fail_indices[i].item()
                sample_digits = digits[idx].tolist()
                sample_ternary = sample[idx].item()
                print(f"Sample {idx}: digits: {sample_digits}, ternary sum: {sample_ternary:.6f}, result: {result[idx].item():.6f}, mapped: {(failing_t[i]).item():.6f}")
            print("Fraction correct:", frac_correct)
            assert torch.all(indicator == 1), frac_correct
    return result.to(torch.float32)

def sample_non_cantor(num_samples, generation, device=torch.device("cpu"), debug=False):
    """
    Generate num_samples points in [-1,1] that do NOT lie in the Cantor set
    using rejection sampling.
    
    Points are sampled uniformly over [-1,1] (via mapping from [0,1]) and
    then filtered using `cantor_indicator` (which assumes inputs in [-1,1]).
    """
    samples = []
    while len(samples) < num_samples:
        # Oversample uniformly in [0,1] then map to [-1,1].
        batch = torch.rand((num_samples, 1), device=device)
        x = 2 * batch - 1  # x in [-1,1]
        # Points for which the indicator returns 0 are not in the Cantor set.
        indicator = cantor_indicator(x, generation=generation)
        valid = x[indicator == 0]
        if valid.numel() > 0:
            samples.append(valid)
    samples = torch.cat(samples, dim=0)
    samples = samples[:num_samples].unsqueeze(1)
    # Assert that all generated samples are NOT in the Cantor set if debug is enabled.
    if debug:
        non_cantor_indicator = cantor_indicator(samples, generation=generation)
        frac_valid = torch.mean(non_cantor_indicator==0, dtype=torch.float64).item()
        if not torch.all(non_cantor_indicator == 0):
            fail_mask = (non_cantor_indicator != 0)
            failing_results = samples[fail_mask]
            mapped_results = (failing_results + 1) / 2.0
            print("Debug [sample_non_cantor]: Found", fail_mask.sum().item(), "incorrect samples out of", num_samples)
            print("Fraction valid (should be 1.0):", frac_valid)
            assert torch.all(non_cantor_indicator == 0), frac_valid
    return samples

class VectorGELUActivation(nn.Module):
    def forward(self, x):
        # Compute the norm of x along the last dimension.
        r = torch.norm(x, p=2, dim=-1, keepdim=True)
        d = x.size(-1)
        a = torch.tensor(d / 2.0, dtype=x.dtype, device=x.device)
        # Compute the scale using the lower incomplete gamma function.
        scale = torch.special.gammainc(a, (r ** 2) / 2.0)
        self.last_scale = scale.detach()  # store the last scale for later inspection
        return x * scale

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias_real = nn.Parameter(torch.randn(out_features) * 0.1)
            self.bias_imag = nn.Parameter(torch.randn(out_features) * 0.1)
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, input):
        weight = torch.complex(self.weight_real, self.weight_imag)
        if self.bias_real is not None:
            bias = torch.complex(self.bias_real, self.bias_imag)
        else:
            bias = None
        return F.linear(input, weight, bias)

class ComplexVectorGELUActivation(nn.Module):
    def forward(self, x):
        # x is complex. Compute the norm as the sqrt(sum(|x|^2)).
        r = torch.sqrt((x.real**2 + x.imag**2).sum(dim=-1, keepdim=True))
        d = x.size(-1)
        a = torch.tensor(d / 2.0, dtype=x.real.dtype, device=x.device)
        scale = torch.special.gammainc(a, (r ** 2) / 2.0)
        self.last_scale = scale.detach()
        return x * scale

class SampledGELUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
         # Save input for backward.
         ctx.save_for_backward(input)
         # Sample a standard normal (mean=0, std=1) for each element.
         sample = torch.randn_like(input)
         # If the sample is greater than the input element, set to zero.
         indicator = (sample <= input).to(input.dtype)
         return input * indicator

    @staticmethod
    def backward(ctx, grad_output):
         (input,) = ctx.saved_tensors
         # Compute the CDF of the standard normal evaluated at input.
         normal = torch.distributions.Normal(torch.zeros_like(input), torch.ones_like(input))
         cdf = normal.cdf(input)
         # Multiply the incoming grad by the cdf.
         return grad_output * cdf

class SamplingGELUActivation(nn.Module):
    def forward(self, x):
         return SampledGELUFunction.apply(x)

class Mish(nn.Module):
    def forward(self, x):
         # Mish: x * tanh(softplus(x))
         return x * torch.tanh(F.softplus(x))

class ElementwiseGELUActivation(nn.Module):
    def __init__(self, a: float = 0.5):
        super().__init__()
        self.a = a
    def forward(self, x):
        # Compute elementwise scale using the regularized lower incomplete gamma function
        # For each element, compute scale = gammainc(a, |x|)
        a_tensor = torch.tensor(self.a, dtype=x.dtype, device=x.device)
        # torch.special.gammainc supports broadcasting and applies elementwise
        scale = torch.special.gammainc(a_tensor, torch.abs(x))
        return x * scale

class GlobalSoftmaxActivation(nn.Module):
    def forward(self, x):
        # Applies softmax over the entire activation vector.
        # For a tensor with shape (batch, features), softmax is performed along dimension 1.
        return torch.softmax(x, dim=1)

class NormalizeActivation(nn.Module):
    def forward(self, x):
        # Normalize each sample's activation vector to have L2 norm = 1.
        # For a tensor of shape (batch, features), normalization is performed along dimension 1.
        eps = 1e-6
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / (norm + eps)

class L1NormalizeActivation(nn.Module):
    def forward(self, x):
        # Normalize each sample's activation vector to have L1 norm = 1.
        # For a tensor of shape (batch, features), normalization is performed along dimension 1.
        eps = 1e-6
        norm = torch.norm(x, p=1, dim=1, keepdim=True)
        return x / (norm + eps)

class LearnedActivation(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply the learned activation elementwise.
        orig_shape = x.shape
        # Flatten all elements into separate rows.
        x_flat = x.view(-1, 1)
        out = self.fc1(x_flat)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        # Restore the original shape.
        return out.view(orig_shape)
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # Define the activation functions to test.
    activation_functions = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
        "SamplingGELU": SamplingGELUActivation,
        "Sigmoid": nn.Sigmoid,
        "LeakyReLU": nn.LeakyReLU,
        "PReLU": nn.PReLU,
        "Mish": Mish,
        "ElementwiseGELU": ElementwiseGELUActivation,
        "GlobalSoftmax": GlobalSoftmaxActivation,
        "Normalize": NormalizeActivation,
        "L1Normalize": L1NormalizeActivation,
        "LearnedActivation": LearnedActivation,
        # More can be added here, e.g., "Tanh": nn.Tanh
    }
    
    # Optionally, filter activation functions if a subset is specified.
    if args.activations:
        selected = [act.strip() for act in args.activations.split(',')]
        activation_functions = {act: fn for act, fn in activation_functions.items() if act in selected}
        if not activation_functions:
            valid_keys = ", ".join(["ReLU", "GELU", "SiLU", "SamplingGELU",
                                     "Sigmoid", "LeakyReLU", "PReLU", "Mish", "ElementwiseGELU", "GlobalSoftmax", "Normalize", "L1Normalize", "LearnedActivation"])
            raise ValueError("No valid activation functions selected! Valid options are: " + valid_keys)

    # Create a network and its optimizer for each activation function.
    nets = {}
    optimizers = {}
    # We'll also keep track of loss history and error history (optional)
    loss_history = {name: [] for name in activation_functions}
    error_history = {name: [] for name in activation_functions}

    for act_name, act_fn in activation_functions.items():
        net = build_net(args.num_layers, args.hidden_size, act_fn, use_complex=False).to(device)
        nets[act_name] = net
        optimizers[act_name] = optim.Adam(net.parameters(), lr=args.lr)
    
    # If requested, load the previously saved learned activation weights.
    if "LearnedActivation" in nets and args.load_learned_activation:
         if os.path.exists(args.learned_activation_file):
             nets["LearnedActivation"].load_state_dict(torch.load(args.learned_activation_file))
             print(f"Loaded LearnedActivation weights from {args.learned_activation_file}")
         else:
             print(f"Warning: {args.learned_activation_file} not found. Starting with random initialization for LearnedActivation.")

    # Initialize predictions history to store updates for re-drawing.
    predictions_history = {name: [] for name in nets}
    # For LearnedActivation, also keep a history of its learned activation snapshots.
    learned_activation_history = {}
    if "LearnedActivation" in nets:
        learned_activation_history["LearnedActivation"] = []

    # Setup matplotlib and create subplots.
    fig, axs = plt.subplots(len(nets), 1, figsize=(6, 4 * len(nets)))
    if len(nets) == 1:
        axs = [axs]  # ensure axs is always iterable
    os.makedirs("fit_1D_plots", exist_ok=True)

    # Plot the true function once on each subplot.
    x_plot_init = torch.linspace(args.x_range[0], args.x_range[1], args.num_plot_points).unsqueeze(1).to(device)
    y_true_init = target_function(x_plot_init, args.generation).cpu().detach().numpy()
    for ax in axs:
        ax.plot(x_plot_init.cpu().numpy(), y_true_init, label="True function", color="black", linewidth=2)

    # Define fixed colors for each activation function.
    colors = {
        "ReLU": "red",
        "GELU": "blue",
        "SiLU": "magenta",
        "SamplingGELU": "cyan",
        "Sigmoid": "darkgreen",
        "LeakyReLU": "orange",
        "PReLU": "brown",
        "Mish": "olive",
        "ElementwiseGELU": "purple",
        "GlobalSoftmax": "pink",
        "Normalize": "teal",
        "L1Normalize": "gold",
        "LearnedActivation": "navy",
    }

    # Training loop.
    profile_context = (torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available(),
                        record_shapes=True) if args.profile else nullcontext())
    with profile_context as prof:
        for step in range(args.steps):
            # Sample half the training batch from the Cantor set and half from its complement.
            half_bs = args.batch_size // 2
            x_cantor = sample_cantor_set(half_bs, generation=args.generation, device=device, debug=args.debug)
            x_non_cantor = sample_non_cantor(half_bs, generation=args.generation, device=device, debug=args.debug)
            x_train = torch.cat([x_cantor, x_non_cantor], dim=0)
            # Optionally, shuffle the training data indices.
            idx = torch.randperm(x_train.shape[0])
            x_train = x_train[idx]
            y_train = target_function(x_train, args.generation).to(device)
            
            # Update each network.
            for act_name, net in nets.items():
                optimizer = optimizers[act_name]
                optimizer.zero_grad()
                y_pred = net(x_train)
                loss = nn.MSELoss()(y_pred, y_train)
                loss.backward()
                optimizer.step()
                loss_history[act_name].append(loss.item())
            
            # Update plots periodically.
            if step % args.plot_interval == 0:
                # Create a grid of x values for plotting.
                x_plot = torch.linspace(args.x_range[0], args.x_range[1], args.num_plot_points).unsqueeze(1).to(device)
                # For each network, compute prediction and store in history.
                for act_name, net in nets.items():
                    net.eval()
                    with torch.no_grad():
                        y_pred_plot = net(x_plot).cpu().numpy()
                    predictions_history[act_name].append((step, y_pred_plot))
                    # Compute average L2 error over the interval (using target_function as ground truth)
                    y_true = target_function(x_plot, args.generation).cpu().numpy()
                    l2_error = np.sqrt(np.mean((y_pred_plot - y_true)**2))
                    error_history[act_name].append((step, l2_error))
                    net.train()

                # For the LearnedActivation network, update its historical snapshots.
                if "LearnedActivation" in nets:
                    # Extract the first occurrence of LearnedActivation from the network.
                    learned_act_instance = None
                    for module in nets["LearnedActivation"].modules():
                        if isinstance(module, LearnedActivation):
                            learned_act_instance = module
                            break
                    if learned_act_instance is not None:
                        x_act = torch.linspace(args.activation_range[0], args.activation_range[1], args.num_plot_points).unsqueeze(1).to(device)
                        with torch.no_grad():
                            y_act_current = learned_act_instance(x_act).cpu().detach().numpy()
                        learned_activation_history["LearnedActivation"].append((step, y_act_current))

                        # Save (overwrite) the current weights to file.
                        torch.save(nets["LearnedActivation"].state_dict(), args.learned_activation_file)

                # Compute the global maximum L2 error across all activations for a common y-range.
                global_max = 0
                for hist in error_history.values():
                    for (_, err) in hist:
                        if err > global_max:
                            global_max = err

                # Create a new grid of subplots:
                n_hist = len(next(iter(predictions_history.values())))  # number of historical prediction plots
                new_n_cols = n_hist + 2  # one column for L2 error plot and one for activation function
                n_rows = len(nets)
                fig_new, axs_new = plt.subplots(n_rows, new_n_cols, figsize=(3*new_n_cols, 3*n_rows), squeeze=False)

                # For each activation (each row):
                for i, act_name in enumerate(nets.keys()):
                    # --- Plot the L2 error history (in the leftmost column) ---
                    ax_err = axs_new[i, 0]
                    if error_history[act_name]:
                        steps, errors = zip(*error_history[act_name])
                        ax_err.plot(steps, errors, label="L2 Error", color="black", linewidth=2)
                    ax_err.set_ylim(0, global_max)
                    ax_err.set_title(f"{act_name} L2 Error")
                    ax_err.set_xlabel("Step")
                    ax_err.set_ylabel("L2 Error")
                    ax_err.legend(fontsize='x-small')
                    
                    # --- Plot the activation function (in the next column) ---
                    ax_act = axs_new[i, 1]
                    x_act = torch.linspace(args.activation_range[0], args.activation_range[1], args.num_plot_points).unsqueeze(1).to(device)
                    if act_name == "LearnedActivation":
                        # Extract the learned activation instance from the trained network.
                        learned_act_instance = None
                        for module in nets[act_name].modules():
                            if isinstance(module, LearnedActivation):
                                learned_act_instance = module
                                break
                        if learned_act_instance is None:
                            learned_act_instance = activation_functions[act_name]().to(device)
                        with torch.no_grad():
                            y_act = learned_act_instance(x_act).cpu().detach().numpy()
                        # Plot without any label so no legend key is shown.
                        ax_act.plot(x_act.cpu().numpy(), y_act,
                                    color=colors.get(act_name, "green"), linewidth=2)

                        # Plot historical learned activation snapshots using a faded Greys colormap (without labels).
                        import matplotlib.cm as cm
                        cmap = cm.get_cmap("Greys")
                        events = learned_activation_history.get("LearnedActivation", [])
                        n_events = len(events)
                        for idx, (s, hist_y) in enumerate(events):
                            norm_val = (idx + 1) / n_events  # older snapshots are lighter
                            color_event = cmap(norm_val)
                            ax_act.plot(x_act.cpu().numpy(), hist_y,
                                        color=color_event, linestyle="--", linewidth=1)
                    else:
                        act_instance = activation_functions[act_name]().to(device)
                        with torch.no_grad():
                            y_act = act_instance(x_act).cpu().detach().numpy()
                        ax_act.plot(x_act.cpu().numpy(), y_act, label=f"{act_name} Activation",
                                    color=colors.get(act_name, "green"), linewidth=2)

                    ax_act.axvline(x=0, color='gray', linestyle='dotted')
                    # (No y-axis constraint is set by default; remove set_ylim.)
                    
                    # --- Plot historical predictions in subsequent columns ---
                    for j, (s, y_pred) in enumerate(predictions_history[act_name]):
                        ax = axs_new[i, j+2]  # shift index by 2 columns (error plot & activation plot)
                        # Plot the true function in black.
                        y_true = target_function(x_plot, args.generation).cpu().detach().numpy()
                        ax.plot(x_plot.cpu().numpy(), y_true, label="True", color="black", linewidth=2)
                        # Plot the predicted function.
                        color = colors.get(act_name, "green")
                        ax.plot(x_plot.cpu().numpy(), y_pred, label=f"Pred {s}", color=color, linewidth=1)
                        ax.set_title(f"{act_name} (step {s})")
                        ax.legend(fontsize='x-small')

                fig_new.tight_layout()
                fig_new.savefig("fit_1D_plots/fit.png")
                plt.close(fig_new)

                # Additionally, save a separate plot for the LearnedActivation activation function.
                if "LearnedActivation" in nets:
                    fig_la, ax_la = plt.subplots(figsize=(6, 4))
                    x_act = torch.linspace(args.activation_range[0], args.activation_range[1], args.num_plot_points).unsqueeze(1).to(device)
                    learned_act_instance = None
                    for module in nets["LearnedActivation"].modules():
                        if isinstance(module, LearnedActivation):
                            learned_act_instance = module
                            break
                    if learned_act_instance is None:
                        learned_act_instance = activation_functions["LearnedActivation"]().to(device)
                    with torch.no_grad():
                        y_act = learned_act_instance(x_act).cpu().detach().numpy()
                    ax_la.plot(x_act.cpu().numpy(), y_act,
                              color=colors.get("LearnedActivation", "green"), linewidth=2)
                    import matplotlib.cm as cm
                    cmap = cm.get_cmap("Greys")
                    events = learned_activation_history.get("LearnedActivation", [])
                    n_events = len(events)
                    for idx, (s, hist_y) in enumerate(events):
                        norm_val = (idx + 1) / n_events
                        color_event = cmap(norm_val)
                        ax_la.plot(x_act.cpu().numpy(), hist_y,
                                  color=color_event, linestyle="--", linewidth=1)
                    ax_la.axvline(x=0, color='gray', linestyle='dotted')
                    # (No y-axis constraint is set by default; remove set_ylim.)
                    ax_la.set_title("LearnedActivation Activation")
                    fig_la.tight_layout()
                    fig_la.savefig("fit_1D_plots/learned_activation_plot.png")
                    plt.close(fig_la)
    if args.profile:
        print(prof.key_averages().table(sort_by="self_cuda_time_total"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare nonlinearities in neural network fits on 1D data.")
    # Total number of layers (including input and output layers)
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Total number of layers (>=1) including input and output.")
    parser.add_argument("--hidden_size", type=int, default=50,
                        help="Hidden layer size.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate.")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of training steps.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--plot_interval", type=int, default=100,
                        help="Number of steps between plot updates.")
    parser.add_argument("--num_plot_points", type=int, default=1000,
                        help="Number of points for plotting the function.")
    parser.add_argument("--x_range", type=float, nargs=2, default=[-1.0, 1.0],
                        help="Lower and upper limits for training data fitting interval.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA.")
    parser.add_argument("--use_complex", action="store_true",
                        help="Use complex layers.")
    parser.add_argument("--generation", type=int, default=3,
                        help="Fractal generation level for Cantor set (digit expansion depth).")
    parser.add_argument("--activations", type=str,
                        help="Comma-separated list of activation functions to run. Valid options: ReLU, GELU, SiLU, SamplingGELU, Sigmoid, LeakyReLU, PReLU, Mish, ElementwiseGELU, GlobalSoftmax, Normalize, L1Normalize, LearnedActivation. If not provided, all are run.")
    parser.add_argument("--profile", action="store_true",
                        help="Profile the training loop using PyTorch's autograd profiler.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug checks for sampling procedures (checks to verify that samples are correct).")
    parser.add_argument("--activation_range", type=float, nargs=2, default=[-2.0, 2.0],
                        help="Lower and upper limits for the activation plot.")
    parser.add_argument("--learned_activation_file", type=str, default="learned_activation_weights.pt",
                        help="File path to save/load the learned activation weights.")
    parser.add_argument("--load_learned_activation", action="store_true",
                        help="If provided, loads previously saved learned activation weights before training.")
    args = parser.parse_args()
    main(args) 