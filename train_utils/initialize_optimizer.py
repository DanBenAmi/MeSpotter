import torch.optim as optim


def initialize_optimizer(optimizer_name, parameters, optimizer_args):
    """
    Dynamically selects and initializes a PyTorch optimizer.

    Args:
        optimizer_name (str): Name of the optimizer (e.g., 'Adam', 'SGD').
        parameters (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        optimizer_args (dict): Arguments specific to the optimizer.

    Returns:
        An instance of the requested optimizer initialized with the provided parameters and arguments.
    """
    optimizer_class = getattr(optim, optimizer_name, None)
    if optimizer_class is None:
        raise ValueError(f"Optimizer {optimizer_name} not found in torch.optim")

    optimizer = optimizer_class(parameters, **optimizer_args)
    return optimizer

if __name__=="main__":
    # Example usage:
    model_parameters = model.parameters()  # Assuming 'model' is your PyTorch model
    optimizer_name = 'Adam'
    optimizer_args = {'lr': 0.001, 'weight_decay': 0.0001}

    optimizer = initialize_optimizer(optimizer_name, model_parameters, optimizer_args)
