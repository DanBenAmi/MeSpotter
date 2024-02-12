import torch.optim.lr_scheduler as lr_scheduler

def initialize_lr_scheduler(scheduler_name, optimizer, scheduler_args):
    """
    Dynamically selects and initializes a PyTorch LR scheduler.

    Args:
        scheduler_name (str): Name of the LR scheduler (e.g., 'StepLR', 'ExponentialLR').
        optimizer (Optimizer): Optimizer instance for which the scheduler will be applied.
        scheduler_args (dict): Arguments specific to the LR scheduler.

    Returns:
        An instance of the requested LR scheduler initialized with the provided optimizer and arguments.
    """
    scheduler_class = getattr(lr_scheduler, scheduler_name, None)
    if scheduler_class is None:
        raise ValueError(f"LR Scheduler {scheduler_name} not found in torch.optim.lr_scheduler")

    scheduler = scheduler_class(optimizer, **scheduler_args)
    return scheduler

if __name__=="main__":
    # Example usage:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Assuming 'model' is your PyTorch model
    scheduler_name = 'StepLR'
    scheduler_args = {'step_size': 30, 'gamma': 0.1}

    scheduler = initialize_lr_scheduler(scheduler_name, optimizer, scheduler_args)
