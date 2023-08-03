from torch import nn

possible_loss_fn_torch = ['mae', 'mse']


def get_loss_function_torch(loss_function):
    if loss_function == 'mae':
        return nn.L1Loss()
    elif loss_function == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError("'{}' assigned to key 'loss_function in 'hyperparameters.json' is invalid."
                         " Please choose among {}.".format(loss_function, possible_loss_fn_torch))
