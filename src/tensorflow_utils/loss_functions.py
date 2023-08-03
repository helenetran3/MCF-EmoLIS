possible_loss_fn_tf = ['mae', 'mse']


def get_loss_function_tf(loss_function):
    if loss_function == 'mae':
        return 'mean_absolute_error'
    elif loss_function == 'mse':
        return 'mean_squared_error'
    else:
        raise ValueError("'{}' assigned to key 'loss_function in 'hyperparameters.json' is invalid."
                         " Please choose among {}.".format(loss_function, possible_loss_fn_tf))
