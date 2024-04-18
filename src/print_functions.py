from src.hyperparam_processing import get_hyperparam_from_json, get_model_module


def print_hyperparameters(model_name, model_id, param_value_to_print, print_archi, loss_function=None, val_metrics=None):
    if print_archi:
        print("Model name:", model_name)
        print("Model id:", model_id)
        param_model_def, param_training_def = get_hyperparam_from_json("param_definitions", model_name=model_name)
        param_def_to_print = param_model_def if print_archi else param_training_def
        if param_def_to_print is not None:
            for key, value in param_def_to_print.items():
                print("{}: {}".format(param_def_to_print[key], param_value_to_print[key]))
    else:
        for key, value in param_value_to_print.items():
            if key == 'loss_function':
                print("loss_function: {}".format(loss_function))
            elif key == 'val_metrics':
                print("val_metrics: {}".format(val_metrics))
            else:
                print("{}: {}".format(key, value))


def print_training_parameters(train_list, valid_list, test_list, with_cross_valid,
                              param_task, param_model, param_training, num_steps, model_name, model_id,
                              loss_function, val_metrics, is_tf):

    predict_neutral_class = param_task['predict_neutral_class']
    label_type = param_task['label_type']
    num_folds = param_training['cross_val_folds']

    y_train, y_valid, y_test = (train_list[0][1], valid_list[0][1], test_list[0][1]) if with_cross_valid \
        else (train_list[1], valid_list[1], test_list[1])
    num_classes = y_train[0].shape[1]  # predict neutral class is already inferred from label preprocessing
    num_train_samples = len(y_train)
    num_valid_samples = len(y_valid)
    num_test_samples = len(test_list[1])
    total_data = num_train_samples + num_valid_samples + num_test_samples

    print("\n\n============================== Training Parameters ===========================================")
    print("\n>>> Dataset")
    print("Number training datapoints: {} ({:.2f}%)".format(num_train_samples, 100 * (num_train_samples / total_data)))
    print(
        "Number validation datapoints: {} ({:.2f}%)".format(num_valid_samples, 100 * (num_valid_samples / total_data)))
    print("Number test datapoints: {} ({:.2f}%)".format(num_test_samples, 100 * (num_test_samples / total_data)))
    print("Number of output cells:", num_classes)
    print("\n>>> Model training")
    print("Task:", label_type)
    print("Cross-validation:", with_cross_valid, "({} folds)".format(num_folds) if with_cross_valid else "")
    if label_type == 'dominant' or label_type == 'present':
        print("Predict neutral class:", predict_neutral_class)
    print("Library: {}".format("TensorFlow" if is_tf else "PyTorch"))
    print_hyperparameters(model_name, model_id, param_training, print_archi=False, loss_function=loss_function,
                          val_metrics=val_metrics)
    print("Fixed number of steps:", num_steps)
    print("\n>>> Model parameters")
    print_hyperparameters(model_name, model_id, param_model, print_archi=True)
    if is_tf:
        x_train = train_list[0][0] if with_cross_valid else train_list[0]
        num_features_text = x_train[0][0].shape[1]
        num_features_image = x_train[0][1].shape[1]
        num_features_audio = x_train[0][2].shape[1]
        model_module = get_model_module(model_name, is_tf=True)
        model_tf = model_module.MModel(model_id, param_model, num_features_text, num_features_image,
                                       num_features_audio, num_steps, num_classes).build()
        print(model_tf.summary())
    print("\n")

    ## Print model for Pytorch
    # if num_steps is not None:
    #     print("\n>>> Model summary (for a sequence of length {})".format(num_steps))
    #     summary(model, [(num_steps, 300), (num_steps, 35), (num_steps, 74)], device='cpu',
    #             dtypes=[torch.bfloat16, torch.bfloat16, torch.bfloat16])
    # else:
    #     print("\n>>> Model summary (for a sequence of length 26)")
    #     summary(model, [(26, 300), (26, 35), (26, 74)], device='cpu',
    #             dtypes=[torch.bfloat16, torch.bfloat16, torch.bfloat16])