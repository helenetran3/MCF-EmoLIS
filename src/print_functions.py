import numpy as np
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
                              loss_function, val_metrics, label_type, is_tf):

    predict_neutral_class = param_task['predict_neutral_class']
    num_folds = param_training['cross_val_folds']
    thres_score_pres = param_task['thres_score_pres']

    y_train, y_valid, y_test = (train_list[0][1], valid_list[0][1], test_list[0][1]) if with_cross_valid \
        else (train_list[1], valid_list[1], test_list[1])
    
    # Compute num_classes, where predict neutral class is already inferred from label preprocessing
    one_hot = ("oh" in label_type) if type(label_type) != dict else \
        {'coarse_task': "oh" in label_type['coarse_task'], 'fine_task': "oh" in label_type['fine_task']}
    if type(label_type) == dict:
        num_classes = y_train['fine_task'][0].shape[1] if not one_hot['fine_task'] else y_train['fine_task'][0].shape[0]
        num_train_samples = len(y_train['fine_task'])
        num_valid_samples = len(y_valid['fine_task'])
        num_test_samples = len(y_test['fine_task'])
    else:
        num_classes = y_train[0].shape[1] if not one_hot else y_train[0].shape[0]
        num_train_samples = len(y_train)
        num_valid_samples = len(y_valid)
        num_test_samples = len(y_test)
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
    print("Score threshold for the presence of an emotion:", thres_score_pres)
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

def print_mean_std_cross_validation(met_reg, met_fin, met_coa, met_pres, met_dom, num_classes, decimals):
    print("\n======================== Mean and std of folds ========================\n")
    print("Number of folds:", len(met_reg))

    def str_overall(val_macro_list):
        val_mean = np.round(np.mean(val_macro_list), decimals=decimals)
        val_std = np.round(np.std(val_macro_list), decimals=decimals)
        return "{} ({})".format(np.mean(val_mean), np.std(val_std))

    def str_emo(met_per_fold):
        met_per_emo = [[met_fold_i[k] for met_fold_i in met_per_fold] for k in range(num_classes)]
        res_str = "["
        for emo_i, met_per_emo_k in enumerate(met_per_emo):
            res_str += "{} ({})".format(np.round(np.mean(met_per_emo_k), decimals=decimals),
                                        np.round(np.std(met_per_emo_k), decimals=decimals))
            if emo_i != num_classes - 1:
                res_str += ", "
            else:
                res_str += "]"
        return res_str

    def print_mean_std_regression(met_regression):
        mae = [met_i[0] for met_i in met_regression]
        mse = [met_i[1] for met_i in met_regression]
        r2 = [met_i[2] for met_i in met_regression]
        mae_per_fold = [met_i[3:3 + num_classes] for met_i in met_regression]
        mse_per_fold = [met_i[3 + num_classes:3 + 2 * num_classes] for met_i in met_regression]
        r2_per_fold = [met_i[3 + 2 * num_classes:3 + 3 * num_classes] for met_i in met_regression]
        print("Mean absolute error (for each emotion):", str_emo(mae_per_fold))
        print("Mean squared error (for each emotion):", str_emo(mse_per_fold))
        print("R2 score (for each emotion):", str_emo(r2_per_fold))
        print("Mean absolute error (overall):", str_overall(mae))
        print("Mean squared error (overall):", str_overall(mse))
        print("R2 score (overall):", str_overall(r2))

    def print_mean_std_classification_score(met_class_score):
        acc_macro = [met_i[0] for met_i in met_class_score]
        f1_macro = [met_i[1] for met_i in met_class_score]
        rec_macro = [met_i[2] for met_i in met_class_score]
        prec_macro = [met_i[3] for met_i in met_class_score]
        acc_per_fold = [met_i[4:4 + num_classes] for met_i in met_class_score]
        f1_per_fold = [met_i[4 + num_classes:4 + 2 * num_classes] for met_i in met_class_score]
        rec_per_fold = [met_i[4 + 2 * num_classes:4 + 3 * num_classes] for met_i in met_class_score]
        prec_per_fold = [met_i[4 + 3 * num_classes:4 + 4 * num_classes] for met_i in met_class_score]
        print("Accuracy (for each, unweighted across scores):", str_emo(acc_per_fold))
        print("Accuracy (unweighted mean):", str_overall(acc_macro))
        print("F1 score (for each, unweighted across scores):", str_emo(f1_per_fold))
        print("F1 score (unweighted mean):", str_overall(f1_macro))
        print("Recall (for each, unweighted across scores):", str_emo(rec_per_fold))
        print("Recall (unweighted mean):", str_overall(rec_macro))
        print("Precision (for each, unweighted across scores):", str_emo(prec_per_fold))
        print("Precision (unweighted mean):", str_overall(prec_macro))

    def print_mean_std_binary_classification(met_class_bin):
        multilabel_acc = [met_i[0] for met_i in met_class_bin]
        bin_acc_macro = [met_i[1] for met_i in met_class_bin]
        wei_acc_macro = [met_i[2] for met_i in met_class_bin]
        f1_macro = [met_i[3] for met_i in met_class_bin]
        rec_macro = [met_i[4] for met_i in met_class_bin]
        prec_macro = [met_i[5] for met_i in met_class_bin]
        roc_auc_macro = [met_i[6] for met_i in met_class_bin]
        bin_acc_per_class = [met_i[7:7 + num_classes] for met_i in met_class_bin]
        wei_acc_per_class = [met_i[7 + num_classes:7 + 2 * num_classes] for met_i in met_class_bin]
        f1_each_rounded = [met_i[7 + 2 * num_classes:7 + 3 * num_classes] for met_i in met_class_bin]
        rec_each_rounded = [met_i[7 + 3 * num_classes:7 + 4 * num_classes] for met_i in met_class_bin]
        prec_each_rounded = [met_i[7 + 4 * num_classes:7 + 5 * num_classes] for met_i in met_class_bin]
        roc_auc_each_rounded = [met_i[7 + 5 * num_classes:7 + 6 * num_classes] for met_i in met_class_bin]
        print("Multilabel accuracy:", str_overall(multilabel_acc))
        print("Binary accuracy (for each):", str_emo(bin_acc_per_class))
        print("Binary accuracy (unweighted mean):", str_overall(bin_acc_macro))
        print("Weighted accuracy (for each):", str_emo(wei_acc_per_class))
        print("Weighted accuracy (unweighted mean):", str_overall(wei_acc_macro))
        print("F1 score (for each):", str_emo(f1_each_rounded))
        print("F1 score (unweighted mean):", str_overall(f1_macro))
        print("Recall (for each):", str_emo(rec_each_rounded))
        print("Recall (unweighted mean):", str_overall(rec_macro))
        print("Precision (for each):", str_emo(prec_each_rounded))
        print("Precision (unweighted mean):", str_overall(prec_macro))
        print("ROC AUC (for each):", str_emo(roc_auc_each_rounded))
        print("ROC AUC (unweighted mean):", str_overall(roc_auc_macro))

    if not (None in met_reg):
        print("\n------- Presence score estimation (regression) --------\n")
        print_mean_std_regression(met_reg)
    if not (None in met_fin):
        print(f"\n------- Presence score classification: fine-grained (all available scores) -------\n")
        print_mean_std_classification_score(met_fin)
    if not (None in met_coa):
        print("\n------- Presence score classification: coarse-grained [0,1,2,3] -------\n")
        print_mean_std_classification_score(met_coa)
    if not (None in met_pres):
        print("\n------- Detecting the presence of emotions ------------\n")
        print_mean_std_binary_classification(met_pres[0])
    if not (None in met_dom):
        print("\n------- Predicting dominant emotions ------------------\n")
        print_mean_std_binary_classification(met_dom)
