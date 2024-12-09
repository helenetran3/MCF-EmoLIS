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
                              loss_function, val_metrics, is_tf):

    predict_neutral_class = param_task['predict_neutral_class']
    label_type = param_task['label_type']
    num_folds = param_training['cross_val_folds']
    thres_score_pres = param_task['thres_score_pres']

    y_train, y_valid, y_test = (train_list[0][1], valid_list[0][1], test_list[0][1]) if with_cross_valid \
        else (train_list[1], valid_list[1], test_list[1])
    num_classes = y_train[0].shape[1]  # predict neutral class is already inferred from label preprocessing
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


def print_mean_std_cross_validation(met_reg, met_score, met_pres, met_dom, label_type, num_classes,
                                    thres_score_pres, decimals):
    print("\n======================== Mean and std of folds ========================\n")
    print("Number of folds:", len(met_reg))

    def str_overall(val_macro_list):
        val_mean = np.mean(val_macro_list)
        val_std = np.std(val_macro_list)
        return "{} ({})".format(np.round(val_mean, decimals=decimals), np.round(val_std, decimals=decimals))

    def str_emo(met_per_fold):
        met_per_emo = [[met_fold_i[k] for met_fold_i in met_per_fold] for k in range(num_classes)]
        res_str = "["
        for emo_i, met_per_emo_k in enumerate(met_per_emo):
            val_mean_i = np.mean(met_per_emo_k)
            val_std_i = np.std(met_per_emo_k)
            res_str += "{} ({})".format(np.round(val_mean_i, decimals=decimals), np.round(val_std_i, decimals=decimals))
            if emo_i != num_classes - 1:
                res_str += ", "
            else:
                res_str += "]"
        return res_str

    def print_mean_std_regression(met_regression):
        length = len(met_regression[0])
        assert(length in [5 + 5 * num_classes, 4 + 4 * num_classes])
        mae = [met_i[0] for met_i in met_regression]
        mse = [met_i[1] for met_i in met_regression]
        r2 = [met_i[2] for met_i in met_regression]
        kl = [met_i[3] for met_i in met_regression]
        start_ind = 5 if length == 5 + 5 * num_classes else 4
        mae_per_fold = [met_i[start_ind:start_ind + num_classes] for met_i in met_regression]
        mse_per_fold = [met_i[start_ind + num_classes:start_ind + 2 * num_classes] for met_i in met_regression]
        r2_per_fold = [met_i[start_ind + 2 * num_classes:start_ind + 3 * num_classes] for met_i in met_regression]
        kl_per_fold = [met_i[start_ind + 3 * num_classes:start_ind + 4 * num_classes] for met_i in met_regression]
        print("Mean absolute error (for each emotion):", str_emo(mae_per_fold))
        print("Mean absolute error (overall):", str_overall(mae))
        print("Mean squared error (for each emotion):", str_emo(mse_per_fold))
        print("Mean squared error (overall):", str_overall(mse))
        print("R2 score (for each emotion):", str_emo(r2_per_fold))
        print("R2 score (overall):", str_overall(r2))
        print("KL divergence (for each emotion):", str_emo(kl_per_fold))
        print("KL divergence (overall):", str_overall(kl))
        if length == 5 + 5 * num_classes:
            c_index = [met_i[4] for met_i in met_regression]
            c_index_per_fold = [met_i[start_ind + 4 * num_classes:start_ind + 5 * num_classes] for met_i in
                                met_regression]
            print("C-index (for each emotion):", str_emo(c_index_per_fold))
            print("C-index (unweighted mean):", str_overall(c_index))

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
        """
        Print mean and std of binary classifications.

        :param met_class_bin: list of num_folds lists of metrics (for presence and dominant emotions), or
          list of num_fold lists of num_scores lists of metrics (for score classification)
        """
        assert(len(met_class_bin[0]) == 13 + 6 * num_classes)
        multilabel_acc = [met_i[0] for met_i in met_class_bin]
        bin_acc_macro = [met_i[1] for met_i in met_class_bin]
        bin_acc_weighted = [met_i[2] for met_i in met_class_bin]
        wei_acc_macro = [met_i[3] for met_i in met_class_bin]
        wei_acc_weighted = [met_i[4] for met_i in met_class_bin]
        f1_macro = [met_i[5] for met_i in met_class_bin]
        f1_weighted = [met_i[6] for met_i in met_class_bin]
        rec_macro = [met_i[7] for met_i in met_class_bin]
        rec_weighted = [met_i[8] for met_i in met_class_bin]
        prec_macro = [met_i[9] for met_i in met_class_bin]
        prec_weighted = [met_i[10] for met_i in met_class_bin]
        jacc_macro = [met_i[11] for met_i in met_class_bin]
        jacc_weighted = [met_i[12] for met_i in met_class_bin]
        # roc_auc_macro = [met_i[13] for met_i in met_class_bin]
        # roc_auc_weighted = [met_i[14] for met_i in met_class_bin]
        bin_acc_per_class = [met_i[13:13 + num_classes] for met_i in met_class_bin]
        wei_acc_per_class = [met_i[13 + num_classes:13 + 2 * num_classes] for met_i in met_class_bin]
        f1_each_rounded = [met_i[13 + 2 * num_classes:13 + 3 * num_classes] for met_i in met_class_bin]
        rec_each_rounded = [met_i[13 + 3 * num_classes:13 + 4 * num_classes] for met_i in met_class_bin]
        prec_each_rounded = [met_i[13 + 4 * num_classes:13 + 5 * num_classes] for met_i in met_class_bin]
        jacc_each_rounded = [met_i[13 + 5 * num_classes:13 + 6 * num_classes] for met_i in met_class_bin]
        # roc_auc_each_rounded = [met_i[15 + 6 * num_classes:15 + 7 * num_classes] for met_i in met_class_bin]
        print("Multilabel accuracy:", str_overall(multilabel_acc))
        print("Binary accuracy (for each):", str_emo(bin_acc_per_class))
        print("Binary accuracy (unweighted mean):", str_overall(bin_acc_macro))
        print("Binary accuracy (weighted mean):", str_overall(bin_acc_weighted))
        print("Weighted accuracy (for each):", str_emo(wei_acc_per_class))
        print("Weighted accuracy (unweighted mean):", str_overall(wei_acc_macro))
        print("Weighted accuracy (weighted mean):", str_overall(wei_acc_weighted))
        print("F1 score (for each):", str_emo(f1_each_rounded))
        print("F1 score (unweighted mean):", str_overall(f1_macro))
        print("F1 score (weighted mean):", str_overall(f1_weighted))
        print("Recall (for each):", str_emo(rec_each_rounded))
        print("Recall (unweighted mean):", str_overall(rec_macro))
        print("Recall (weighted mean):", str_overall(rec_weighted))
        print("Precision (for each):", str_emo(prec_each_rounded))
        print("Precision (unweighted mean):", str_overall(prec_macro))
        print("Precision (weighted mean):", str_overall(prec_weighted))
        print("Jaccard score (for each):", str_emo(jacc_each_rounded))
        print("Jaccard score (unweighted mean):", str_overall(jacc_macro))
        print("Jaccard score (weighted mean):", str_overall(jacc_weighted))
        # print("ROC AUC (for each):", str_emo(roc_auc_each_rounded))
        # print("ROC AUC (unweighted mean):", str_overall(roc_auc_macro))
        # print("ROC AUC (weighted mean):", str_overall(roc_auc_weighted))

    if not (None in met_reg):
        print("\n------- Presence score estimation (regression) --------\n")
        print_mean_std_regression(met_reg)
    if not (None in met_score):
        scores = [0, 0.33, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3] if label_type == 'all' else [0, 1, 2, 3]
        print(f"\n------- Presence score classification -------\n")
        for i, sc_i in enumerate(scores):
            print("\n>>> Score threshold:", sc_i)
            print_mean_std_binary_classification([met_score_fold[i] for met_score_fold in met_score])
    if not (None in met_pres):
        print("\n------- Detecting the presence of emotions ------------\n")
        for j, thres in enumerate(thres_score_pres):
            print('\n>>> Threshold:', thres)
            print_mean_std_binary_classification([met_pres_fold[j] for met_pres_fold in met_pres])
    if not (None in met_dom):
        print("\n------- Predicting dominant emotions ------------------\n")
        print_mean_std_binary_classification(met_dom)
