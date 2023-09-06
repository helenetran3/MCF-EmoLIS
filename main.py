from src.pickle_functions import pickle_file_exists, load_from_pickle
from src.dataset_utils import get_dataset_from_sdk, get_fold_ids, process_and_split_dataset, get_length_dataset
from src.hyperparam_processing import get_hyperparam_from_json
from src.csv_functions import get_current_model_id, save_model_param_in_csv_file, csv_summary
from src.label_processing import get_available_presence_scores_arr, update_labels, balance_dataset
from src.model_metrics import compute_and_save_model_metrics


def main():

    param_log = get_hyperparam_from_json("param_logistics")
    param_task = get_hyperparam_from_json("param_task")
    model_name = param_task['model_name']
    param_model = get_hyperparam_from_json("param_model", model_name=model_name)
    param_training = get_hyperparam_from_json("param_training")
    is_tf = bool(param_model['is_TensorFlow'])

    # Get data for training, validation and test sets from pickle (split provided by the SDK)
    pickle_name_fold = param_log['pickle_name_fold']
    if pickle_file_exists(pickle_name_fold + "_train", "raw_folds", "cmu_mosei"):
        train_list = load_from_pickle(pickle_name_fold + "_train", "raw_folds", "cmu_mosei")
        valid_list = load_from_pickle(pickle_name_fold + "_valid", "raw_folds", "cmu_mosei")
        test_list = load_from_pickle(pickle_name_fold + "_test",  "raw_folds", "cmu_mosei")

    else:
        # Load CMU-MOSEI dataset
        dataset = get_dataset_from_sdk(param_log, param_task)

        # Get data for training, validation and test sets (split provided by the SDK)
        train_ids, valid_ids, test_ids = get_fold_ids(param_task)
        train_list, valid_list, test_list = process_and_split_dataset(dataset, train_ids, valid_ids, test_ids,
                                                                      param_task, param_log)

    all_scores = get_available_presence_scores_arr(train_list, valid_list, test_list)
    num_steps = get_length_dataset(train_list, valid_list, test_list, param_task, is_tf)

    # Update labels so that it matches with the targeted task given by args.label_type (all, coarse, present, dominant)
    train_list, valid_list, test_list = update_labels(train_list, valid_list, test_list, param_task, all_scores)

    # Balance dataset  #TODO
    balance_dataset(train_list, valid_list, test_list, param_task, all_scores)

    # Return model id
    model_id = get_current_model_id(model_name)

    # Model training
    if is_tf:
        from src.tensorflow_utils.model_training import train_tf_model
        from src.tensorflow_utils.model_evaluation import evaluate_tf_model

        train_tf_model(train_list, valid_list, test_list,
                       param_task, param_model, param_training, param_log, num_steps, model_name, model_id)

        loss_function_val, pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
            evaluate_tf_model(test_list, param_task, param_training, param_log, num_steps, model_name, model_id,
                              all_scores)

    else:
        import torch
        from src.pytorch_utils.model_training import train_torch_model
        from src.pytorch_utils.model_evaluation import evaluate_torch_model

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"\nUsing {device} device")

        train_torch_model(train_list, valid_list, test_list,
                          param_task, param_model, param_training, param_log, num_steps, model_name, model_id, device)

        loss_function_val, pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
            evaluate_torch_model(test_list, param_task, param_model, param_training, param_log, num_steps, model_name,
                                 model_id, all_scores, device)

    # Save model parameters in csv file
    save_model_param_in_csv_file(model_id, param_task, param_model, param_training, model_name)

    # Compute and save metrics
    compute_and_save_model_metrics(test_list, loss_function_val,
                                   pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom,
                                   param_task, param_training, param_log, model_name, model_id, all_scores)

    # Summary files
    csv_summary()


if __name__ == "__main__":
    main()
