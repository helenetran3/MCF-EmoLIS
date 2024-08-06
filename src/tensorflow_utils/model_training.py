import os.path

from src.pickle_functions import save_with_pickle
from src.tensorflow_utils.dataset_loader import DatasetTf
from src.hyperparam_processing import get_model_module, get_loss_function, get_val_metrics, get_optimizer
from src.print_functions import print_training_parameters

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

from matplotlib import pyplot as plt


def create_model_folder_and_path(model_name, ext_name):
    """
    Create model folder and path to file containing model weights.

    :param model_name: Name of the model
    :param ext_name: Extension name (str)
    :return: model_save_path
    """

    model_folder = os.path.join('results', model_name, 'models')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_save_path = os.path.join(model_folder, "model_{}.h5".format(ext_name))

    return model_save_path


def save_and_plot_history(history, ext_name, history_folder, root_folder, display_fig):
    """
    Save and plot history, display and save figures

    :param history: TensorFlow history object
    :param ext_name: Extension name (str)
    :param history_folder: Name of the folder where the history plot will be saved
    :param root_folder: Name of the root folder where the history folder is created
    :param display_fig: Whether we display the figures
    """

    # Create full path folder
    full_path_folder = os.path.join(root_folder, history_folder)
    if not os.path.isdir(full_path_folder):
        os.makedirs(full_path_folder)

    # save history
    save_with_pickle(history, "history_{}".format(ext_name), pickle_folder="history", root_folder=root_folder)

    # summarize history for training and validation metric used to monitor
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    history_plot_path = os.path.join(full_path_folder, 'history_{}_loss.png'.format(ext_name))
    plt.savefig(history_plot_path, bbox_inches='tight')
    if display_fig:
        plt.show()


def run_one_training(train_list, valid_list, param_model, param_training, param_log,
                     num_steps, model_name, model_id, loss_function_str, val_metrics_str, with_cross_val, fold_id=None):

    batch_size = param_training['batch_size']
    optimizer_name = param_training['optimizer']
    learning_rate = param_training['learning_rate']
    num_epochs = param_training['num_epochs']
    patience = param_training['patience']
    display_fig = param_log['display_fig']
    ext_model_name = "{}_fold_{}".format(model_id, fold_id) if with_cross_val else model_id

    x_train = train_list[0]  # list of lists of 3 arrays
    y_train = train_list[1]
    seg_train = train_list[2]
    x_valid = valid_list[0]  # list of lists of 3 arrays
    y_valid = valid_list[1]
    seg_valid = valid_list[2]
    num_classes = y_train[0].shape[1]  # predict neutral class is already inferred from label preprocessing
    num_train_samples = len(y_train)
    num_valid_samples = len(y_valid)

    # Create TensorFlow datasets for model training
    train_dataset = DatasetTf(x_train, y_train, seg_train, num_steps, num_classes, batch_size, train_mode=True).build()
    valid_dataset = DatasetTf(x_valid, y_valid, seg_valid, num_steps, num_classes, batch_size, train_mode=True).build()

    # Initialize callbacks
    checkpoint = ModelCheckpoint(filepath=create_model_folder_and_path(model_name, ext_model_name),
                                 verbose=1,
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0.0005,
                                   patience=patience,
                                   verbose=1,
                                   mode='min',
                                   restore_best_weights=True)

    callbacks = [checkpoint] if patience == -1 else [checkpoint, early_stopping]

    # Build model
    num_features_text = x_train[0][0].shape[1]
    num_features_image = x_train[0][1].shape[1]
    num_features_audio = x_train[0][2].shape[1]
    loss_function = get_loss_function(loss_function_str, param_training, is_tf=True)
    val_metrics = get_val_metrics(val_metrics_str, param_training, is_tf=True)
    optimizer_tf = get_optimizer(optimizer_name, learning_rate, is_tf=True)
    model_module = get_model_module(model_name, is_tf=True)
    model_class = model_module.MModel(model_id, param_model, num_features_text, num_features_image, num_features_audio,
                                      num_steps, num_classes)
    model_tf = model_class.build()

    fold_text = ": Fold {}".format(fold_id) if with_cross_val else ""
    print("================================= Model Training{} =============================================\n"
          .format(fold_text))

    model_tf.compile(loss=loss_function,
                     metrics=val_metrics,
                     optimizer=optimizer_tf, run_eagerly=True)

    history = model_tf.fit(x=train_dataset,
                           epochs=num_epochs,
                           verbose=1,
                           steps_per_epoch=num_train_samples // batch_size,
                           validation_data=valid_dataset,
                           validation_steps=num_valid_samples // batch_size,
                           callbacks=callbacks)

    model_folder = os.path.join("results", model_name)

    # Save model plot (only one time, hence the if condition)
    if fold_id is None or fold_id == 0:
        plot_model(model_tf, to_file=os.path.join(model_folder, "models", "model_{}.png".format(ext_model_name)),
                   show_shapes=True, show_layer_activations=True)

    # Save history, display and save history plot
    save_and_plot_history(history, ext_model_name, history_folder="history", root_folder=model_folder,
                          display_fig=display_fig)


def train_tf_model(train_list, valid_list, test_list, param_task, param_model, param_training, param_log, num_steps,
                   model_name, model_id, loss_function, val_metrics, with_cross_valid):
    """
    Train the model using TensorFlow.
    Note that to use batches in TensorFlow, time sequences are padded to the max length in the training/validation sets.

    :param train_list: [x_train, y_train, seg_train] where x_train is a list of 3 lists of arrays of shape
    (num_steps, num_features), y_train a list arrays of shape (1, 6 or 7), and seg_train a list of segment ids
    (ex: 'zk2jTlAtvSU[1]')
    :param valid_list: [x_valid, y_valid, seg_valid]
    :param test_list: [x_test, y_test, seg_test]
    :param param_task: dictionary of 'task' hyperparameters (cf. param_training_task.json)
    :param param_model: dictionary of 'architecture' hyperparameters of the model chosen (cf. param_model.json)
    :param param_training: dictionary of 'training' hyperparameters (cf. param_training_task.json)
    :param param_log: dictionary of 'logistics' hyperparameters (cf. param_logistics.json)
    :param num_steps: Fixed length for all sequences in the dataset
    :param model_name: Name of the model currently tested
    :param model_id: Model id (int)
    :param loss_function: Name of the loss function (str)
    :param val_metrics: Name of the validation metric used for monitoring (str)
    :param with_cross_valid: boolean (cross_validation or not)
    """

    print_training_parameters(train_list, valid_list, test_list, with_cross_valid,
                              param_task, param_model, param_training, num_steps, model_name, model_id,
                              loss_function, val_metrics, is_tf=True)

    if with_cross_valid:
        for fold_id, (train_list_i, valid_list_i) in enumerate(zip(train_list, valid_list)):
            run_one_training(train_list_i, valid_list_i, param_model, param_training, param_log,
                             num_steps, model_name, model_id, loss_function, val_metrics, with_cross_valid, fold_id)

    else:
        run_one_training(train_list, valid_list, param_model, param_training, param_log,
                         num_steps, model_name, model_id, loss_function, val_metrics, with_cross_valid)

