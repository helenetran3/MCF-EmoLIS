import os.path

from src.pickle_functions import save_with_pickle
from src.tensorflow_utils.dataset_loader import DatasetTf
from src.hyperparam_processing import get_model_module, get_loss_function, get_optimizer

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

from matplotlib import pyplot as plt


def create_model_folder_and_path(model_name, model_id):
    """
    Create model folder and path to file containing model weights.

    :param model_name: Name of the model
    :param model_id: Model id (int)
    :return: model_save_path
    """

    model_folder = os.path.join('results', model_name, 'models')
    if not os.path.isdir('results'):
        os.makedirs(model_folder)
    model_save_path = os.path.join(model_folder, "model_{}.h5".format(model_id))

    return model_save_path


def plot_history(history, model_id, history_folder, root_folder, display_fig):
    """
    Plot history, display and save figures

    :param history: TensorFlow history object
    :param model_id: Model id (int)
    :param history_folder: Name of the folder where the history plot will be saved
    :param root_folder: Name of the root folder where the history folder is created
    :param display_fig: Whether we display the figures
    """

    # Create full path folder
    full_path_folder = os.path.join(root_folder, history_folder)
    if not os.path.isdir(full_path_folder):
        os.makedirs(full_path_folder)

    # summarize history for training and validation metric used to monitor
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    history_plot_path = os.path.join(full_path_folder, 'history_{}_loss.png'.format(model_id))
    plt.savefig(history_plot_path, bbox_inches='tight')
    if display_fig:
        plt.show()


def train_tf_model(train_list, valid_list, test_list, param_archi, param_training, num_steps, model_name, model_id,
                   predict_neutral_class, display_fig, label_type):
    """
    Train the model using TensorFlow.
    Note that to use batches in TensorFlow, time sequences are padded to the max length in the training/validation sets.

    :param train_list: [x_train, y_train, seg_train] where x_train is a list of 3 lists of arrays of shape
    (num_steps, num_features), y_train a list arrays of shape (1, 6 or 7), and seg_train a list of segment ids
    (ex: 'zk2jTlAtvSU[1]')
    :param valid_list: [x_valid, y_valid, seg_valid]
    :param test_list: [x_test, y_test, seg_test]
    :param param_archi: dictionary of 'architecture' hyperparameters of the model chosen (cf. hyperparameters.json)
    :param param_training: dictionary of 'training' hyperparameters of the model chosen (cf. hyperparameters.json)
    :param num_steps: Fixed length for all sequences in the dataset
    :param model_name: Name of the model currently tested
    :param predict_neutral_class: Whether we predict the neutral class
    :param model_id: Model id (int)
    :param display_fig: Whether we display the figures
    :param label_type: Type of labels used for model training and evaluation.
    """

    num_epochs = param_training['num_epochs']
    patience = param_training['patience']
    batch_size = param_training['batch_size']
    optimizer_name = param_training['optimizer']
    loss_function_str = param_training['loss_function']
    learning_rate = param_training['learning_rate']

    x_train = train_list[0]  # list of lists of 3 arrays
    y_train = train_list[1]
    seg_train = train_list[2]
    x_valid = valid_list[0]  # list of lists of 3 arrays
    y_valid = valid_list[1]
    seg_valid = valid_list[2]
    num_classes = y_train[0].shape[1]  # predict neutral class is already inferred from label preprocessing
    num_train_samples = len(y_train)
    num_valid_samples = len(y_valid)
    num_test_samples = len(test_list[1])
    total_data = num_train_samples + num_valid_samples + num_test_samples

    # Create TensorFlow datasets for model training
    train_dataset = DatasetTf(x_train, y_train, seg_train, num_steps, num_classes, batch_size, train_mode=True).build()
    valid_dataset = DatasetTf(x_valid, y_valid, seg_valid, num_steps, num_classes, batch_size, train_mode=True).build()

    # Parameters to save model
    model_save_path = create_model_folder_and_path(model_name, model_id)

    # Initialize callbacks
    checkpoint = ModelCheckpoint(filepath=model_save_path,
                                 verbose=1,
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min',
                                   restore_best_weights=True)

    # Build model
    num_features_text = x_train[0][0].shape[1]
    num_features_image = x_train[0][1].shape[1]
    num_features_audio = x_train[0][2].shape[1]
    loss_function = get_loss_function(loss_function_str, is_tf=True)
    optimizer_tf = get_optimizer(optimizer_name, learning_rate, is_tf=True)
    model_module = get_model_module(model_name, is_tf=True)
    model_class = model_module.MModel(model_id, param_archi, num_features_text, num_features_image, num_features_audio,
                                      num_steps, num_classes)
    model_tf = model_class.build()

    print("\n\n============================== Training Parameters ===========================================")
    print("\n>>> Dataset")
    print("Number training datapoints: {} ({:.2f}%)".format(num_train_samples, 100 * (num_train_samples / total_data)))
    print(
        "Number validation datapoints: {} ({:.2f}%)".format(num_valid_samples, 100 * (num_valid_samples / total_data)))
    print("Number test datapoints: {} ({:.2f}%)".format(num_test_samples, 100 * (num_test_samples / total_data)))
    print("Number of output cells:", num_classes)
    print("\n>>> Model training")
    print("Task:", label_type)
    if label_type == 'dominant' or label_type == 'present':
        print("Predict neutral class:", predict_neutral_class)
    print("Library: TensorFlow")
    print("Batch size:", batch_size)
    print("Number epochs:", num_epochs)
    print("Patience:", patience)
    print("Learning rate:", learning_rate)
    print("Optimizer:", optimizer_name)
    print("Loss function:", loss_function_str)
    print("\n>>> Model parameters")
    model_class.print_values()
    print("\n")
    print(model_tf.summary())

    print("\n\n================================= Model Training =============================================\n")
    model_tf.compile(loss=loss_function, optimizer=optimizer_tf)
    history = model_tf.fit(x=train_dataset,
                           epochs=num_epochs,
                           verbose=1,
                           steps_per_epoch=num_train_samples // batch_size,
                           validation_data=valid_dataset,
                           validation_steps=num_valid_samples // batch_size,
                           callbacks=[checkpoint, early_stopping])

    model_folder = os.path.join("results", model_name)
    save_with_pickle(history, "history_{}".format(model_id), pickle_folder="history", root_folder=model_folder)

    # Save model plot
    plot_model(model_tf, to_file=os.path.join(model_folder, "models", "model_{}.png".format(model_id)),
               show_shapes=True, show_layer_activations=True)

    # Display and save history plot
    plot_history(history, model_id, history_folder="history", root_folder=model_folder, display_fig=display_fig)
