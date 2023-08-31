from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, BatchNormalization, Bidirectional, Dropout, Dense, LSTM


class MModel:
    """
    Williams, J., Kleinegesse, S., Comanescu, R., & Radu, O. (2018, July). Recognizing Emotions in Video Using
    Multimodal DNN Feature Fusion. In Proceedings of Grand Challenge and Workshop on Human Multimodal Language
    (Challenge-HML) (pp. 11-19).
    """

    def __init__(self, model_id, param_archi, num_features_text, num_features_image, num_features_audio, num_steps,
                 num_classes):
        """
        Build the model described in the paper (cf. README for the reference).
        Works only when the number of steps is the same for all datapoints.

        :param model_id: Model id (int)
        :param param_archi: dictionary of 'architecture' hyperparameters of the model chosen (cf. hyperparameters.json)
        :param num_features_text: size of text feature vector (it should be the same at each step)
        :param num_features_image: size of image feature vector (it should be the same at each step)
        :param num_features_audio: size of audio feature vector (it should be the same at each step)
        :param num_steps: total number of timesteps
        :param num_classes: number of classes to predict
        """

        self.model_name = 'ef_williams'
        self.model_id = model_id
        self.num_features_text = num_features_text
        self.num_features_image = num_features_image
        self.num_features_audio = num_features_audio
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.num_layers = param_archi['num_layers']
        self.num_nodes = param_archi['num_nodes']
        self.dropout_rate = param_archi['dropout_rate']
        self.final_activ = param_archi['final_activ']

    def build(self):
        """
        :return: the model built
        """

        bidir = None

        in_text = Input(shape=(self.num_steps, self.num_features_text), name="text")
        in_image = Input(shape=(self.num_steps, self.num_features_image), name="image")
        in_audio = Input(shape=(self.num_steps, self.num_features_audio), name="audio")

        concat = Concatenate(axis=2)([in_text, in_image, in_audio])
        bnorm = BatchNormalization()(concat)

        all_layers = range(self.num_layers)
        for i in all_layers:
            return_seq = True if i != all_layers[-1] else False  # Return sequence for all except the last layer
            if i == 0:
                bidir = Bidirectional(LSTM(64, return_sequences=return_seq))(bnorm)
            else:
                bidir = Bidirectional(LSTM(64, return_sequences=return_seq))(bidir)
            bidir = Dropout(self.dropout_rate)(bidir)

        d1 = Dense(self.num_nodes, activation="relu")(bidir)
        d1bis = Dropout(self.dropout_rate)(d1)
        out = Dense(self.num_classes, activation=self.final_activ)(d1bis)

        model = Model(inputs=[in_text, in_image, in_audio], outputs=out)

        return model


def get_data_for_csv(model_name, param_archi, num_classes):
    """
    Get header and data to save model hyperparameters in csv file (ie. number of layers, nodes, dropout rate).

    :param model_name: Model name (string)
    :param param_archi: dictionary of 'architecture' hyperparameters of the model chosen (cf. hyperparameters.json)
    :param num_classes: number of emotion classes (7 with neutral class, else 6)
    :return: two lists of strings (one for header, one for data) to save in csv
    """

    data = [model_name, param_archi['num_layers'], param_archi['num_nodes'], param_archi['dropout_rate'],
            param_archi['final_activ'], num_classes]
    header = ['model_name', 'num_layers', 'num_nodes', 'dropout_rate', 'final_activ', 'num_classes']

    return header, data
