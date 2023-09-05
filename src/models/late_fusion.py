from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Bidirectional, Dropout, Dense, LSTM, Average


class MModel:
    """
    Williams, J., Kleinegesse, S., Comanescu, R., & Radu, O. (2018, July). Recognizing Emotions in Video Using
    Multimodal DNN Feature Fusion. In Proceedings of Grand Challenge and Workshop on Human Multimodal Language
    (Challenge-HML) (pp. 11-19).
    """

    def __init__(self, model_id, param_model, num_features_text, num_features_image, num_features_audio, num_steps,
                 num_classes):
        """
        Build the model described in the paper (cf. README for the reference).
        Works only when the number of steps is the same for all datapoints.

        :param model_id: Model id (int)
        :param param_model: dictionary of 'architecture' hyperparameters of the model chosen (cf. hyperparameters.json)
        :param num_features_text: size of text feature vector (it should be the same at each step)
        :param num_features_image: size of image feature vector (it should be the same at each step)
        :param num_features_audio: size of audio feature vector (it should be the same at each step)
        :param num_steps: total number of timesteps
        :param num_classes: number of classes to predict
        """

        self.model_name = 'late_fusion'
        self.model_id = model_id
        self.num_features_text = num_features_text
        self.num_features_image = num_features_image
        self.num_features_audio = num_features_audio
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.num_layers = param_model['num_layers']
        self.num_nodes = param_model['num_nodes']
        self.dropout_rate = param_model['dropout_rate']
        self.final_activ = param_model['final_activ']

    def build(self):
        """
        :return: the model built
        """

        bidir_text, bidir_image, bidir_audio = None, None, None

        in_text = Input(shape=(self.num_steps, self.num_features_text), name="text")
        in_image = Input(shape=(self.num_steps, self.num_features_image), name="image")
        in_audio = Input(shape=(self.num_steps, self.num_features_audio), name="audio")

        bnorm_text = BatchNormalization()(in_text)
        bnorm_image = BatchNormalization()(in_image)
        bnorm_audio = BatchNormalization()(in_audio)

        all_layers = range(self.num_layers)
        for i in all_layers:
            return_seq = True if i != all_layers[-1] else False  # Return sequence for all except the last layer
            if i == 0:
                bidir_text = Bidirectional(LSTM(64, return_sequences=return_seq))(bnorm_text)
                bidir_image = Bidirectional(LSTM(64, return_sequences=return_seq))(bnorm_image)
                bidir_audio = Bidirectional(LSTM(64, return_sequences=return_seq))(bnorm_audio)
            else:
                bidir_text = Bidirectional(LSTM(64, return_sequences=return_seq))(bidir_text)
                bidir_image = Bidirectional(LSTM(64, return_sequences=return_seq))(bidir_image)
                bidir_audio = Bidirectional(LSTM(64, return_sequences=return_seq))(bidir_audio)
            bidir_text = Dropout(self.dropout_rate)(bidir_text)
            bidir_image = Dropout(self.dropout_rate)(bidir_image)
            bidir_audio = Dropout(self.dropout_rate)(bidir_audio)

        d1_text = Dense(self.num_nodes, activation="relu")(bidir_text)
        d1_image = Dense(self.num_nodes, activation="relu")(bidir_image)
        d1_audio = Dense(self.num_nodes, activation="relu")(bidir_audio)
        d1bis_text = Dropout(self.dropout_rate)(d1_text)
        d1bis_image = Dropout(self.dropout_rate)(d1_image)
        d1bis_audio = Dropout(self.dropout_rate)(d1_audio)
        out_text = Dense(self.num_classes, activation=self.final_activ)(d1bis_text)
        out_image = Dense(self.num_classes, activation=self.final_activ)(d1bis_image)
        out_audio = Dense(self.num_classes, activation=self.final_activ)(d1bis_audio)
        out = Average()([out_text, out_image, out_audio])

        model = Model(inputs=[in_text, in_image, in_audio], outputs=out)

        return model
