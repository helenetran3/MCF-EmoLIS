# SOTA models for Multimodal Emotion Recognition

:dart: This repository is intended to experiment state-of-the-art codes on Multimodal Emotion Recognition.

Feel free to open an issue on this repository if you have any question or remark. Thank you for your interest!


### README Structure

1. [CMU-MOSEI Database](#CMU-MOSEI-Database)
2. [Environment](#Environment)
3. [How to run the code](#How-to-run-the-code)
4. [Output generated](#outputs-generated)
5. [Data preprocessing and resulting statistics](#Data-preprocessing-and-resulting-statistics)
6. [Models available](#models-available)
7. [How to add a new model](#how-to-add-a-new-model)

## CMU-MOSEI Database

The CMU - *Multimodal Opinion Sentiment and Emotion Intensity* database is one of the key databases in Multimodal Emotion
Recognition.

  <p font="italic" align="center">  

  <img src="images/cmu_mosei.png" align="center" alt="cmu_mosei_database">

  </p>

  <p font="italic" align="center">
  Source: http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/
  </p>

### General description

- Modalities: Audio, Video, Text (transcript)
- Emotions: Anger, Disgust, Fear, Happiness, Sadness, Surprise + Neutral
- 23 453 YouTube videos covering 250 topics
- 1000 speakers facing the camera
- Speech language: English
- Wild emotions

```commandline
@inproceedings{Zadeh2018,
  title={{Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph}},
  author={Zadeh, AmirAli Bagher and Liang, Paul Pu and Vanbriesen, Jonathan and Poria, Soujanya and Tong, Edmund and Cambria, Erik and Chen, Minghai and Morency, Louis-Philippe},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics},
  pages={2236--2246},
  year={2018}
}
```

### SDK for data retrieval

*UPDATE: The link provided below does not exist anymore. The CMU MultiComp Lab provides a new link here : https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.
We did not test yet if this code is able to retrieve the data from this repository, and if the data structure stays the
same as the previous one. These will be tested later, and the README will be updated once they are verified.*

-----------

The authors provide an SDK that can be found here: [CMU-Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK). 

Some useful details on the [CMU-Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) used to retrieve CMU-MOSEI data:
- The handcrafted features provided by the SDK are **OpenFace 2** (image), **FACET 4.2** (image), **COVAREP** (audio), 
**glove_vectors** (text).
- The labels provided for each segment are: **[sentiment, happy, sad, anger, surprise, disgust, fear]** in this order,
with sentiment in range [-3,3] and the six emotions in range [0,3]. In this repository, we only focus on emotions.

[Go to top](#sota-models-for-multimodal-emotion-recognition) :arrow_up:

## Environment
- Python 3.9.7
- TensorFlow 2.8
- Keras 2.8
- torch 1.13.1
- torchvision 0.14.1
- torchaudio 0.13.1
- TensorBoard 2.8.0
- NumPy 1.20.3
- pydot 1.4.2
- graphviz 0.20.1
- alive-progress 3.0.1

[Go to top](#sota-models-for-multimodal-emotion-recognition) :arrow_up:

## How to run the code
1. Clone [CMU MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) and follow the installation steps outlined there.
2. Two ways to run the code:

   1. **Using bash script** (recommended, parameters can be set by editing the file):
   
    ```commandline
    ./run_cluster.sh
    ```
    

   2. **Running main.py in the command line:**

    ```commandline
     usage: main.py [-h] [-pnd PICKLE_NAME_DATASET] [-pnf PICKLE_NAME_FOLD] [-t] [-al]
               [-f {facet,openface}] [-c] [-mn {ef_williams,ef_williams_debug,mult}]
               [-lt {all,coarse,present,dominant}] [-nc]
               [-tp THRESHOLD_EMO_PRESENT [THRESHOLD_EMO_PRESENT ...]] [-rd ROUND_DECIMALS]
               [-sp] [-pcm] [-df]

    SOTA Multimodal Emotion Recognition models using CMU-MOSEI database.
    
    optional arguments:
      -h, --help            show this help message and exit
      -pnd PICKLE_NAME_DATASET, --pickle_name_dataset PICKLE_NAME_DATASET
                            Name of the pickle object that will contain the CMU-MOSEI mmdataset.
      -pnf PICKLE_NAME_FOLD, --pickle_name_fold PICKLE_NAME_FOLD
                            Name of the pickle object that will contain the training, validation
                            and test folds.
      -t, --align_to_text   Data will be aligned to the textual modality.
      -al, --append_label_to_data
                            Append annotations to the dataset.
      -f {facet,openface}, --image_feature {facet,openface}
                            Image features. Values: facet or openface.
      -c, --with_custom_split
                            Perform custom split on training and validation sets (for more
                            details, cf. Williams et al. (2018) paper).
      -mn {ef_williams,ef_williams_debug,mult}, --model_name {ef_williams,ef_williams_debug,mult}
                            Name of the model currently tested. Values: ['ef_williams',
                            'ef_williams_debug', 'mult'].
      -lt {all,coarse,present,dominant}, --label_type {all,coarse,present,dominant}
                            Type of labels used for model training and evaluation. Possible
                            values: all (for all presence scores provided by the dataset), coarse
                            (presence score for each emotion among [0, 1, 2, 3]), present (multi-
                            label classification), dominant (multi-class classification)
      -nc, --predict_neutral_class
                            Predict neutral class.
      -tp THRESHOLD_EMO_PRESENT [THRESHOLD_EMO_PRESENT ...], --threshold_emo_present THRESHOLD_EMO_PRESENT [THRESHOLD_EMO_PRESENT ...]
                            Threshold at which emotions are considered to be present. Values must
                            be between 0 and 3. Note that setting thresholds greater than 0 might
                            lead to no positive true and predicted classes and skew
                            classification metrics (F1, precision, recall).
      -rd ROUND_DECIMALS, --round_decimals ROUND_DECIMALS
                            Number of decimals to be rounded for metrics.
      -sp, --save_predictions
                            Save predictions with pickle
      -pcm, --plot_confusion_matrix
                            Plot confusion matrix
      -df, --display_fig    Whether we display the figures
    ```

### Useful parameters to set
`--label_type {all,coarse,present,dominant}`: Choose the task the model has to perform for model training and evaluation.
- **all** is to estimate the presence score provided by the dataset [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
for each emotion
- **coarse** is to estimate the presence score among [0, 1, 2, 3] for each emotion
- **present** is to classify each emotion as present (1) or not present (0) in the sample
- **dominant** is to assign a single emotion (the dominant one) to the sample

`--predict_neutral_class`: Whether we predict neutral class 


### How to use Tensorboard

```commandline
tensorboard --logdir=runs
```

It will show the localhost address where you can see the model plots, the training graphs and some evaluation metrics. 

[Go to top](#sota-models-for-multimodal-emotion-recognition) :arrow_up:

## Outputs generated

All words in curly brackets (except `id`) are parameters that can be set (cf. previous section). 
`id` is the model identifier: its parameter is shown in `model_ids.csv` file (cf. fourth row of the table).

Some pickle names have an extension: 
- *_emo*: annotations only limited to the six emotions (sentiment excluded)
- *_with_n*: annotations include the neutral class

| Objects                        | Files                                                                                                                                                  | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Folder                                         |
|--------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|
| Dataset                        | `{pickle_name_dataset}.pkl`                                                                                                                            | mmdataset object obtained from CMU-MOSEI SDK                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | cmu_mosei/<br>mmdataset/                       |
| Fold                           | `{pickle_name_fold}_train.pkl` <br> `{pickle_name_fold}_valid.pkl` <br> `{pickle_name_fold}_test.pkl`                                                  | Lists of 3 elements for training, validation, and test sets respectively: <br> - **features (x)**: list of arrays of shape (number steps, number features) for text/image/audio features (concatenated in this order) <br> - **labels (y)**: list of arrays of shape (1, 7) for the 7 emotions <br> - **segment ids (seg)**: list of ids of the segment described by (x, y). Example: 'zk2jTlAtvSU[1]'                                                                                                                                                                                                                  | cmu_mosei/<br>raw_folds/                       |
| Model ids                      | `model_ids.csv`                                                                                                                                        | List of ids for each model, with the model architecture and training parameters. Useful for identifying models when saving metrics in csv files (cf. **Metrics** row)                                                                                                                                                                                                                                                                                                                                                                                                                                                   | results/<br>*model_name*/csv/                  |
| Models                         | `model_{id}.h5`                                                                                                                                        | Best model with a given set of hyperparameters                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | results/<br>*model_name/*<br>models/           |
| History                        | `history_{id}.pkl`                                                                                                                                     | Model training history                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | results/<br>*model_name/*<br>history/          |
| True labels from test set      | `true_scores_all.pkl` <br> `true_scores_coa.pkl` <br> `true_classes_pres.pkl` <br> `true_classes_dom.pkl`                                              | All are arrays of shape (test size, number classes) with number of classes equal to 7 with the neutral class, or 6 without. <br><br> Represent the true labels with different granularity: <br> **scores_all**: presence scores provided by the annotations, with values among [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3] <br> **scores_coa**: coarse presence scores with values among [0, 1, 2, 3] <br> **classes_pres**: binary array detecting the presence of an emotion (presence score > 0) <br> **classes_dom**: binary array identifying the dominant emotion(s) (the highest presence score) | cmu_mosei/<br>processed_folds/                 |
| Predicted labels from test set | `pred_raw_{id}.pkl` <br> `pred_scores_all_{id}.pkl` <br> `pred_scores_coa_{id}.pkl` <br> `pred_classes_pres_{id}.pkl` <br> `pred_classes_dom_{id}.pkl` | Files generated with `--save_predictions` argument. <br><br> The predicted presence scores or class. **pred_raw** gives the raw presence scores predicted by the model. For the rest, cf. description above done for "True labels from test set".                                                                                                                                                                                                                                                                                                                                                                       | results/<br>*model_name/*<br>predictions/      |
| Confusion matrix               | `conf_matrix_{id}`<br>`_t_{thres}.pkl`                                                                                                                 | File generated with `--plot_confusion_matrix` argument.<br><br> Multilabel confusion matrix based on the presence/absence of each emotion with a presence score threshold *thres*                                                                                                                                                                                                                                                                                                                                                                                                                                       | results/<br>*model_name/*<br>confusion_matrix/ |
| Metrics                        | `regression.csv` <br> `classif_score_coarse.csv`  <br> `classif_pres_t_{thres}.csv`  <br> `classif_dom.csv`                                            | CSV files with all the metrics: <br> **regression**: Regression metrics calculated from the closeness with the presence score <br> **classif_score_coarse**: Classification metrics based on presence score (from 0 to 3) for each emotion <br> **classif_pres**: Classification metrics based on the presence/absence of an emotion (presence score >= thres) <br> **classif_dom**: Classification metrics based on identifying the dominant emotion(s) (the highest presence score)                                                                                                                                   | results/<br>*model_name*/csv/                  |

[Go to top](#sota-models-for-multimodal-emotion-recognition) :arrow_up:


## Data preprocessing and resulting statistics

This section presents the work conducted on feature data and labels, as well as the statistics obtained after the data preprocessing step.

### Preprocessing of feature data
Based on [this tutorial on CMU-Multimodal SDK](https://github.com/Justin1904/CMU-MultimodalSDK-Tutorials/blob/master/tutorial_interactive.ipynb) 
(recommended by the authors of CMU-MOSEI), here are the steps undertaken:

1. The following datapoints are removed:
   - datapoints with one or two missing modalities
   - the three modalities do not have the same number of time steps
   - datapoints that do not belong to any of the training, validation and test id lists provided by the SDK
2. All NaN are replaced by numbers
3. Visual and acoustic modalities are z-normalised

The output structure of the processed data from the function `process_and_split_dataset` is the following 
(example with the training set `train_list`):

  <p font="italic" align="center">  

  <img src="images/cmu_processed_data.png" align="center" alt="processed_data_structure">

  </p>

### Preprocessing of labels
The labels are processed based on the targeted task chosen by the user (can be set by the parameter `label_type`):
- `all`: original presence scores provided by the dataset
- `coarse`: presence score for each emotion among [0, 1, 2, 3]
- `present`: one-hot encoded labels for multi-label classification
- `dominant`: one-hot encoded labels for one-label classification

### Data split statistics

|            | # of samples | Proportion |
|------------|--------------|------------|
| Training   | 16315        | 71.43%     |
| Validation | 1871         | 8.19%      |
| Test       | 4654         | 20.38%     |

### Label statistics

The two diagrams below show the label distribution for training, validation and test sets. 
It is worth noting that "happiness" is more expressed in the videos than the other 
emotion classes (the blue area is larger when the presence score is high). If you want to manipulate statistics and
generate other graphs, please refer to `label_statistics.ipynb` file.

  <p font="italic" align="center">  

  <img src="images/cmu-mosei_stats_area.png" align="center" alt="cmu_mosei_label_distribution_area">

  </p>

  <p font="italic" align="center">  

  <img src="images/cmu-mosei_stats_hist.png" align="center" alt="cmu_mosei_label_distribution_hist">

  </p>

[Go to top](#sota-models-for-multimodal-emotion-recognition) :arrow_up:

## Models available

*The current code only runs the following model. We aim to add more models in this repository.*

Want to try one of the following models? Set `--model_name` parameter with the string value in bold.

- **ef_williams**: Williams, J., Kleinegesse, S., Comanescu, R., & Radu, O. (2018, July). 
[Recognizing Emotions in Video Using Multimodal DNN Feature Fusion](http://www.aclweb.org/anthology/W18-3302). 
In *Proceedings of Grand Challenge and Workshop on Human Multimodal Language (Challenge-HML)* (pp. 11-19). 
   - Updated version of [Multimodal DNN](https://github.com/rhoposit/MultimodalDNN)
   - Use standard training, validation and test sets from CMU-MOSEI SDK (instead of using only training and validation sets 
originally specified in the paper)

- **mult**: Tsai, Y. H. H., Bai, S., Liang, P. P., Kolter, J. Z., Morency, L. P., & Salakhutdinov, R. (2019, July). 
[Multimodal Transformer for Unaligned Multimodal Language Sequences](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7195022/). 
In *Proceedings of the conference. Association for Computational Linguistics. Meeting* (Vol. 2019, p. 6558). NIH Public Access.
  - Original repository: [Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer)
  - 6 output cells (or 7 with the neutral class) instead of 1 

- **ef\_williams\_debug**: the same model as **ef_williams** used for debugging on PyTorch or TensorFlow.

[Go to top](#sota-models-for-multimodal-emotion-recognition) :arrow_up:

## How to add a new model

*The instructions below give the guidelines for introducing a new model. Some issue might occur that is not covered 
here. Before creating an issue, make sure you follow the instructions. If the problem persists, I will try to help as 
soon as possible. Once your model runs, you can create a pull request. 
Thank you for your contribution!*

The three main steps to follow and are explained here are:
- How to structure the code of your model (for both PyTorch and TensorFlow)
- How to set the hyperparameters of your model
- How to make it usable in this repository

**How to structure the code of your model**

For PyTorch:

1. Create your model in a Python class and name it `MModel`: `class MModel(nn.Module)`.
2. The class must at least contain the following functions:
   1. `__init__` with these exact parameters: `def __init__(self, model_id, param_archi, num_features_text, 
   num_features_image, num_features_audio, num_classes)`. The parameter `param_archi` is a dictionary defined in 
   `config/hyperparameters.json` (will be explained in the next subsection) 
   2. `forward` which inputs the three feature arrays (text - image - audio, in this order) and returns the output 
   (the dimension must be *self.num_classes*): `def forward(self, x_text, x_image, x_audio)`. If you work with variable 
   length, each array will have dimension [num_steps, num_features]. If you work with fixed length, it will be 
   [batch_size, num_steps, num_features].

For TensorFlow:
1. Create your model in a Python class and name it `MModel`: `class MModel`.
2. The class must at least contain the following functions:
   1. `__init__` with these exact parameters: `def __init__(self, model_id, param_archi, num_features_text, 
   num_features_image, num_features_audio, num_steps, num_classes)`. The parameter `param_archi` is a dictionary defined
   in `config/hyperparameters.json` (will be explained in the next subsection)
   2. `build` which returns a `Model` object from `keras` library: `def build(self)`. The output of the model must be of
   dimension *self.num_classes*. Note that TensorFlow do not support variable input length (as far as I know).

**How to set the hyperparameters of your model**

[//]: # (#TODO Must list the compulsory hyperparameters and force it in the code.)

**How to make it usable in this repository**

1. Add the `.py` file containing the code of the model in `src/models/` folder. Choose wisely the name of your model: 
it will help others to identify your model. 
   - *If you have multiple files (like `mult`), create a folder with the name of your model.*
2. In `src/hyperparam_processing.py`: 
   1. Add the name of your model in the global variable `possible_models`
   2. Import the `.py` file: `from src.models import your_model_name`
      - *If you have multiple files, import the main file containing the class of the model:* 
      `from src.models.your_model_name import main_file_of_your_model`
   3. Add your model in `model_dict` of `get_model_module` function:
      - key: name in `possible_models`
      - value: name of imported file

Now your model is ready to be trained!


[Go to top](#sota-models-for-multimodal-emotion-recognition) :arrow_up:
