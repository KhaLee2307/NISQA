# NISQA: Speech Quality and Naturalness Assessment

*+++ News: The NISQA model has recently been updated to NISQA v2.0. The new version offers multidimensional predictions with higher accuracy and allows for training and finetuning the model.*

**Speech Quality Prediction:**   
NISQA is a deep learning model/framework for speech quality prediction. The NISQA model weights can be used to predict the quality of a speech sample that has been sent through a communication system (e.g telephone or video call). Besides overall speech quality, NISQA also provides predictions for the quality dimensions *Noisiness*, *Coloration*, *Discontinuity*, and *Loudness* to give more insight into the cause of the quality degradation. 

**TTS Naturalness Prediction:**  
The NISQA-TTS model weights can be used to estimate the *Naturalness* of synthetic speech generated by a Text-To-Speech system (Siri, Alexa, etc.).

**Training/Finetuning:**   
NISQA can be used to train new single-ended or double-ended speech quality prediction models with different deep learning architectures, such as CNN or DFF -> Self-Attention or LSTM -> Attention-Pooling or Max-Pooling. The provided model weights can also be applied to finetune the trained model towards new data or for transfer-learning to a different regression task (e.g. quality estimation of enhanced speech, speaker similarity estimation, or  emotion recognition) .

**Speech Quality Datasets:**  
We provide a large corpus of more than 14,000 speech samples with subjective speech quality and speech quality dimension labels. 

## Table of Contents
- [Installation](#installation)
- [Using NISQA](#using-nisqa)
  - [Prediction](#prediction)
  - [Training](#training)
    - [Finetuning / Transfer Learning](#finetuning--transfer-learning)
    - [Training a new model](#training-a-new-model)
  - [Evaluation](#evaluation)
- [NISQA Corpus](#nisqa-corpus)
- [Paper and License](#paper-and-license)

More information about the deep learning model structure, the used training datasets, and the training options, see the [NISQA paper](https://arxiv.org/abs/2104.09494) and the [Wiki](https://github.com/gabrielmittag/NISQA/wiki/).


## Installation

To install requirements install [Anaconda](https://www.anaconda.com/products/individual) and then use:

```setup
conda env create -f env.yml
```

This will create a new environment with the name "nisqa". Activate this environment to go on:

```setup2
conda activate nisqa
```



## Using NISQA

We provide examples for using NISQA to predict the quality of speech samples, to train a new speech quality model, and to evaluate the performance of a trained speech quality model. 

There are three different model weights available, the appropriate weights should be loaded depending on the domain:

| Model                 | Prediction Output                                           | Domain             | Filename           |
| --------------------- | ----------------------------------------------------------- | ------------------ | ------------------ |
| NISQA (v2.0)          | Overall Quality, Noisiness, Coloration, Noisiness, Loudness | Transmitted Speech | nisqa.tar          |
| NISQA (v2.0) mos only | Overall Quality only (for finetuning/transfer learning)     | Transmitted Speech | nisqa_mos_only.tar |
| NISQA-TTS (v1.0)      | Naturalness                                                 | Synthesized Speech | nisqa_tts.tar      |

### Prediction

There are three modes available to predict the quality of speech via command line arguments:
* Predict a single file
* Predict all files in a folder
* Predict all files in a CSV table

Select "*nisqa.tar*" to predict the quality of a transmitted speech sample and "*nisqa_tts.tar*" to predict the Naturalness of a synthesized speech sample.

To predict the quality of a single .wav file use:

```
python run_predict.py --mode predict_file --pretrained_model weights/nisqa.tar --deg /path/to/wav/file.wav --output_dir /path/to/dir/with/results
```
To predict the quality of all .wav files in a folder use:
```
python run_predict.py --mode predict_dir --pretrained_model weights/nisqa.tar --data_dir /path/to/folder/with/wavs --num_workers 0 --bs 10 --output_dir /path/to/dir/with/results
```

To predict the quality of all .wav files listed in a csv table use:
```
python run_predict.py --mode predict_csv --pretrained_model weights/nisqa.tar --csv_file files.csv --csv_deg column_name_of_filepaths --num_workers 0 --bs 10 --output_dir /path/to/dir/with/results
```

The results will be printed to the console and saved to a csv file in a given folder (optional with --output_dir). To speed up the prediction, the number of workers and batch size of the Pytorch Dataloader can be increased (optional with --num_workers and --bs). 

### Training

#### Finetuning / Transfer Learning

To use the model weights to finetune the model on a new dataset, only a CSV file with the filenames and labels is needed. The training configuration is controlled from a YAML file and can be started as follows:

```
python run_train.py --yaml config/finetune_nisqa.yaml
```

- If you use the [NISQA Corpus](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus), you only need to update two arguments in the YAML file and you are ready to go: The `input_dir` to the extracted NISQA_Corpus folder and the `output_dir`, where the results should be stored.

- If you use your own dataset or want to load the NISQA-TTS model, some other updates are needed. 

  Your CSV file needs to contain at least three columns with the following names

  - `db` with the individual dataset names for each file
  - `filepath_deg` filepath to the degraded WAV file, either absolute paths or relative to the `input_dir` (CSV column name can be changed in YAML)
  - `mos` with the target labels (CSV column name can be changed in YAML)

  The `finetune_nisqa.yaml` needs to be updated as follows:

  - `input_dir` path to the main folder, which contains the CSV file and the datasets
  - `output_dir` path to output folder with saved model weights and results
  - `pretrained_model` filename of the pretrained model, either `nisqa_mos_only.tar` for natural speech or `nisqa_tts.tar` for synthesized speech
  - `csv_file` name of the CSV with filepaths and target labels
  - `csv_deg` CSV column name that contains filepaths (e.g. `filepath_deg`)
  - `csv_mos_train` and `csv_mos_val` CSV column names of the target value (e.g. `mos`)
  - `csv_db_train` and `csv_db_val` names of the datasets you want to use for training and validation. Datasets names must be in the `db` column.

See the comments in the YAML configuration file and the [Wiki](https://github.com/gabrielmittag/NISQA/wiki/) (not yet added) for more advanced training options. A good starting point would be to use the NISQA Corpus to get the training started with the standard configuration.

#### Training a new model

NISQA can also be used as a framework to train new speech quality models with different deep learning architectures. The general model structure is as follows:

1. *Framewise model:* CNN or Feedforward network
2. *Time-Dependency* model: Self-Attention or LSTM
3. *Pooling:* Average, Max, Attention or Last-Step-Pooling

The framewise and time-dependency models can be skipped, for example to train an LSTM model without CNN that uses the last-time step for prediction. Also a second time-dependency stage can be added, for example for LSTM-Self-Attention structure. The model structure can be easily controlled via the YAML configuration file. The training with the standard NISQA model configuration can be started with the [NISQA Corpus](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus) as follows:

```
python run_train.py --yaml config/train_nisqa_cnn_sa_ap.yaml
```

If you use the [NISQA Corpus](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus), you only need to update the `input_dir` to the extracted NISQA_Corpus folder and the `output_dir` in the YAML file. Otherwise, see the previous [finetuning section](#finetuning-transfer-learning) for updating the YAML file if you use your own dataset.

It is also possible to train any other combination of neural networks, for example, to train a model with LSTM instead of Self-Attention, the `train_nisqa_cnn_lstm_avg.yaml` example configuration file is provided. 

To train a **double-ended** model for full-reference speech quality prediction, the `train_nisqa_double_ended.yaml` configuration file can be used as an example. See the comments in the YAML files and the [Wiki](https://github.com/gabrielmittag/NISQA/wiki/) (not yet added) for more details on different possible model structures and advanced training options.

### Evaluation

Trained models can be evaluated on a given dataset as follows (can also be used as a conformance test of the model installation):

```
python run_evaluate.py
```

Before running, the options and paths inside the Python script `run_evaluate.py` should be updated. If the [NISQA Corpus](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus) is used, only the `data_dir` and `output_dir` paths need to be adjusted. Besides Pearson's Correlation and RMSE, also an RMSE after first-order polynomial mapping is calculated. If a CSV file with per-condition labels is provided, the script will also output per-condition results and RMSE*. Optionally, correlation diagrams can be plotted. The script should return the same results as in the NISQA paper when it is run on the NISQA Corpus.

## NISQA Corpus

The NISQA Corpus includes more than 14,000 speech samples with simulated (e.g. codecs, packet-loss, background noise) and live (e.g. mobile phone, Zoom, Skype, WhatsApp) conditions.   

For the download link and more details on the datasets and used source speech samples see the [NISQA Corpus Wiki](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus).

## Paper and License

- If you use the **NISQA model** or the **NISQA Corpus** for your research, please cite following paper:  
  [G. Mittag, B. Naderi, A. Chehadi, and S. Möller “NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets,” arXiv:2104.09494 [eess.AS], 2021.](https://arxiv.org/abs/2104.09494)
- Please cite following paper if you use the **NISQA-TTS** model for Naturalness prediction of synthesized speech:  
  [G. Mittag and S. Moller, “Deep Learning Based Assessment of Synthetic Speech Naturalness,” in Proc. Interspeech 2020, 2020.](https://www.isca-speech.org/archive/Interspeech_2020/abstracts/2382.html)
- Please cite following paper if you use the **double-ended NISQA model**:  
  [G. Mittag and S. Möller. Full-reference speech quality estimation with attentional Siamese neural networks. In Proc. ICASSP 2020, 2020.](https://ieeexplore.ieee.org/document/9053951)
- The older NISQA (v0.42) model version is described in following paper:  
  [G.  Mittag  and  S.  Möller,  “Non-intrusive  speech  quality  assessment  for  super-wideband  speech  communication  networks,”  in Proc. ICASSP 2019, 2019](https://ieeexplore.ieee.org/document/8683770)

The NISQA code is licensed under [MIT License](LICENSE).

The model weights (nisqa.tar, nisqa_mos_only.tar, nisqa_tts.tar) are provided under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License](weights/LICENSE_model_weights)

The NISQA Corpus is provided under the original terms of the used source speech and noise samples. More information can be found in the  [NISQA Corpus Wiki](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus).

Copyright © 2021 Gabriel Mittag  
www.qu.tu-berlin.de

