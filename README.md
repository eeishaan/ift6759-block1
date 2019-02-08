# Introduction
Code for work done as part of block 1 assignment.
Code for following approaches is present:
- CNN for classification
- CNN for three regression problems
- CNN for classification + regression
- Deterministic algorithm for regression + (PCA+LDA) for classification

# Usage
Top level command
```
╔║ 02:27 PM ║ user20@helios1 ║ ~/ift6759-block1 ║╗
╚═> s_exec python3 -m omsignal -h
WARNING: underlay of /usr/bin/nvidia-smi required more than 50 (668) bind mounts
usage: omsignal [-h] {train,test} ...

optional arguments:
  -h, --help    show this help message and exit

commands:
  {train,test}
    train       Train models
    test        Test pre-trained models
```

Train command
```
╔║ 02:34 PM ║ user20@helios1 ║ ~/ift6759-block1 ║╗
╚═> s_exec python3 -m omsignal train -h
WARNING: underlay of /usr/bin/nvidia-smi required more than 50 (668) bind mounts
usage: omsignal train [-h] --model
                      {cnn_classification,cnn_regression,cnn_multi_task,lstm_classification,deterministic_task,best_model}
                      [--validation-data VALIDATION_DATA]
                      [--train-data TRAIN_DATA] [--params PARAMS]

optional arguments:
  -h, --help            show this help message and exit
  --model {cnn_classification,cnn_regression,cnn_multi_task,lstm_classification,deterministic_task,best_model}
                        Model type to train
  --validation-data VALIDATION_DATA
                        Validation data file location
  --train-data TRAIN_DATA
                        Train data file location
  --params PARAMS       Model param file location. For information about param
                        file format refer README.md

```

Test command
```
╔║ 02:34 PM ║ user20@helios1 ║ ~/ift6759-block1 ║╗
╚═> s_exec python3 -m omsignal test -h
WARNING: underlay of /usr/bin/nvidia-smi required more than 50 (668) bind mounts
usage: omsignal test [-h] --model
                     {cnn_classification,cnn_regression,cnn_multi_task,deterministic_task,best_model}
                     --model-path MODEL_PATH --data-file DATA_FILE

optional arguments:
  -h, --help            show this help message and exit
  --model {cnn_classification,cnn_regression,cnn_multi_task,deterministic_task,best_model}
                        Type of model to evaluate
  --model-path MODEL_PATH
                        File path of the model
  --data-file DATA_FILE
                        File path of the test data
```


# Definitions
- Experiment: A collection of net, optimizer and loss function.
- Model: A class sub-classing `torch.nn.Module`

# Algorithm addition
- To add a new algorithm one has to first make a new model.
- Then an experiment should be created that subclasses the `OmExperiment` template experiment.
- `OmExperiment` provides a lot of boiler plate code and reduces code duplication. It also provided a lot of hooks that can be used to write custom algorithms.
- After an experiment has been designed, we need functions to train and test the experiments.
- These functions are placed under `train.py` and `test.py` files.

# Param file
- Each experiment should have a param file located under `omsignal/params/` directory
- Example param file
    ```yml
    model_params: 
    n_layers: 1
    optimiser_params:
    lr: 0.0005
    weight_decay: 0.0001
    model_file: lstm_exp.pt  
    batch_size: 8
    epochs: 120
    ```


# Dir structure
```
├── evaluation
│   ├── eval.py
│   └── evaluation_stats.py
├── omsignal
│   ├── constants.py
│   ├── experiments
│   │   ├── cnn_experiment.py
│   │   ├── deterministic.py
│   │   ├── __init__.py
│   │   └── lstm_experiment.py
│   ├── fakedata
│   │   ├── MILA_TrainLabeledData.dat
│   │   └── MILA_ValidationLabeledData.dat
│   ├── __init__.py
│   ├── label_mapping.json
│   ├── logging_config.ini
│   ├── __main__.py
│   ├── models
│   │   ├── cnn.py
│   │   ├── __init__.py
│   │   └── lstm.py
│   ├── params
│   │   ├── cnn_classification.yml
│   │   ├── cnn_multi_task.yml
│   │   ├── cnn_regression.yml
│   │   ├── deterministic.yml
│   │   └── lstm_exp.yml
│   ├── results
│   │   ├── deterministic_train.png
│   │   └── deterministic_valid.png
│   ├── saved_models
│   │   └── deterministic.pkl
│   ├── task.py
│   ├── test.py
│   ├── train.py
│   └── utils
│       ├── __init__.py
│       ├── loader.py
│       ├── memfile.py
│       ├── misc.py
│       ├── score.py
│       ├── signal_stats.py
│       ├── transform
│       │   ├── basic.py
│       │   ├── dim_reduction.py
│       │   ├── __init__.py
│       │   ├── preprocessor.py
│       │   └── signal.py
│       └── vis.py
├── README.md
```