# XBCR-net (Cross B-Cell Receptor network) for antibody-antigen binding prediction




This repo provides an implementation of the training and inference pipeline of XBCR-net based on tensorflow and Keras. The original implementation of its backbone network ACNN could be found in [ACNN repo](https://github.com/XiaoYunZhou27/ACNN).

Any publication that discloses findings arising from using this source code or the model parameters should [cite](#citing-this-work) the
[XBCR-net paper](https://doi.org/10.1038/s41422-022-00727-6) and, if applicable, the [ACNN paper](https://ieeexplore.ieee.org/abstract/document/9197328).

![header](imgs/fig1.jpg)

## Requirement
* Python 3.8
* Tensorflow == 2.4.1
* numpy == 1.19.5
* pandas == 1.1.0
(Other version might be also applicable but cannot be guaranteed)

## Usage

* Setup
```
$DOWNLOAD_DIR/XBCR-net/           
    data/[data_name]/
        pos/
            # experimental dataset for training (.xlsx|.csv files)
        neg/
            # negative samples for training (.xlsx|.csv files)
        test/
            to_pred/
                proc_data.xlsx/csv # the file for inference
            results/
                results_rbd_[model_name]-[model_num].xlsx # the file to print the inference results
    models/[data_name]/
        [data_name]-[model_name]/
            # the files of model parameters (.tf.index and .tf.data-000000-of-00001 files)
```

* Training
```
python main_train.py --model_name XBCR_net --data_name [data_name] --model_num [model_num] --max_epochs [max_epochs] --include_light [1/0]
```

* Inference
```
python main_infer.py --model_name XBCR_net --data_name [data_name] --model_num [model_num] --include_light [1/0]
```


