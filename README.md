# XBCR-net (Cross B-Cell Receptor network) for antibody-antigen binding prediction

Code for [Deep learning-based rapid generation of broadly reactive antibodies against SARS-CoV-2 and its Omicron variant](https://doi.org/10.1038/s41422-022-00727-6)
```
Deep learning-based rapid generation of broadly reactive antibodies against SARS-CoV-2 and its Omicron variant
```

This repo provides an implementation of the training and inference pipeline of XBCR-net based on tensorflow and Keras. The original implementation of its backbone network ACNN could be found in [ACNN repo](https://github.com/XiaoYunZhou27/ACNN).



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

# Citing this work

Any publication that discloses findings arising from using this source code or the network model should cite
```bibtex
@article{lou2022deep,
  title={Deep learning-based rapid generation of broadly reactive antibodies against SARS-CoV-2 and its Omicron variant},
  author={Lou, Hantao and Zheng, Jianqing and Fang, Xiaohang Leo and Liang, Zhu and Zhang, Meihan and Chen, Yu and Wang, Chunmei and Cao, Xuetao},
  journal={Cell Research},
  pages={1--3},
  year={2022},
  publisher={Nature Publishing Group}
}
```
and, if applicable, the [ACNN paper](https://ieeexplore.ieee.org/abstract/document/9197328):
```bibtex
@inproceedings{zhou2020acnn,
  title={Acnn: a full resolution dcnn for medical image segmentation},
  author={Zhou, Xiao-Yun and Zheng, Jian-Qing and Li, Peichao and Yang, Guang-Zhong},
  booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={8455--8461},
  year={2020},
  organization={IEEE}
}
```

