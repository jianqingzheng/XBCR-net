# XBCR-net (Cross B-Cell Receptor network) for antibody-antigen binding prediction

[![DOI](https://img.shields.io/badge/DOI-10.1038%2Fs41422--022--00727--6-darkgray)](https://www.nature.com/articles/s41422-022-00727-6)

Code for [Deep learning-based rapid generation of broadly reactive antibodies against SARS-CoV-2 and its Omicron variant](https://doi.org/10.1038/s41422-022-00727-6)


This repo provides an implementation of the training and inference pipeline of XBCR-net based on tensorflow and Keras. The original implementation of its backbone network ACNN could be found in [ACNN repo](https://github.com/XiaoYunZhou27/ACNN).



![header](imgs/fig1.jpg)
*a The features of the amino acid sequences of VH, VL and RBD sequences were extracted, localized and max-pooled to be concatenated together as input to the fully connected layers. The active features in the latent space were then processed by Multi-Layer Perceptron to predict the binding probability of antibody to multiple antigens. The impact score of VH, VL and RBD is calculated on the local histogram impact score map, representing how much weight is given to the specified amino acids on VH, VL (y axis) and RBD (x axis). Prediction results are evaluated by Precision-Recall curves of ACNN (Violet), Transformer (Gray), FCN (Red) and CNN (Blue). b The HCDR3 sequences of the predicted SARS-CoV-2 and Omicron variant binders are clustered by using an 80% sequence similarity. Cluster size represents the number of BCR sequences in the cluster. For each expanded cluster, the HCDR3 sequences are visualized as a sequence logo plot, where y-axis represents the frequency of the individual amino acid at the corresponding position in x-axis. The frequency of the dominating VH gene is listed above the logo. c Circos plot showing the frequency of antibodies encoded by the specified V region to J region pairing of the pan-SARS2 sequences. d The diversity of the four groups of BCR repertoire is analyzed, which is linked to the sample number of each group. e Binding of the predicted cross-reactive antibodies to RBD of SARS-CoV-2 and Omicron variants (left panel) was examined by ELISA. Representative OD reading is plotted as heatmap ranging from 0.05 to 5.0, and OD of 0.1 is used as cut-off value (n = 3 per group). f The SARS-CoV-2 Omicron variant (BA.1) pseudovirus neutralization curves of XBN-1, XBN-6 and XBN-11 mAbs were generated from luciferase readings at 8 dilutions (n = 3). g HCDR3 sequences of the XBN-1 and XBN-11 are aligned with the most convergent anti-SARS-CoV-2 antibodies from the published studies. h The HCDR3 sequence frequency of the dominant cluster (encoded by IGHV3-30 and IGKV1-13) of the pan-SARS group is shown.*

## Requirement
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4.1-blue)](www.tensorflow.org)
[![Numpy](https://img.shields.io/badge/Numpy-1.19.5-blue)](www.tensorflow.org)

* Python 3.8
* Tensorflow == 2.4.1
* numpy == 1.19.5
* pandas == 1.1.0
(Other version might be also applicable but cannot be guaranteed)

## Usage

* Setup
```
[$DOWNLOAD_DIR]/XBCR-net/           
    data/[$data_name]/
        pos/
            # experimental dataset for training (.xlsx|.csv files)
            example-experimental_data.xlsx
        neg/
            # negative samples for training (.xlsx|.csv files)
            example-negative_data.xlsx
        test/
            ab_to_pred/
                # the antibody data for inference
                example-antibody_to_predict.xlsx 
            ag_to_pred/
                # the antigen data for inference
                example-antigen_to_predict.xlsx 
            results/
                # the files to print the inference results
                results_rbd_[$model_name]-[$model_num].xlsx 
    models/[$data_name]/
        [$data_name]-[$model_name]/
            # the files of model parameters (.tf.index and .tf.data-000000-of-00001 files)
            model_rbd_[$model_num].tf.index
            model_rbd_[$model_num].tf.data-000000-of-00001
```
Download [Data_S1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41422-022-00727-6/MediaObjects/41422_2022_727_MOESM2_ESM.xlsx) (optional)

* Training
```
cd $DOWNLOAD_DIR/XBCR-net
python ./main_train.py --model_name XBCR_net --data_name $data_name --model_num $model_num --max_epochs max_epochs --include_light [1/0]
```
example for training (default):
```
cd $DOWNLOAD_DIR/XBCR-net
python ./main_train.py --model_name XBCR_net --data_name binding --model_num 0 --max_epochs 100 --include_light 1
```


* Inference
```
cd $DOWNLOAD_DIR/XBCR-net
python ./main_infer.py --model_name XBCR_net --data_name $data_name --model_num $model_num --include_light [1/0]
```
example for inference (default):
```
cd $DOWNLOAD_DIR/XBCR-net
python ./main_infer.py --model_name XBCR_net --data_name binding --model_num 0 --include_light 1
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

