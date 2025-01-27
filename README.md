<div align="center">    
 
# Deep Traffic Generation
<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  

ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)
-->

<!--  
Conference   
-->   
</div>
 
## Description   

Package that provides neural networks (mostly autoencoders) to embed and generate traffic trajectories. This project relies on [traffic](https://traffic-viz.github.io/) and [Pytorch-Lightning](https://www.pytorchlightning.ai/) libraries.

## Installation

```bash
# create new python environment for traffic
conda create -n traffic -c conda-forge python=3.9 traffic
conda activate traffic

# clone project   
git clone https://github.com/kruuZHAW/deep-traffic-generation

# install project
cd deep-traffic-generation
pip install .
```

## How to run   
 Navigate to any python file in `deep_traffic_generation` and run it.   
 ```bash
# module folder
cd deep_traffic_generation

# example: run module with custom arguments
python tcvae.py --data_path *path to data* --prior vampprior --encoding_dim 32 --h_dims 64 64 64 --lr 0.001 --batch_size 100 --n_components 200 --features track groundspeed altitude timedelta --info_features latitude longitude --info_index -1
```

You can use Tensorboard to visualize training logs.

```bash
tensorboard --logdir lightning_logs
```
