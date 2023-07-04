---

<div align="center">    
 
# SAMBA: Regularized Autoencoders perform Sharpness-Aware Minimization

<!--  
[//]: # ([![Paper]&#40;http://img.shields.io/badge/paper-arxiv.2206.02416-B31B1B.svg&#41;]&#40;https://arxiv.org/abs/2206.02416&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/ICLR-2019-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018&#41;)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/AnyConference-year-4b44ce.svg&#41;]&#40;https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018&#41;  )

[![Paper](http://img.shields.io/badge/arxiv-stat.ML:2206.02416-B31B1B.svg)](https://arxiv.org/abs/2206.02416)
-->  
[![Conference](http://img.shields.io/badge/AABI-2023.svg)]([https://openreview.net/forum?id=G4GpqX4bKAH](https://openreview.net/forum?id=gk3PAmy_UNz))

![CI testing](https://github.com/rpatrik96/vae-sam/workflows/CI%20testing/badge.svg?branch=master&event=push)

<!-- 
[![DOI](https://zenodo.org/badge/431811003.svg)](https://zenodo.org/badge/latestdoi/431811003)
-->  

<!--  
Conference   
-->   
</div>
 
## Description   


## How to run   
First, install dependencies   
```bash
# clone vae-sam   
git clone --recurse-submodules https://github.com/rpatrik96/vae-sam

# if forgot to pull submodules, run
git submodule update --init

# install vae-sam   
cd vae-sam
pip install -e .   
pip install -r requirements.txt



# install submodule requirements
pip install --requirement tests/requirements.txt --quiet

# install pre-commit hooks (only necessary for development)
pre-commit install
 ```   
 Next, navigate to the `vae-sam` directory and run `vae_sam/cli.py.   
```bash
 python3 vae_sam/cli.py fit --help
 python3 vae_sam/cli.py fit --config configs/config.yaml
```

### Hyperparameter optimization

First, you need to log into `wandb`
```bash
wandb login #you will find your API key at https://wandb.ai/authorize
```

Then you can create and run the sweep
```bash
wandb sweep sweeps/sam.yaml  # returns sweep ID
wandb agent <ID-comes-here> --count=<number of runs> # when used on a cluster, set it to one and start multiple processes
```



## Citation   

```

@inproceedings{
anonymous2023samba,
title={{SAMBA}: Regularized Autoencoders perform Sharpness-Aware Minimization},
author={Anonymous},
booktitle={Fifth Symposium on Advances in Approximate Bayesian Inference},
year={2023},
url={https://openreview.net/forum?id=gk3PAmy_UNz}
}

```   
