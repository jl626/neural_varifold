# neural_varifold
neural varifold implementation

## Requirment
We tested our code on Ubuntu 22.04 environment. 

* CUDA 11.8
* Tensorflow 2.15
* Jax 0.4.21
* torch 2.7.1 + cuda118
* torch_geometric 2.6.1
* neural_tangents 0.6.4

## How to install

please type on commandline prompt 

```
conda env create -f environment_linux.yml
```

In order to use the library, one can activate conda virtual environment as follow: 

```
conda activate neural-varifold
```

## Data

data is available at [google drive](https://drive.google.com/file/d/1ANYa3JKhr696rxMt1kaxD265-xq5N7qO/view?usp=drive_link)

## Citing Neural Varifold

```
@article{lee2024neural,
  title={Neural varifolds: an aggregate representation for quantifying the geometry of point clouds},
  author={Lee, Juheon and Cai, Xiaohao and Sch{\"o}nlieb, Carola-Bibian and Masnou, Simon},
  journal={arXiv preprint arXiv:2407.04844},
  year={2024}
}
```
