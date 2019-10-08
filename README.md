# Efficient Lifelong Learning with A-GEM

This repo is forked from the [official implementation of RWalk and AGEM](https://github.com/facebookresearch/agem) and modified to enable single-headed evaluation for incremental learning.

It is used to generate baseline results for EWC++, SI, MAS, and RWalk reported in our paper [Class-incremental Learning via Deep Model Consolidation](https://arxiv.org/abs/1903.07864).

## Requirements

TensorFlow >= v1.9.0.

## Training

To replicate the results of the paper on a particular dataset, execute (see the Note below for downloading the CUB and AWA datasets):
```bash
$ ./replicate_results.sh <DATASET> <THREAD-ID> <JE>
```
Example runs are:
```bash
$ ./replicate_results.sh MNIST 3      /* Train PNN and A-GEM on MNIST */
$ ./replicate_results.sh CUB 1 1      /* Train JE models of RWALK and A-GEM on CUB */
```

### Note
For CUB and AWA experiments, download the dataset prior to running the above script. Run following for downloading the datasets:

```bash
$ ./download_cub_awa.sh
```
The plotting code is provided under the folder `plotting_code/`. Update the paths in the plotting code accordingly.
 
When using this code, please consider cite our paper:

```
@article{zhang2019class,
  title={Class-incremental learning via deep model consolidation},
  author={Zhang, Junting and Zhang, Jie and Ghosh, Shalini and Li, Dawei and Tasci, Serafettin and Heck, Larry and Zhang, Heming and Kuo, C-C Jay},
  journal={arXiv preprint arXiv:1903.07864},
  year={2019}
}

and the papers by the original author:
@inproceedings{AGEM,
  title={Efficient Lifelong Learning with A-GEM},
  author={Chaudhry, Arslan and Ranzato, Marc’Aurelio and Rohrbach, Marcus and Elhoseiny, Mohamed},
  booktitle={ICLR},
  year={2019}
}

@inproceedings{chaudhry2018riemannian,
  title={Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence},
  author={Chaudhry, Arslan and Dokania, Puneet K and Ajanthan, Thalaiyasingam and Torr, Philip HS},
  booktitle={ECCV},
  year={2018}
}
```
## License
This source code is released under The MIT License found in the LICENSE file in the root directory of this source tree. 
