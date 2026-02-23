# [[CVPR-2024] LEAD: Learning Decomposition for Source-free Universal Domain Adaptation](https://arxiv.org/abs/2403.03421)

## Introduction
The main challenge for Source-free Universal Domain Adaptation (SF-UniDA) is determining whether covariate-shifted samples belong to target-private unknown categories. Existing methods tackle this either through hand-crafted thresholding or by developing time-consuming iterative clustering strategies. In this paper, we propose a new idea of LEArning Decomposition (LEAD), which decouples features into source-known and -unknown components to identify target-private data.  Technically, LEAD initially leverages the orthogonal decomposition analysis for feature decomposition. Then, LEAD builds instance-level decision boundaries to adaptively identify target-private data. Extensive experiments across various UniDA scenarios have demonstrated the effectiveness and superiority of LEAD. Notably, in the OPDA scenario on VisDA dataset, LEAD outperforms GLC by 3.5\% overall H-score and reduces 75\% time to derive pseudo-labeling decision boundaries.

## Framework
<img src="figures/LEAD_framework.png" width="1000"/>

## Prerequisites
- python3, pytorch, numpy, PIL, scipy, sklearn, tqdm, etc.
- We have presented the our conda environment file in `./environment.yml`.

## Dataset
We have conducted extensive expeirments on four datasets with three category shift scenario, i.e., Partial-set DA (PDA), Open-set DA (OSDA), and Open-partial DA (OPDA). The following is the details of class split for each scenario. Here, $\mathcal{Y}$, $\mathcal{\bar{Y}_s}$, and $\mathcal{\bar{Y}_t}$ denotes the source-target-shared class, the source-private class, and the target-private class, respectively. 

| Datasets    | Class Split| $\mathcal{Y}/\mathcal{\bar{Y}_s}/\mathcal{\bar{Y}_t}$| |
| ----------- | --------   | -------- | -------- |
|     | OPDA       | OSDA     | PDA      |
| Office-31   | 10/10/11   | 10/0/11  | 10/21/0  |
| Office-Home | 10/5/50    | 25/0/40  | 25/40/0  |
| VisDA-C     | 6/3/3      | 6/0/6    | 6/6/0    |
| DomainNet   | 150/50/145 |          |          |

Please manually download these datasets from the official websites, and unzip them to the `./data` folder. To ease your implementation, we have provide the `image_unida_list.txt` for each dataset subdomains. 

```
./data
├── Office
│   ├── Amazon
|       ├── ...
│       ├── image_unida_list.txt
│   ├── Dslr
|       ├── ...
│       ├── image_unida_list.txt
│   ├── Webcam
|       ├── ...
│       ├── image_unida_list.txt
├── OfficeHome
│   ├── ...
├── VisDA
│   ├── ...
```
### Step
1. Please prepare the environment first.
2. Please download the datasets from the corresponding official websites, and then unzip them to the `./data` folder.
3. Preparing the source model.
4. Performing the target model adaptation.

## Training
1. Open-partial Domain Adaptation (OPDA) on Office, OfficeHome, and VisDA
```
# Source Model Preparing
bash ./scripts/train_source_OPDA.sh
# Target Model Adaptation
bash ./scripts/train_target_OPDA.sh
```
2. Open-set Domain Adaptation (OSDA) on Office, OfficeHome, and VisDA
```
# Source Model Preparing
bash ./scripts/train_source_OSDA.sh
# Target Model Adaptation
bash ./scripts/train_target_OSDA.sh
```
3. Partial-set Domain Adaptation (PDA) on Office, OfficeHome, and VisDA
```
# Source Model Preparing
bash ./scripts/train_source_PDA.sh
# Target Model Adaptation
bash ./scripts/train_target_PDA.sh
```

## Citation
If you find our codebase helpful, please star our project and cite our paper:
```
@inproceedings{sanqing2024LEAD,
  title={LEAD: Learning Decomposition for Source-free Universal Domain Adaptation},
  author={Qu, Sanqing and Zou, Tianpei and He, Lianghua and Röhrbein, Florian and Knoll, Alois and Chen, Guang and Jiang, Changjun},
  booktitle={CVPR},
  year={2024},
}

@inproceedings{sanqing2023GLC,
  title={Upcycling Models under Domain and Category Shift},
  author={Qu, Sanqing and Zou, Tianpei and Röhrbein, Florian and Lu, Cewu and Chen, Guang and Tao, Dacheng and Jiang, Changjun},
  booktitle={CVPR},
  year={2023},
}

@inproceedings{sanqing2022BMD,
  title={BMD: A general class-balanced multicentric dynamic prototype strategy for source-free domain adaptation},
  author={Qu, Sanqing and Chen, Guang and Zhang, Jing and Li, Zhijun and He, Wei and Tao, Dacheng},
  booktitle={ECCV},
  year={2022}
}
```

## Contact
- sanqingqu@gmail.com or 2011444@tongji.edu.cn

---

# LEAD-

在开始前我先说一下我的复现大致历程吧，年前的时间，继续在本机上进行Pythorch的训练及数据复现，在Google Colab上进行计图的复现，
为了防止断掉，其实我也准备在本机上做，年后继续在Colab上跑，中间断了一次，但是pro的计算单元已经不够再跑一次了，我就临时在本机上
跑复现了，这里有两个问题，一个是计图好像不适配Mac，所以我对计图编译器那部分的代码做了一定的调整，最后可以跑，另一问题是计图不支
持MPS（MacBook的GPU），也就是说只能用CPU跑，也就是对于一些epch长的我需要缩短它的epoch，但是很显然，这样的复现也只能证明其可
以跑得通，其数据其实是没有价值的，我在这里放的数据也就只供参考了

至于对编译代码的改动后的版本，我后续看看整理出来

至于一些还要说的，大概就是，基于想要从头训练来说，一块GPU还是相对吃力并且不保险的
