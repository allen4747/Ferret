# Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models

This repo contains the official implementation for the work “**Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models**”. See more details in our [paper](https://arxiv.org/abs/2409.06277). 

> Large Language Models (LLMs) have become indispensable in numerous real-world applications. Unfortunately, fine-tuning these models at scale, especially in federated settings where data privacy and communication efficiency are critical, presents significant challenges. Existing methods often resort to parameter-efficient fine-tuning (PEFT) to mitigate communication overhead, but this typically comes at the cost of model accuracy. To address these limitations, we propose *federated full-parameter tuning at scale for LLMs* (Ferret), the first first-order method with shared randomness to enable scalable full-parameter tuning of LLMs across decentralized data sources while maintaining competitive model accuracy. **Ferret** accomplishes this through three aspects:(1) it employs widely applied first-order methods for efficient local updates; (2) it projects these updates into a low-dimensional space to considerably reduce communication overhead; and (3) it reconstructs local updates from this low-dimensional space with shared randomness to facilitate effective full-parameter global aggregation, ensuring fast convergence and competitive final performance. Our rigorous theoretical analyses and insights along with extensive experiments, show that **Ferret** significantly enhances the scalability of existing federated full-parameter tuning approaches by achieving high computational efficiency, reduced communication overhead, and fast convergence, all while maintaining competitive model accuracy.



## Project Structure
```Markdown
.
├── optimizers
│   └── ferret_optimizer.py  // implementation of Ferret
├── scripts
│   └── ferret.sh // example script to reproduce the result on NI task
├── utils_data
│   ├── default_tokens.py  // definitions of some special tokens
│   ├── llm_dataset.py  // utilities to load Dolly-15K
│   ├── load_data.py  // entrance to get dataloaders
│   ├── natural_instruction_loader.py  // utilities to load Natural Instructions
│   └── partition_data.py  // utilities to partition datasets in Dirichlet distribution
├── client.py
├── evaluations.py
├── main.py
└── server.py
```

## Requirements
Please see `requirements.txt`.

## Prepare the data
1. Natural Instructions:
To run experiments on [Natural Instructions v2.8](https://github.com/allenai/natural-instructions/releases/tag/v2.8), you need to unzip the downloaded dataset in directory `~/.datasets`.

2. Dolly-15K:
To run experiments on [Dolly-15K](https://github.com/databrickslabs/dolly), you need to download the corresponding dataset in directory `~/.datasets`, with its name as `databricks-dolly-15k.jsonl`.

## Run our code
We provide the example script to reproduce the experiment on Natural Instructions. The arguments can be adjusted according to the `help` information in their definitions. Our code are based on the code from [FedKSeed](https://github.com/zhenqincn/FedKSeed).

```Shell
# Ferret on Natural Instructions 
bash scripts/ferret.sh
```


## BibTeX
```latex
@article{shu2024ferret,
      title={Ferret: Federated Full-Parameter Tuning at Scale for Large Language Models}, 
      author={Yao Shu and Wenyang Hu and See-Kiong Ng and Bryan Kian Hsiang Low and Fei Richard Yu},
      journal={arXiv preprint arXiv:2409.06277}
      year={2024},
}
```
