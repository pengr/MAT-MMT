## HybridVocab: Towards Multi-Modal Machine Translation via Multi-Aspect Alignment [[Paper]](https://scholar.archive.org/work/pgyxeohzt5bbhoqkcfvla5brtq/access/wayback/https://dl.acm.org/doi/pdf/10.1145/3512527.3531386)
![](https://github.com/pengr/MAT-MMT/blob/master/MAT-MMT.png)


## Step1: Requirements
- Build running environment (two ways)
```shell
  1. pip install --editable .  
  2. python setup.py build_ext --inplace
````
- Install the syntax parser
```shell
  pip install Stanza 1.2.2 Stanza_batch 0.2.2
````
- pytorch==1.7.0, torchvision==0.8.0, cudatoolkit=10.1 (pip install is also work)
```shell
  conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch 
````
- Python 3.7.6
- [Meteor-1.5](https://www.cs.cmu.edu/~alavie/METEOR/README.html), and its compiler [Java JDK 1.8.0](https://www.oracle.com/sg/java/technologies/javase/javase8-archive-downloads.html) (or higher)


## Step2: Data Preparation
The dataset used in this work is Multi30K, 
both its original and preprocessed versions (that I used) 
are available at [here](https://github.com/multi30k/dataset/tree/master/data/task2).

You can download your own data set and then refer to 
*experiments/prepare-iwslt14.sh* or *experiments/prepare-wmt14en2de.sh* to pre-process the data set.

File Name | Description |  Download
---|---|---
`resnet50-avgpool.npy` | pre-extracted image features, each image is represented as a 2048-dimensional vector. | [Link](https://1drv.ms/u/s!AuOGIeqv1TybbQeJMw8CdqOphfA?e=l8k4df)
`Multi30K EN-DE Task` | BPE+TOK text, Image Index, Label for English-German task (including train, val, test2016/17/mscoco) | [Link](https://github.com/multi30k/dataset/tree/master/data/task2/tok)
`Multi30K EN-FR Task` | BPE+TOK text, Image Index, Label for English-French task (including train, val, test2016/17/mscoco) | [Link](https://github.com/multi30k/dataset/tree/master/data/task2/tok)


## Step3: Running code
You can let this code works by run the scripts in the directory *expriments*.

1. preprocess dataset into torch type
    ```bash
    bash pre.sh
    ```
    
2. train model
    ```bash
    bash train.sh
    ```
   
3. generate target sentence
    ```bash
    bash gen.sh
    ```


## Citation
If you use the code in your research, please cite:
```bibtex
@inproceedings{peng2022hybridvocab,
    title={HybridVocab: Towards Multi-Modal Machine Translation via Multi-Aspect Alignment},
    author={Peng, Ru and Zeng, Yawen and Zhao, Junbo},
    booktitle={Proceedings of the 2022 International Conference on Multimedia Retrieval},
    pages={380--388},
    year={2022}
}
```
