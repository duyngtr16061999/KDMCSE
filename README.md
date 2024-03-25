# KDMCSE: Knowledge Distillation Multimodal Sentence Embeddings with Adaptive Angular margin Contrastive Learning
This repository contains code and pre-trained models for our NAACL-2024 paper [KDMCSE: Knowledge Distillation Multimodal Sentence Embeddings with Adaptive Angular margin Contrastive Learning]().

## Quickstart
### Setup
- Pytorch
- Other packages:
```
pip install -r requirements.txt
```

### Data Preparation

Please organize the data directory as following:
```
REPO ROOT
|
|--data    
|  |--wiki1m_for_simcse.txt  
|  |--flickr30k_ViT_L14.json
|  |--train_coco_ViT_L14.json
```

**Wiki1M**
```shell script
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
```

**Flickr30k & MS-COCO** \
Download the preprocessed data:
- [flickr30k_ViT_L14.json]()
- [train_coco_ViT_L14.json]()

Or preprocess the data by yourself (using raw images from Flickr and COCO with CLIP).

### Train & Evaluation
1. Prepare the senteval datasets for evaluation:
    ```
    cd SentEval/data/downstream/
    bash download_dataset.sh
    ```

2. Run scripts:
    ```shell script
    # For example:
    sh scripts/run_wiki_flickr.sh

    sh scripts/run_wiki_coco.sh
    ```

## Acknowledgements
- The extremely clear and well organized codebase: [MCSE](https://github.com/uds-lsv/MCSE)
- [SentEval](https://github.com/facebookresearch/SentEval) toolkit
