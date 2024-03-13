# ArabicMorph

Deep Active Learning for Morphophonological Processing

## Abstract

Building a system for morphological processing is a challenging task in morphologically complex languages like Arabic. Although there are some deep learning based models that achieve successful results, these models rely on a large amount of annotated data. Building such datasets, specially for some of the lower-resource Arabic dialects, is very difficult, time-consuming, and expensive. In addition, some parts of the annotated data do not contain useful information for training machine learning models. Active learning strategies allow the learner algorithm to select the most informative samples for annotation. There has been little research that focuses on applying active learning for morphological inflection and morphophonological processing. In this paper, we have proposed a deep active learning method for this task. Our experiments on Egyptian Arabic show that with only about 30% of annotated data, we achieve the same results as does the state-of-the-art model on the whole dataset.

## Download Paper

- https://aclanthology.org/2023.acl-short.69/
- https://aclanthology.org/2023.acl-short.69.pdf

## How to Cite

If you find this work useful to you, please cite:

### ACL Anthology

```
Seyed Morteza Mirbostani, Yasaman Boreshban, Salam Khalifa, SeyedAbolghasem Mirroshandel, and Owen Rambow. 2023. Deep Active Learning for Morphophonological Processing. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 793–803, Toronto, Canada. Association for Computational Linguistics.
```

### Cite (Informal)

```
Deep Active Learning for Morphophonological Processing (Mirbostani et al., ACL 2023)
```

### BibTeX

```
@inproceedings{mirbostani-etal-2023-deep,
    title = "Deep Active Learning for Morphophonological Processing",
    author = "Mirbostani, Seyed Morteza  and
      Boreshban, Yasaman  and
      Khalifa, Salam  and
      Mirroshandel, SeyedAbolghasem  and
      Rambow, Owen",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-short.69",
    doi = "10.18653/v1/2023.acl-short.69",
    pages = "793--803",
    abstract = "Building a system for morphological processing is a challenging task in morphologically complex languages like Arabic. Although there are some deep learning based models that achieve successful results, these models rely on a large amount of annotated data. Building such datasets, specially for some of the lower-resource Arabic dialects, is very difficult, time-consuming, and expensive. In addition, some parts of the annotated data do not contain useful information for training machine learning models. Active learning strategies allow the learner algorithm to select the most informative samples for annotation. There has been little research that focuses on applying active learning for morphological inflection and morphophonological processing. In this paper, we have proposed a deep active learning method for this task. Our experiments on Egyptian Arabic show that with only about 30{\%} of annotated data, we achieve the same results as does the state-of-the-art model on the whole dataset.",
}
```

## Setup

Clone the following repository to your local system:

```bash
$ git clone https://github.com/mirbostani/ArabicMorph
$ cd ArabicMorph
```

Create an environment based on the provided `environment.yml` file to install the dependencies:

```bash
$ conda env create -f environment.yml
$ conda activate arabicmorph
```

## Active Learning

Copy the required `*.tsv` files from `data/arabicmorph/Egy_Data/ECAL/baseline_data` to `data/arabicmorph/active`.

```shell
$ cd data/arabicmorph/active
$ tree -L 1
.
├── 0_train_1000.tsv
├── 0_train_13171.tsv
├── 0_tune_1000.tsv
├── 0_tune_13171.tsv
├── dev_oov.tsv
├── dev.tsv
├── test_oov.tsv
└── test.tsv

0 directories, 8 files
```

Use the following bash script to train a transformer model with active learning.

```shell
$ cd arabicmorph
$ bash example/transformer/arabicmorph_al.sh
```

The content of the bash script is as follows:

```shell
python src/train_al.py \
    --prefix "am0" \
    --start_cycle 0 \
    --num_train_samples 900 \
    --num_tune_samples 500 \
    --num_cycle_samples 250 \
    --data_dir "data/arabicmorph/active" \
    --checkpoint_dir "checkpoints/transformer" \
    --dataset_files \
    data/arabicmorph/active/0_tune_13171.tsv \
    data/arabicmorph/active/0_train_13171.tsv \
    --train_reference_files \
    data/arabicmorph/active/0_train_1000.tsv
```

For more information, use `--help` option.

```shell
$ python src/train_al.py --help
usage: train_al.py [-h] [--data_dir DATA_DIR] [--checkpoint_dir CHECKPOINT_DIR] [--dataset_files [DATASET_FILES ...]] [--train_reference_files [TRAIN_REFERENCE_FILES ...]] [--start_cycle START_CYCLE] [--num_train_samples NUM_TRAIN_SAMPLES] [--num_tune_samples NUM_TUNE_SAMPLES]
                   [--num_cycle_samples NUM_CYCLE_SAMPLES] --prefix PREFIX

optional arguments:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Path to the database directory
  --checkpoint_dir CHECKPOINT_DIR
                        Path to save trained models
  --dataset_files [DATASET_FILES ...]
                        Dataset samples in one or multiple files
  --train_reference_files [TRAIN_REFERENCE_FILES ...]
                        Use these files as references to extract initial training samples from dataset files
  --start_cycle START_CYCLE
                        Start active learning train cycle
  --num_train_samples NUM_TRAIN_SAMPLES
                        Number of samples randomly selected from merged train files for initial training
  --num_tune_samples NUM_TUNE_SAMPLES
                        Number of samples for tuning models
  --num_cycle_samples NUM_CYCLE_SAMPLES
                        Number of samples selected by active learning approache to be added to the previous training set
  --prefix PREFIX       Prefix used for the name of the files generated during the active learning procedure
```

## Egyptian Arabic Dataset

Please refer to the [paper](https://aclanthology.org/2023.acl-short.69.pdf) for more information regarding the dataset generated from the Egyptian Colloquial Arabic Lexicon (ECAL).

## Baseline

The baseline model extended for the experiments in this work is based on the character-level transducer model presented in [shijie-wu/neural-transducer](https://github.com/shijie-wu/neural-transducer).

## License

MIT