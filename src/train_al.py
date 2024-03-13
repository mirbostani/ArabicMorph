"""
Active Learning
"""
from tqdm import tqdm
from typing import Union, List, Tuple
import os
import argparse
import logging
import json
import subprocess
from collections import defaultdict
from pprint import pprint

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.cluster import KMeans

# import chars2vec
from lib import chars2vec # source is included due to some inconsistencies

import Levenshtein
import random
from sklearn.decomposition import PCA

from utils.generalizer_and_distance import UR_SF_generalizer


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)
logger = logging.getLogger(__name__)


class ActiveLearning:

    def __init__(self):
        self.args = self.parse_arguments()
        self.cycle = self.args.start_cycle

        self.arch = "transformer"
        self.database = "arabicmorphbaseline" if not self.args.database else self.args.database
        
        # In reversed mode, (ur, sf) columns are swapped in the baseline mode's ArabicMorphBaseLineReversed.read_file
        # Here, pool prediction means model is predicting sf in arabicmorphbaseline database mode and ur in arabicmorphbaselinereversed database mode
        # Therefore, in arabicmorphbaseline, prediction and target are model's sf prediction and dataset's label.
        # and in arabicmorphbaselinereversed, prediction and target are model's ur prediction and dataset's label.
        self.is_database_reversed = True if "reversed" in self.database else False

        self.lr = 0.001
        self.scheduler = "warmupinvsqr"
        self.max_steps = 20000
        self.warmup = 4000
        self.beta2 = 0.98
        self.label_smooth = 0.1
        self.total_eval = 50
        self.train_bs = 400 if not self.args.train_batch_size else self.args.train_batch_size
        self.test_bs = 1 if not self.args.test_batch_size else self.args.test_batch_size
        self.model_name_prefix = "model"

        # Transformers
        self.layers = 4
        self.hs = 1024
        self.embed_dim = 256
        self.nb_heads = 4
        self.dropout = 0.3

        self.tests = ["test", "test_oov", "dev", "dev_oov"]

        if self.args.training_method == "al":
            self.train_with_al()
        elif self.args.training_method == "rand":
            self.train_with_rand()

        # if self.cycle == 0:
        #     self.prepare_initial_datasets()
        # self.train(cycle=self.cycle)
        # self.save_to_results("test", self.cycle, "model")
        # for test in self.tests:
        #     self.test(test_name=test, cycle=self.cycle)
        #     self.save_to_results(log_prefix=test, cycle=self.cycle)
        # self.test(test_name="pool", cycle=self.cycle)
        # self.prepare_next_dataset(self.cycle)  # increases cycle

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data_dir",
            type=str,
            default="data/arabicmorph/active",
            help="Path to the database directory"
        )
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=400,
            help="Training batch size"
        )
        parser.add_argument(
            "--test_batch_size",
            type=int,
            default=1,
            help="Evaluation (tuning) batch size"
        )
        parser.add_argument(
            "--training_method",
            type=str,
            default="al",
            choices=["al", "rand"],
            help="Method of training the network."
        )
        parser.add_argument(
            "--database",
            type=str,
            default="arabicmorphbaseline",
            choices=["arabicmorphbaseline", "arabicmorphbaselinereversed"],
            help="Reverse the training process by switching (UR, SR) to (SR, UR)."
        )
        parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="checkpoints/transformer",
            help="Path to save trained models"
        )
        parser.add_argument(
            "--dataset_files",
            type=str,
            nargs="*",
            help="Dataset samples in one or multiple files"
        )
        parser.add_argument(
            "--train_reference_files",
            type=str,
            nargs="*",
            required=False,
            help="Use these files as references to extract initial training samples from dataset files"
        )
        # parser.add_argument(
        #     "--tune_reference_files",
        #     type=str,
        #     nargs="*",
        #     required=False,
        #     help="Use these files as references to extract initial test samples from dataset files"
        # )
        parser.add_argument(
            "--shuffle_dataset",
            action="store_true",
            default=False,
            help="Shuffle samples of the merged dataset files before training."
        )
        parser.add_argument(
            "--start_cycle",
            type=int,
            default=0,
            help="Start active learning train cycle"
        )
        parser.add_argument(
            "--num_train_samples",
            type=int,
            default=1000,
            help="Number of samples randomly selected from merged train files for initial training"
        )
        parser.add_argument(
            "--num_tune_samples",
            type=int,
            default=500,
            help="Number of samples for tuning models"
        )
        parser.add_argument(
            "--num_cycle_samples",
            type=int,
            default=250,
            help="Number of samples selected by active learning approache to be added to the previous training set"
        )
        # parser.add_argument(
        #     "--percent_tune_samples",
        #     type=int,
        #     required=False,
        #     help="Extract X percent of samples from current train set to be used for tuning"
        # )
        parser.add_argument(
            "--prefix",
            type=str,
            required=True,
            help="Prefix used for the name of the files generated during the active learning procedure"
        )
        parser.add_argument(
            "--sampling_method",
            type=str,
            choices=["max_entropy", "max_vowel_entropy"],
            default="max_entropy",
            help="Method of active learning sampling in each cycle."
        )
        parser.add_argument(
            "--num_clusters",
            type=int,
            default=1,
            help="Divide the selected samples in each cycle to clusters, then do the sampling proportionally"
        )
        parser.add_argument(
            "--num_pre_clustering_samples",
            type=int,
            default=1000, # 250 x 4
            help="These selected samples will be used for clustering."
        )
        parser.add_argument(
            "--clustering_method",
            type=str,
            default="kmeans-chars2vec",
            choices=["kmeans-tfidf", "kmeans-chars2vec", "kmeans-distance"],
            help="Clustering method"
        )
        parser.add_argument(
            "--kmeans_init_method",
            type=str,
            default="k-means++",
            choices=["k-means++", "random"],
            help="KMeans clustering initialization method."
        )
        parser.add_argument(
            "--tfidf_ngram_range",
            type=int,
            default=3,
            help="The ngram range used in tf-idf method"
        )
        parser.add_argument(
            "--chars2vec_emb_dim",
            type=int,
            default=50,
            help="Embedding dimension of chars2vec model"
        )
        parser.add_argument(
            "--pca_n_components",
            type=int,
            default=3,
            help="Number of components for PCA to reduce dimensions of chars2vec embeddings"
        )

        args = parser.parse_args()
        logger.info(json.dumps(vars(args), indent=4))
        return args

    def train_with_rand(self):
        logger.info("Training with random method...")

        if self.cycle == 0:
            self.prepare_initial_datasets()

        # Prepare next cycle if model is already trained
        model_reldirpath = self.get_model_reldirpath(self.cycle)
        if os.path.isdir(model_reldirpath):
            # Validate model on "test"
            self.save_to_results(
                log_prefix=self.model_name_prefix,
                cycle=self.cycle
            )
            # Validate tests
            for test in self.tests:
                self.test(test_name=test, cycle=self.cycle)
                self.save_to_results(log_prefix=test, cycle=self.cycle)
            # Predict pool (only used for finding common errors)
            ok = self.test(test_name="pool", cycle=self.cycle)
            if not ok:
                logger.info("Finished!")
                return
            ok = self.prepare_next_dataset_with_rand(self.cycle)
            if not ok:
                logger.info("Finished!")
                return

        while True:
            self.train(cycle=self.cycle)
            self.save_to_results(
                log_prefix=self.model_name_prefix,
                cycle=self.cycle
            )
            # Validate tests
            for test in self.tests:
                self.test(test_name=test, cycle=self.cycle)
                self.save_to_results(log_prefix=test, cycle=self.cycle)
            # Predict pool (only used for finding common errors)
            ok = self.test(test_name="pool", cycle=self.cycle)  # predict pool
            if not ok:
                logger.info("Finished!")
                break

            ok = self.prepare_next_dataset_with_rand(self.cycle) # increases cycle
            if not ok:
                logger.info("Finished!")
                break

    def train_with_al(self):
        logger.info("Training with active learning...")

        if self.cycle == 0:
            self.prepare_initial_datasets()

        # Prepare next cycle if model is already trained
        model_reldirpath = self.get_model_reldirpath(self.cycle)
        if os.path.isdir(model_reldirpath):
            # Validate model on "test"
            self.save_to_results(
                log_prefix=self.model_name_prefix,
                cycle=self.cycle
            )
            # Validate tests
            for test in self.tests:
                self.test(test_name=test, cycle=self.cycle)
                self.save_to_results(log_prefix=test, cycle=self.cycle)
            # Predict pool
            ok = self.test(test_name="pool", cycle=self.cycle)
            if not ok:
                logger.info("Finished!")
                return
            ok = self.prepare_next_dataset_with_al(self.cycle)
            if not ok:
                logger.info("Finished!")
                return

        while True:
            self.train(cycle=self.cycle)
            self.save_to_results(
                log_prefix=self.model_name_prefix,
                cycle=self.cycle
            )
            # Validate tests
            for test in self.tests:
                self.test(test_name=test, cycle=self.cycle)
                self.save_to_results(log_prefix=test, cycle=self.cycle)
            # Predict pool
            ok = self.test(test_name="pool", cycle=self.cycle)  # predict pool
            if not ok:
                logger.info("Finished!")
                break

            ok = self.prepare_next_dataset_with_al(self.cycle)  # increases cycle
            if not ok:
                logger.info("Finished!")
                break

    def merge_file_samples(self, name: str, files: List[str]):
        samples = []
        for file in files:
            filepath = os.path.join(os.getcwd(), file)
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"File {filepath} not found!")
            with open(filepath, "r", encoding="utf-8") as fp:
                samples += [line.strip() for line in fp.readlines()]
                if self.args.shuffle_dataset:
                    logger.info("Dataset is shuffled.")
                    random.shuffle(samples)
        logger.info(f"{name} samples: {len(samples)}")
        return samples

    def get_filename(self, name: str, cycle: int, count: int):
        return f"{self.args.prefix}_{cycle}_{name}_{count}.tsv"

    def get_filepath(self, filename):
        return os.path.join(os.getcwd(), self.args.data_dir, filename)

    def save_file(self, filepath: str, samples: List[str]):
        with open(filepath, "w", encoding="utf-8") as fp:
            fp.writelines([f"{sample}\n" for sample in samples])

    def save_samples(self, name: str, cycle: int, samples: List):
        count = len(samples)
        filename = self.get_filename(name=name, cycle=cycle, count=count)
        filepath = self.get_filepath(filename=filename)
        self.save_file(filepath=filepath, samples=samples)
        logger.info(f"{name} samples: {count}")
        logger.info(f"{filepath} saved")

    def get_samples(self, name: str, cycle: int):
        filepath = self.get_samples_filepath(name=name, cycle=cycle)
        if not filepath:
            return []
        if not os.path.isfile(filepath):
            logger.warning(f"File {filepath} not found!")
            return []
        with open(filepath, "r", encoding="utf-8") as fp:
            samples = [line.strip() for line in fp.readlines()]
            return samples

    def get_samples_filename(self, name: str, cycle: int):
        filenames = []
        for dirpath, dirnames, file_names in os.walk(self.args.data_dir):
            filenames.extend(file_names)
            break
        prefix = f"{self.args.prefix}_{cycle}_{name}_"
        for filename in filenames:
            if filename.startswith(prefix):
                return filename
        logger.warning(f"File {prefix}*.tsv not found!")
        return None

    def get_samples_relfilepath(self, name: str, cycle: int):
        filename = self.get_samples_filename(name=name, cycle=cycle)
        if filename:
            return os.path.join(self.args.data_dir, filename)
        return None

    def get_samples_filepath(self, name: str, cycle: int):
        relfilepath = self.get_samples_relfilepath(name=name, cycle=cycle)
        if not relfilepath:
            return None
        return os.path.join(os.getcwd(), relfilepath)

    def get_tests_relfilepath(self, name):
        relfilepath = os.path.join(self.args.data_dir, f"{name}.tsv")
        if not os.path.isfile(os.path.join(os.getcwd(), relfilepath)):
            raise FileNotFoundError(f"File {relfilepath} not found!")
        return relfilepath

    def get_model_dirname(self, cycle):
        filename = self.get_samples_relfilepath(name="train", cycle=cycle)
        count = int(filename.split("/")[-1].split(".")[0].split("_")[-1])
        return f"{self.args.prefix}_{cycle}_{count}"

    def get_model_reldirpath(self, cycle):
        return os.path.join(
            self.args.checkpoint_dir,
            self.arch,
            self.get_model_dirname(cycle)
        )

    def get_model_prefix(self, cycle: int):
        return os.path.join(
            self.get_model_reldirpath(cycle),
            self.model_name_prefix
        )

    def get_chars2vec_model_reldirpath(self, cycle):
        return os.path.join(
            self.args.checkpoint_dir,
            self.arch,
            self.get_model_dirname(cycle),
            "chars2vec"
        )

    def prepare_initial_datasets(self):
        self.dataset = self.merge_file_samples(
            name="dataset",
            files=self.args.dataset_files
        )

        # Select samples from dataset that are in refereces
        if self.args.train_reference_files:
            self.train_ref_dataset = self.merge_file_samples(
                name="train_reference",
                files=self.args.train_reference_files
            )
            self.train_set = [
                sample
                for sample in self.dataset
                if sample in self.train_ref_dataset
            ][:self.args.num_train_samples]
        # Select samples from dataset (no reference is available)
        else:
            self.train_set = [
                sample
                for sample in self.dataset
            ][:self.args.num_train_samples]

        self.save_samples(name="train", cycle=0, samples=self.train_set)

        # Select samples from beginning of the dataset that are not in the train set
        self.tune_set = [
            sample
            for sample in self.dataset
            if sample not in self.train_set
        ][:self.args.num_tune_samples]

        self.save_samples(name="tune", cycle=0, samples=self.tune_set)

        # Select samples from dataset that are not in the train and tune sets
        self.pool_set = [
            sample
            for sample in self.dataset
            if sample not in self.train_set and sample not in self.tune_set
        ]

        self.save_samples(name="pool", cycle=0, samples=self.pool_set)

    def train(self, cycle: int):
        train_samples = self.get_samples(name="train", cycle=cycle)
        if len(train_samples) == 0:
            logger.error("No train set for training.")
            return False

        tune_samples = self.get_samples(name="tune", cycle=0)  # always 0
        if len(tune_samples) == 0:
            logger.error("No tune set for training.")
            return False

        # pool_samples = self.get_samples(name="pool", cycle=cycle)
        # if len(pool_samples) == 0:
        #     logger.error("No pool set for training.")
        #     return False

        train_relfilepath = self.get_samples_relfilepath("train", cycle)
        tune_relfilepath = self.get_samples_relfilepath("tune", 0) # always 0
        test_relfilepath = self.get_tests_relfilepath("test")
        model_prefix = self.get_model_prefix(cycle)
        cmd = f"""
        python src/train.py \
            --dataset "{self.database}" \
            --train "{train_relfilepath}" \
            --dev "{tune_relfilepath}" \
            --test "{test_relfilepath}" \
            --model "{model_prefix}" \
            --embed_dim {self.embed_dim} \
            --src_hs {self.hs} \
            --trg_hs {self.hs} \
            --dropout {self.dropout} \
            --nb_heads {self.nb_heads} \
            --label_smooth {self.label_smooth} \
            --total_eval {self.total_eval} \
            --src_layer {self.layers} \
            --trg_layer {self.layers} \
            --max_norm 1 \
            --lr {self.lr} \
            --shuffle \
            --arch {self.arch} \
            --gpuid 0 \
            --estop 1e-8 \
            --bs {self.train_bs} \
            --max_steps {self.max_steps} \
            --scheduler {self.scheduler} \
            --warmup_steps {self.warmup} \
            --cleanup_anyway \
            --beta2 {self.beta2} \
            --bestacc
        """
        output = subprocess.check_output(cmd, shell=True)
        logger.info(f"{model_prefix} is trained.")

        return True

    def test(self, test_name: str, cycle: int) -> bool:
        train_samples = self.get_samples(name="train", cycle=cycle)
        if len(train_samples) == 0:
            logger.error("No train set for evaluation.")
            return False

        tune_samples = self.get_samples(name="tune", cycle=0)  # always 0
        if len(tune_samples) == 0:
            logger.error("No tune set for evaluation.")
            return False

        if "pool" in test_name:
            pool_samples = self.get_samples(name="pool", cycle=cycle)
            if len(pool_samples) == 0:
                return False

        train_relfilepath = self.get_samples_relfilepath("train", cycle)
        tune_relfilepath = self.get_samples_relfilepath("tune", 0) # always 0
        if "pool" in test_name:
            test_relfilepath = self.get_samples_relfilepath("pool", cycle)
        else:
            test_relfilepath = self.get_tests_relfilepath(test_name)
        model_prefix = self.get_model_prefix(cycle)
        model_reldirpath = self.get_model_reldirpath(cycle)

        model_relfilepath = None
        filenames = []
        for dirpath, dirnames, file_names in os.walk(model_reldirpath):
            filenames.extend(file_names)
            break
        for filename in filenames:
            if filename.startswith(self.model_name_prefix) and ".epoch_" in filename:
                model_relfilepath = os.path.join(model_reldirpath, filename)
        if not model_relfilepath:
            raise FileNotFoundError(f"Model {model_reldirpath} not found!")

        cmd = f"""
        python src/test.py \
            --dataset "{self.database}" \
            --train "{train_relfilepath}" \
            --dev "{tune_relfilepath}" \
            --test "{test_relfilepath}" \
            --model "{os.path.join(model_reldirpath, test_name)}" \
            --load "{model_relfilepath}" \
            --embed_dim {self.embed_dim} \
            --src_hs {self.hs} \
            --trg_hs {self.hs} \
            --dropout {self.dropout} \
            --nb_heads {self.nb_heads} \
            --label_smooth {self.label_smooth} \
            --total_eval {self.total_eval} \
            --src_layer {self.layers} \
            --trg_layer {self.layers} \
            --max_norm 1 \
            --lr {self.lr} \
            --shuffle \
            --arch {self.arch} \
            --gpuid 0 \
            --estop 1e-8 \
            --bs {self.test_bs} \
            --max_steps {self.max_steps} \
            --scheduler {self.scheduler} \
            --warmup_steps {self.warmup} \
            --cleanup_anyway \
            --beta2 {self.beta2} \
            --bestacc
        """
        output = subprocess.check_output(cmd, shell=True)
        logger.info(f"{model_prefix} is evaluated.")

        return True

    def save_to_results(self, cycle: int, log_prefix: str):
        dirname = self.get_model_dirname(cycle)
        reldirpath = self.get_model_reldirpath(cycle)
        log_filepath = os.path.join(
            os.getcwd(),
            reldirpath,
            f"{log_prefix}.log"
        )

        if not os.path.isfile(log_filepath):
            raise FileNotFoundError(f"Log file {log_filepath} not found!")

        with open(log_filepath, "r", encoding="utf-8") as fp:
            lines = [line.strip() for line in fp.readlines()]
            inst = lines[-4].split(" - ")[-1].strip().split(" ")[2]
            acc = lines[-1].split(" - ")[-1].strip().split(" ")[3]
            dist = lines[-1].split(" - ")[-1].strip().split(" ")[5]
            loss = lines[-6].split(" - ")[-1].strip().split(" ")[4]
            message = f"{dirname}\t{log_prefix}\t{inst}\t{acc}\t{dist}\t{loss}"
            self.save_message_to_results(message)

    def save_message_to_results(self, message):
        filepath = os.path.join(
            os.getcwd(),
            self.args.data_dir,
            f"{self.args.prefix}_results.tsv"
        )
        with open(filepath, "a", encoding="utf-8") as fp:
            fp.write(f"{message}\n")
            logger.info(f"Results saved: {message}")

    def prepare_next_dataset_with_rand(self, cycle: int) -> bool:
        """
        Prepare next dataset based on the random training method.
        """

        logger.info("Preparing next dataset with random training...")

        train = self.parse_train_file(cycle)
        pool = self.parse_pool_file(cycle)
        if len(pool) == 0:
            return False
        self.sample_from_pool_rand(pool=pool)
        self.cycle += 1
        p = self.save_next_pool(pool, self.cycle)
        t = self.save_next_train(train, pool, self.cycle)
        return p and t

    def prepare_next_dataset_with_al(self, cycle: int) -> bool:
        """
        Prepare next dataset based on the active learning.
        """

        logger.info("Preparing next dataset with active learning...")

        train = self.parse_train_file(cycle)
        pool = self.parse_pool_file(cycle)
        if len(pool) == 0:
            return False
        prediction = self.parse_pool_prediction_file(cycle)
        self.sample_from_pool_al(pool=pool, prediction=prediction)
        self.cycle += 1
        p = self.save_next_pool(pool, self.cycle)
        t = self.save_next_train(train, pool, self.cycle)
        return p and t

    def parse_train_file(self, cycle) -> List:
        filepath = self.get_samples_filepath(name="train", cycle=cycle)
        with open(filepath, "r", encoding="utf-8") as f:
            examples = [
                {
                    "example": line.strip(),
                    "ur": line.strip().split("\t")[0],  # underlying representation
                    "sf": line.strip().split("\t")[1],  # surface form (target)
                }
                for line in f.readlines()
            ]
            return examples

    def parse_tune_file(self, cycle) -> List:
        filepath = self.get_samples_filepath(name="tune", cycle=cycle)
        with open(filepath, "r", encoding="utf-8") as f:
            examples = [
                {
                    "example": line.strip(),
                    "ur": line.strip().split("\t")[0],  # underlying representation
                    "sf": line.strip().split("\t")[1],  # surface form (target)
                }
                for line in f.readlines()
            ]
            return examples

    def parse_pool_file(self, cycle) -> List:
        filepath = self.get_samples_filepath(name="pool", cycle=cycle)
        with open(filepath, "r", encoding="utf-8") as f:
            examples = [
                {
                    "example": line.strip(),
                    "ur": line.strip().split("\t")[0],  # underlying representation
                    "sf": line.strip().split("\t")[1],  # surface form (target)
                    "sampled": False
                }
                for line in f.readlines()
            ]
            return examples

    def parse_pool_prediction_file(self, cycle: int):
        dirpath = self.get_model_reldirpath(cycle=cycle)
        filepath = os.path.join(dirpath, "pool.decode.test.tsv")
        with open(filepath, "r", encoding="utf-8") as f:
            examples = [
                line.strip().split("\t")
                for line in f.readlines()[1:]  # skip header
            ]
            # The transducer model reverses the (ur, sf) in reversed mode.
            # The order of data appearance in tsv files won't change.
            # arabicmorphbaseline: (ur, sf) -> prediction is model's output (sf), target is label (sf)
            # arabicmorphbaselinereversed: (ur, sf) -> prediction is models output (ur), target is label (ur)
            examples = [
                {
                    "prediction": prediction,
                    "target": target,
                    "loss": loss,
                    "edit_distance": int(dist),
                    "max_entropy": float(max_entropy),
                    "max_vowel_entropy_v2": float(max_vowel_entropy_v2), # not used in decoding.py and train.py
                    "max_vowel_entropy_v3": float(max_vowel_entropy_v3)
                }
                for prediction, target, loss, dist, max_entropy, max_vowel_entropy_v2, max_vowel_entropy_v3 in examples
            ]
            return examples

    def sample_from_pool_rand(self, pool: List):
        logger.info("Sampling from pool randomly...")

        num_cycles = self.args.num_cycle_samples
        if len(pool) < num_cycles:
            num_cycles = len(pool)

        selected = random.sample(pool, k=num_cycles) # random unique samples
        for v in tqdm(selected, desc="Random sampling"):
            v["sampled"] = True

    def sample_from_pool_al(self, prediction: List, pool: List):
        # Do not use edit distance as a sampling method for active learning.
        # It is calculated based on the prediction and target (label) comparison.
        # However, target is not available (should be hidden) in active learning.
        # key = "edit_distance"

        key = "max_entropy"
        if self.args.sampling_method == "max_vowel_entropy":
            key = "max_vowel_entropy_v3"
        # elif self.args.sampling_method == "max_vowel_entropy_limited":
        #     key = "max_vowel_entropy_v2"

        if self.args.num_clusters > 1:
            self.sample_from_pool_al_with_cluster(prediction=prediction, pool=pool, method=key)
        else:
            self.sample_from_pool_al_no_cluster(prediction=prediction, pool=pool, method=key)


    def sample_from_pool_al_no_cluster(self, prediction: List, pool: List, method: str):
        logger.info("Sampling from pool with no clustering...")

        assert len(pool) == len(prediction)

        # Add pool index to model predictions in order to find and remove samples
        # from the pool more easily. (It has nothing to do with AL method.)
        for i, v in enumerate(prediction):
            v["pool_idx"] = i

        # Sort based on the highest method value (e.g., "entropy")
        prediction = sorted(
            prediction,
            key=lambda k: -k[method]
        )

        num_cycles = self.args.num_cycle_samples
        if len(prediction) < num_cycles:
            num_cycles = len(prediction)

        tgt_col = "sf"
        if self.is_database_reversed:
            tgt_col = "ur"    

        # prediction[0] -> {'prediction': '0 b E D 0', 'target': '0 b E D 0', 'loss': '0.0606859028339386', 'edit_distance': 0, 'max_entropy': 0.42284250259399414, 'max_vowel_entropy_v2': 0.0, 'max_vowel_entropy_v3': 0.3525691628456116, 'pool_idx': 0}
        # pool[0]       -> {'example': '0 b E D 0\t0 b E D 0', 'ur': '0 b E D 0', 'sf': '0 b E D 0', 'sampled': False}
        selected = prediction[:num_cycles]
        for v in selected:
            idx = v["pool_idx"]
            if v["target"] == pool[idx][tgt_col]:
                pool[idx]["sampled"] = True
            else:
                logger.warn("Selected sample not found in pool!") # weird!

        # for i in range(num_cycles):
        #     # arabicmorphbaseline target is sf
        #     # arabicmorphbaselinereversed target is ur
        #     target = prediction[i]["target"]
        #     found = False
        #     for v in pool:
        #         if v[tgt_col] == target:
        #             v["sampled"] = True
        #             # print(i, tuple(self.pool_prediction[i].values()))
        #             found = True
        #             break
        #     if not found:
        #         raise Exception("Prediction's target is missing from pool!")

    
    def sample_from_pool_al_with_cluster(self, prediction: List, pool: List, method: str):
        logger.info("Sampling from pool with clustering...")

        assert len(pool) == len(prediction)

        # Add pool index to model predictions in order to find and remove samples
        # from the pool more easily. (It has nothing to do with AL method.)
        for i, v in enumerate(prediction):
            v["pool_idx"] = i

        # Sort based on the highest method value (e.g., "entropy")
        prediction = sorted(
            prediction,
            key=lambda k: -k[method]
        )

        # Select pool predictions for clustering
        preds = prediction[:self.args.num_pre_clustering_samples]

        if self.args.clustering_method == "kmeans-tfidf":
            words = [v["prediction"] for v in preds]
            clusters_index = self.kmeans_tfidf_clustering(words=words) # [3, 0, 1, ...]
        if self.args.clustering_method == "kmeans-chars2vec":
            words = [v["prediction"] for v in preds]
            clusters_index = self.kmeans_chars2vec_clustering(words=words)
        if self.args.clustering_method == "kmeans-distance":
            # normal: (ur, predicted_sf)
            # reversed: (sf, predicted_ur)
            input_col = "ur" if not self.is_database_reversed else "sf"
            pairs = [(pool[v["pool_idx"]][input_col], v["prediction"]) for v in preds]
            # clusters_index = self.kmeans_stem_clustering(pairs=pairs)
            # clusters_index = self.distance_clustering(pairs=pairs)
            clusters_index = self.kmeans_distance_clustering(pairs=pairs)
        else:
            raise Exception("Unknown clustering method.")

        # Divide samples to separate lists (i.e., clusters)
        # word_clusters = [[] for _ in range(self.args.num_clusters)]
        clusters = [[] for _ in range(self.args.num_clusters)]
        for i, cluster_index in enumerate(clusters_index):
            # word_clusters[cluster_index].append(words[i])
            clusters[cluster_index].append(preds[i])

        # Limit maximum number of cycle samples when samples are limited
        num_cycles = self.args.num_cycle_samples
        if len(preds) < num_cycles:
            num_cycles = len(preds)

        tgt_col = "sf"
        if self.is_database_reversed:
            tgt_col = "ur" 
        
        # Prepare number of samples to be selected based on proportions
        lengths = [len(c) for c in clusters]
        total = sum(lengths)
        proportions = [int((l/total) * num_cycles) for l in lengths]
        delta = num_cycles - sum(proportions)
        for j in range(len(clusters)):
            if delta == 0:
                break
            if lengths[j] > proportions[j]:
                proportions[j] += 1
                delta -= 1
        
        # Select samples from clusters based on proportions
        clusters_sampled = [v[:proportions[i]] for i, v in enumerate(clusters)]
        samples = [w for c in clusters_sampled for w in c]

        for v in samples:
            idx = v["pool_idx"]
            if v["target"] == pool[idx][tgt_col]:
                pool[idx]["sampled"] = True
            else:
                logger.warn("Selected sample not found in pool!") # weird!

        # # Mark selected samples in pool predictions dataset
        # for sample in samples:
        #     target = sample["target"]
        #     found = False
        #     for v in pool:
        #         if v[tgt_col] == target:
        #             v["sampled"] = True
        #             found = True
        #             break
        #     if not found:
        #         raise Exception("Prediction's target is missing from pool!")


    def strip_word(self, word, reversed: bool = False, remove_zero: bool = True):
        if reversed:
            if remove_zero:
                word = ['0'] + list(word) + ['0']
            word = " ".join(word)
            return word

        word = "".join(word.strip().split()) # remove spaces
        if remove_zero:
            word = word[1:-1] if len(word) > 2 else word # remove start/end '0'
        return word


    def kmeans_tfidf_clustering(self, *, words: List):
        # Remove spaces and start/end '0'
        words = [self.strip_word(w) for w in words]

        ngr = self.args.tfidf_ngram_range
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(ngr, ngr) # (1, 1) for single chars
        )
        fitted_vector = vectorizer.fit(words)
        terms = fitted_vector.get_feature_names_out()
        tfidf_vector = fitted_vector.transform(words)
        X = tfidf_vector.toarray()

        model = KMeans(
            n_clusters=self.args.num_clusters,
            init=self.args.kmeans_init_method, # "k-means++" or "random"
            max_iter=1000,
            n_init=1
        )
        model.fit(X)
        order_centroids = model.cluster_centers_.argsort()[:, ::-1] # reversed on dim=1
        
        w = vectorizer.transform(words)
        predictions = model.predict(w) # closest cluster for each sample in 

        return predictions


    def kmeans_chars2vec_clustering(self, *, words: List):
        # Prepare embeddings train dataset
        train = self.parse_train_file(cycle=self.cycle)
        tune = self.parse_tune_file(cycle=0)
        emb_train = train + tune
        emb_dim = self.args.chars2vec_emb_dim

        # Extract and prepare valid characters
        chars = set()
        for sample in emb_train:
            sample_chars = list(set(list(
                self.strip_word(sample["ur"]) + 
                self.strip_word(sample["sf"])
            )))
            chars.update(sample_chars)

        # ['$', "'", '*', '-', '=', 'A', 'D', 'E', 'H', 'I', 'O', 'S', 'T', 
        #  'U', 'Z', 'a', 'b', 'c', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 
        #  'm', 'n', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        chars = sorted(list(chars)) # e.g., 38

        # Perpare training pair samples.
        # Similar words' target is 0 and not similar words's target is 1
        # e.g., ("jinEn=at", "jinEnit") is similar, and target is 0
        # e.g., ("jinEn=at", "'iSSugayyara") is not similar, and target is 1
        X_train_s = []
        y_train_s = []
        X_train_ns = []
        y_train_ns = []
        
        # selected_ns_examples = [] # ['0 j i n E n = a t 0\t0 j i n E n i t 0', ...]

        for sample in tqdm(emb_train, total=len(emb_train), desc="Preparing chars2vec training dataset"):
            # Similar
            sample_ur_stripped = self.strip_word(sample['ur'])
            sample_sf_stripped = self.strip_word(sample['sf'])
            if self.is_database_reversed:
                X_train_s.append((sample_sf_stripped, sample_ur_stripped)) # ur is target
            else:
                X_train_s.append((sample_ur_stripped, sample_sf_stripped)) # sf is target
            y_train_s.append(0) # similar

            # Not Similar
            dist = []
            for o in emb_train:
                o_ur_stripped = self.strip_word(o['ur'])
                o_sf_stripped = self.strip_word(o['sf'])
                if o_ur_stripped != sample_ur_stripped and o_sf_stripped != sample_sf_stripped:
                    if self.is_database_reversed:
                        levenshtein_distance = Levenshtein.distance(sample_ur_stripped, o_ur_stripped) # ur is target
                    else:
                        levenshtein_distance = Levenshtein.distance(sample_sf_stripped, o_sf_stripped) # sf is target
                    if levenshtein_distance > 0 and not o.get("paired", False):
                        dist.append({
                            "sample_example": sample["example"],
                            "sample_ur_stripped" : sample_ur_stripped,
                            "sample_sf_stripped": sample_sf_stripped,
                            "o_example": o["example"],
                            "o_ur_stripped": o_ur_stripped,
                            "o_sf_stripped": o_sf_stripped,
                            "dist": levenshtein_distance,
                            "source_sample": sample,
                            "target_sample": o
                        })

            # dist_nz = [
            #     d for d in dist 
            #     if d["dist"] > 0 and d["o_example"] not in selected_ns_examples
            # ] 
            # if len(dist_nz) == 0:
            #     logger.warn("No non-similar example is found in dist_nz")
            #     continue
            # selected_ns = choices(dist_nz, k=1)
            # if len(selected_ns) == 0:
            #     logger.warn("No non-similar example is found")
            #     continue

            if len(dist) == 0:
                logger.warn("No non-similar example is found in dist")
                continue

            selected_ns = random.choices(dist, k=1)
            
            if len(selected_ns) == 0:
                logger.warn("No non-similar example is found")
                continue            

            selected_ns = selected_ns[0]

            # selected_ns_examples.append(selected_ns["o_example"])
            selected_ns["target_sample"]["paired"] = True
            
            if self.is_database_reversed:
                # ur is target
                X_train_ns.append(
                    (sample_sf_stripped, selected_ns["o_ur_stripped"])
                )
            else:
                # sf is target
                X_train_ns.append(
                    (sample_ur_stripped, selected_ns["o_sf_stripped"])
                )
            y_train_ns.append(1) # not similar

        logger.info(f"Similar: {len(y_train_s)}, Not Similar: {len(y_train_ns)}")

        X_train = X_train_s + X_train_ns
        y_train = y_train_s + y_train_ns

        # Train embeddings model
        c2v_model = chars2vec.train_model(
            emb_dim=emb_dim,
            X_train=X_train,
            y_train=y_train,
            model_chars=chars
        )

        # Save the embeddings model in the model's directory
        path_to_model = self.get_chars2vec_model_reldirpath(cycle=self.cycle)
        chars2vec.save_model(c2v_model=c2v_model, path_to_model=path_to_model)

        # Remove spaces and start/end '0' from words
        words = [self.strip_word(w) for w in words]

        # Convert words to embeddings vector
        X = c2v_model.vectorize_words(words)

        # # Scale features
        scaler = StandardScaler()
        # # scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)

        # Project embeddings on plane
        Xd = PCA(n_components=self.args.pca_n_components).fit_transform(Xs) # or X with no scale

        # Find optimum number of clusters
        sse = []
        r = range(1, self.args.num_clusters + 1)
        for k in r:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=42
            )
            kmeans.fit(Xd)
            sse.append(kmeans.inertia_)
        kn = KneeLocator(
            r,
            sse,
            curve="convex",
            direction="decreasing"
        )

        logger.info(f"Best n_cluster: {kn.elbow}")


        # Clustering model
        model = KMeans(
            # n_clusters=self.args.num_clusters,
            n_clusters=kn.elbow,
            init=self.args.kmeans_init_method, # "k-means++" or "random",
            max_iter=1000,
            n_init=1
        )
        model.fit(Xd)
        predictions = model.predict(Xd)

        return predictions


    def kmeans_distance_clustering(self, *, pairs: List[Tuple[str, str]]) -> List[int]:
        dists = []
        for idx, (input_word, pred_word) in tqdm(enumerate(pairs), desc="Distance"):
            if self.is_database_reversed:
                raise Exception("SF to UR requires the UR_SF_generalizer to be updated.")
            else:
                # ("jinEn=at", "'iSSugayyara") -> ('CiCEC=at', 'iCCugayyara')
                input_word_gen, _ = UR_SF_generalizer(
                    self.strip_word(word=input_word, remove_zero=False),
                    "" # pass empty for the function to work properly
                )
            
            # Calculate distance of a word from all other words
            dist = []
            for input_word2, pred_word2 in pairs:
                if self.is_database_reversed:
                    raise Exception("SF to UR requires the UR_SF_generalizer to be updated.")
                else:
                    input_word2_gen, _ = UR_SF_generalizer(
                        self.strip_word(word=input_word2, remove_zero=False),
                        "" # passing empty string is mandatory
                    )
                distance = Levenshtein.distance(input_word_gen, input_word2_gen)
                dist.append(distance)

            dists.append(dist)

        # convert to numpy vector, dimension is dependent on the number of words
        X = np.array(dists)

        # Scale features
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Project on a plane
        Xd = PCA(n_components=self.args.pca_n_components).fit_transform(Xs)
        # Xd = X

        # Find optimum number of clusters
        sse = []
        r = range(1, self.args.num_clusters + 1)
        for k in r:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=20,
                max_iter=600,
                random_state=43
            )
            kmeans.fit(Xd)
            sse.append(kmeans.inertia_)
        kn = KneeLocator(
            r,
            sse,
            curve="convex",
            direction="decreasing"
        )

        logger.info(f"Best n_cluster: {kn.elbow}")


        # Clustering model
        model = KMeans(
            # n_clusters=self.args.num_clusters,
            n_clusters=kn.elbow,
            init=self.args.kmeans_init_method, # "k-means++" or "random",
            max_iter=1000,
            n_init=1
        )
        model.fit(Xd)
        predictions = model.predict(Xd)

        return predictions


    def distance_clustering(self, *, pairs: List[Tuple[str, str]]):
        clusters = defaultdict(list)
        for idx, (input_word, pred_word) in enumerate(pairs):
            # ("jinEn=at", "'iSSugayyara") -> ('CiCEC=at', 'iCCugayyara')
            (input_word_gen, pred_word_gen) = UR_SF_generalizer(
                self.strip_word(word=input_word, remove_zero=False),
                self.strip_word(word=pred_word, remove_zero=False)
            )
            dist = Levenshtein.distance(input_word_gen, pred_word_gen)
            clusters[dist].append({
                "idx": idx,
                "input": input_word,
                "input_gen": input_word_gen,
                "pred": pred_word,
                "pred_gen": pred_word_gen,
                "dist": dist,
                "cluster": None
            })
        
        # Extract samples with the highest distances and select only the 
        # `num_clusters` clusters from them. Merge the rest.
        dists = sorted(list(clusters.keys()), reverse=True) # e.g., [9, 7, 3, 2, 1]
        samples = []
        index = 0
        for dist in dists:
            for sample in clusters[dist]:
                sample["cluster"] = index
                samples.append(sample)
            index += 1
            if index >= (self.args.num_clusters - 1):
                index = self.args.num_clusters - 1

        # Sort by idx to match initial pairs list
        samples_sorted = sorted(samples, key=lambda k: k["idx"])
        assert len(pairs) == len(samples)

        # Extract clusters index
        clusters_index = [v["cluster"] for v in samples_sorted]

        return clusters_index
            

    def kmeans_stem_clustering(self, *, pairs: List[Tuple[str, str]]):
        # Prepare embeddings train dataset
        train = self.parse_train_file(cycle=self.cycle)
        tune = self.parse_tune_file(cycle=0)
        emb_train = train + tune
        emb_dim = self.args.chars2vec_emb_dim

        # Extract and prepare valid characters
        chars = set()
        for sample in emb_train:
            (ur_gen, sf_gen) = UR_SF_generalizer(
                self.strip_word(word=sample["ur"], remove_zero=False), 
                self.strip_word(word=sample["sf"], remove_zero=False)
            )
            sample_chars = list(set(list(ur_gen + sf_gen)))
            chars.update(sample_chars)

        # ['$', "'", '-', '0', '=', 'A', 'C', 'D', 'E', 'H', 'I', 'O', 
        # 'S', 'T', 'U', 'Z', 'a', 'b', 'd', 'f', 'h', 'i', 'j', 'k', 'l', 
        # 'm', 'n', 'q', 'r', 's', 't', 'u', 'y', 'z']
        chars = sorted(list(chars)) # e.g., 38

        # Perpare training pair samples using generalizer
        # Similar words' target is 0 and not similar words's target is 1
        # e.g., ("jinEn=at", "jinEnit") -> ('CiCEC=at', 'CiCECit') is similar, and target is 0
        # e.g., ("jinEn=at", "'iSSugayyara") -> ('CiCEC=at', 'iCCugayyara') is not similar, and target is 1
        X_train_s = []
        y_train_s = []
        X_train_ns = []
        y_train_ns = []

        for sample in tqdm(emb_train, total=len(emb_train), desc="Preparing chars2vec training dataset"):
            # Similar
            (sample_ur_gen, sample_sf_gen) = UR_SF_generalizer(
                self.strip_word(word=sample['ur'], remove_zero=False), 
                self.strip_word(word=sample['sf'], remove_zero=False)
            )
            if self.is_database_reversed:
                X_train_s.append((sample_sf_gen, sample_ur_gen)) # ur is target
            else:
                X_train_s.append((sample_ur_gen, sample_sf_gen)) # sf is target
            y_train_s.append(0) # similar

            # Not Similar
            dist = []
            for o in emb_train:
                (o_ur_gen, o_sf_gen) = UR_SF_generalizer(o['ur'], o['sf'])
                if o_ur_gen != sample_ur_gen and o_sf_gen != sample_sf_gen:
                    if self.is_database_reversed:
                        levenshtein_distance = Levenshtein.distance(sample_ur_gen, o_ur_gen) # ur is target
                    else:
                        levenshtein_distance = Levenshtein.distance(sample_sf_gen, o_sf_gen) # sf is target
                    if levenshtein_distance > 0 and not o.get("paired", False):
                        dist.append({
                            "sample_example": sample["example"],
                            "sample_ur_gen" : sample_ur_gen,
                            "sample_sf_gen": sample_sf_gen,
                            "o_example": o["example"],
                            "o_ur_gen": o_ur_gen,
                            "o_sf_gen": o_sf_gen,
                            "dist": levenshtein_distance,
                            "source_sample": sample,
                            "target_sample": o
                        })

            if len(dist) == 0:
                logger.warn("No non-similar example is found in dist")
                continue

            selected_ns = random.choices(dist, k=1)
            
            if len(selected_ns) == 0:
                logger.warn("No non-similar example is found")
                continue            

            selected_ns = selected_ns[0]
            selected_ns["target_sample"]["paired"] = True # Mark the sample
            
            if self.is_database_reversed:
                # ur is target
                X_train_ns.append(
                    (sample_sf_gen, selected_ns["o_ur_gen"])
                )
            else:
                # sf is target
                X_train_ns.append(
                    (sample_ur_gen, selected_ns["o_sf_gen"])
                )
            y_train_ns.append(1) # not similar

        logger.info(f"Similar: {len(y_train_s)}, Not Similar: {len(y_train_ns)}")

        X_train = X_train_s + X_train_ns
        y_train = y_train_s + y_train_ns

        # Train embeddings model
        c2v_model = chars2vec.train_model(
            emb_dim=emb_dim,
            X_train=X_train,
            y_train=y_train,
            model_chars=chars
        )

        # Save the embeddings model in the model's directory
        path_to_model = self.get_chars2vec_model_reldirpath(cycle=self.cycle)
        chars2vec.save_model(c2v_model=c2v_model, path_to_model=path_to_model)

        # Input and prediction pairs, e.g., ('CiCEC=at', 'CiCECit')
        gen_pairs = [UR_SF_generalizer(
            self.strip_word(input_word, remove_zero=False),
            self.strip_word(pred_word, remove_zero=False)
        ) for input_word, pred_word in pairs]

        pred_words = [p[1] for p in gen_pairs] # use model's predictions

        # Convert words to embeddings vector
        X = c2v_model.vectorize_words(pred_words)

        # # Scale features
        scaler = StandardScaler()
        # # scaler = MinMaxScaler()
        Xs = scaler.fit_transform(X)

        # Project embeddings on plane
        Xd = PCA(n_components=self.args.pca_n_components).fit_transform(Xs) # or X with no scale

        # Find optimum number of clusters
        sse = []
        r = range(1, self.args.num_clusters + 1)
        for k in r:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=42
            )
            kmeans.fit(Xd)
            sse.append(kmeans.inertia_)
        kn = KneeLocator(
            r,
            sse,
            curve="convex",
            direction="decreasing"
        )

        logger.info(f"Best n_cluster: {kn.elbow}")


        # Clustering model
        model = KMeans(
            # n_clusters=self.args.num_clusters,
            n_clusters=kn.elbow,
            init=self.args.kmeans_init_method, # "k-means++" or "random",
            max_iter=1000,
            n_init=1
        )
        model.fit(Xd)
        predictions = model.predict(Xd)

        return predictions
        

    def save_next_pool(self, pool_samples: List, cycle: int):
        samples = [v["example"] for v in pool_samples if not v["sampled"]]
        if len(samples) == len(pool_samples):
            logger.info("No sample is extracted from the pool set.")
            return False
        self.save_samples("pool", cycle, samples)
        return True

    def save_next_train(self, train_samples: List, pool_samples: List, cycle: int):
        samples = [
            v["example"]
            for v in train_samples
        ] + [
            v["example"]
            for v in pool_samples
            if v["sampled"]
        ]
        if len(samples) == len(train_samples):
            logger.info("No sample is added to the train set.")
            return False
        self.save_samples("train", cycle, samples)
        return True


if __name__ == "__main__":
    # os.environ['MKL_SERVICE_FORCE_INTEL'] = '1' # NumPy using OpenMP
    os.environ['MKL_THREADING_LAYER'] = 'GNU' # PyTorch
    al = ActiveLearning()
