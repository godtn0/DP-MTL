import os

import pickle

import pandas as pd

import numpy as np

import torch
from torch.utils import data
from torch.utils.data import Dataset, ConcatDataset

from  datasets.data_utils import PadUptoEnd



class ToeicScoreDataset(Dataset):
    def __init__(
        self,
        data_path=None,
        dataset_dict=None,
        mapper_root=None,
        mean=None,
        std=None,
        transforms=None,
        max_seq_len=None,
        num_q=None,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.mean = mean
        self.std = std
        self.num_q = num_q
        if mapper_root is not None:
            self.mappers = self._load_mapper(mapper_root)
        if data_path is not None:
            self._set_up_from_data_pth(data_path)
        elif dataset_dict is not None:
            self._set_up_from_data_dict(dataset_dict, max_seq_len, num_q)

    def _load_mapper(self, mapper_root):
        mappers = {}
        mapper_key_list = ["qid_to_idx", "sid_to_idx", "idx_to_qid", "idx_to_sid"]
        for key in mapper_key_list:
            with open(mapper_root + "/" + key + ".json", "rb") as f:
                mappers[key] = pickle.load(f)
        return mappers

    def _set_up_from_data_pth(self, data_path):
        if not os.path.exists(data_path):
            assert "there is no data_path. plz check the data_path."

        self.sample_list = []
        toeic_df = pd.read_csv(data_path)

        toeic_df["student_choice"] = toeic_df.pop("user_answer") - 1
        toeic_df["correct_choice"] = toeic_df.pop("correct_answer") - 1
        if self.num_q is None:
            self.num_q = np.max(list(self.mappers["idx_to_qid"].keys())) + 1
        toeic_df = self._redefine_id_(toeic_df, self.mappers["qid_to_idx"], "item_id")
        toeic_df = self._redefine_id_(toeic_df, self.mappers["sid_to_idx"], "sample_id")
        if "new_id" in toeic_df.keys():
            toeic_df.pop("new_id")
        test = True
        if self.mean is None:
            self.mean = toeic_df["score"].mean()
            if test:
                self.mean = toeic_df.groupby("sample_id").mean()["score"].mean()
        if self.std is None:
            self.std = toeic_df["score"].std()
            if test:
                self.std = toeic_df.groupby("sample_id").mean()["score"].std()

        unique_sample_id = np.unique(toeic_df["sample_id"])
        self.max_seq_len = -1
        for sample_id in unique_sample_id:
            df = toeic_df[toeic_df["sample_id"] == sample_id]
            if len(df) > self.max_seq_len:
                self.max_seq_len = len(df)
            score = (df["score"].mean() - self.mean) / (self.std + 1e-7)
            drop_keys = ["sample_id", "TOEIC_LC", "TOEIC_RC", "score"]
            for key in drop_keys:
                df.pop(key)
            self.sample_list.append(
                {
                    "score": score,
                    "user_id": sample_id,
                    "interactions": df.to_records(index=False),
                }
            )

    def _set_up_from_data_dict(self, dict, max_seq_len, num_q):
        self.max_seq_len = max_seq_len
        self.num_q = num_q
        self.sample_list = dict

    def _redefine_id_(self, dataframe, mapper, id_key):
        dataframe[id_key] = [mapper[id] for id in dataframe[id_key]]
        return dataframe

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index: int):
        sample = self.sample_list[index].copy()
        if self.transforms is not None:
            key_list = list(sample.keys())
            for key in key_list:
                if "interactions" in key:
                    interactions = sample[key]
                    for transform in self.transforms:
                        interactions = transform(interactions)
                    sample.update({key: interactions})
        return sample


def merge_by_user(data_lists, split):
    merged_list = []
    for i in range(len(data_lists[0])):
        user_data = {}
        user_data["train_interactions"] = data_lists[2][i]["interactions"]
        if split[1] > 0:
            user_data["val_interactions"] = data_lists[1][i]["interactions"]
        if split[0] > 0:
            user_data["test_interactions"] = data_lists[0][i]["interactions"]
        user_data["user_id"] = data_lists[0][i]["user_id"]
        user_data["score"] = data_lists[0][i]["score"]
        merged_list.append(user_data)
    return merged_list


def divide_interactions_per_user(dataset, ratio):
    divided_list = [[] for _ in range(len(ratio))]
    for data in dataset:
        inters = data["interactions"]
        num_inter_per_user = len(inters)
        item_idx = np.arange(num_inter_per_user)

        for i, r in enumerate(ratio):
            if r == 0:
                continue
            if i == len(ratio) - 1:
                divided_list[i].append(
                    {
                        "interactions": inters[item_idx],
                        "user_id": data["user_id"],
                        "score": data["score"],
                    }
                )
            else:
                split = int(num_inter_per_user * r)
                np.random.shuffle(item_idx)
                split_indices = item_idx[:split]
                inters_split = inters[split_indices]
                divided_list[i].append(
                    {
                        "interactions": inters_split,
                        "user_id": data["user_id"],
                        "score": data["score"],
                    }
                )
                item_idx = item_idx[split:]
    return divided_list


def split_toeic_score_for_kt(
    dataset, split, mean, std, max_seq_len, num_q, transforms=None, seq=False
):
    datasets = {}

    split = [split[1], split[0], 1.0 - split[0] - split[1]]
    data_split_list = divide_interactions_per_user(dataset, split)
    if seq:
        merged_list = merge_by_user(data_split_list, split)
        datasets["train"] = ToeicScoreDataset(
            dataset_dict=merged_list,
            transforms=transforms,
            mean=mean,
            std=std,
            max_seq_len=max_seq_len,
            num_q=num_q,
        )
        del dataset
        return datasets
    if split[0] > 0:
        datasets["test"] = ToeicScoreDataset(
            dataset_dict=data_split_list[0],
            transforms=transforms,
            mean=mean,
            std=std,
        )
    if split[1] > 0:
        datasets["val"] = ToeicScoreDataset(
            dataset_dict=data_split_list[1],
            transforms=transforms,
            mean=mean,
            std=std,
        )
    if split[2] > 0:
        datasets["train"] = ToeicScoreDataset(
            dataset_dict=data_split_list[2],
            transforms=transforms,
            mean=mean,
            std=std,
        )
    return datasets


def toeic_score_dataset(cfg, test=True):
    np.random.seed(0)
    root_path = cfg.data.root
    if cfg.data.version in ['10', "25","50"]:
        train_path = (
            root_path
            + "/"
            + cfg.data.version
            + "/train_processed_"
            + str(cfg.data.max_seq_len)
            + ".csv"
        )
    else:
        train_path = (
            root_path
            + "/"
            + cfg.data.version
            + "/train_processed.csv"
        )
    dataset = ToeicScoreDataset(
        train_path,
        mapper_root=cfg.data.mapper,
    )
    if cfg.data.pad:
        pad_value_dict = {
            "item_id": "0",
            "is_correct": 0,
            "student_choice": -1,
            "correct_choice": -1,
        }
        transforms = [PadUptoEnd(dataset.max_seq_len, pad_value_dict)]
    else:
        transforms = None
    datasets = {}
    if cfg is not None and cfg.split:
        if cfg.type == "sp":
            datasets["train"] = dataset
        elif cfg.type == "mix":
            if cfg.data.version in ['10', "25","50"]:
                test_path = (
                    root_path
                    + "/"
                    + cfg.data.version
                    + "/test_processed_"
                    + str(cfg.data.max_seq_len)
                    + ".csv"
                )
            else:
                test_path = (
                    root_path
                    + "/"
                    + cfg.data.version
                    + "/test_processed.csv"
                )
            test_dataset = ToeicScoreDataset(
                data_path=test_path,
                mean=dataset.mean,
                std=dataset.std,
                mapper_root=cfg.data.mapper,
            )
            concat_dataset = ConcatDataset([dataset, test_dataset])
            max_seq_len = (
                dataset.max_seq_len
                if dataset.max_seq_len > test_dataset.max_seq_len
                else test_dataset.max_seq_len
            )
            num_q = (
                dataset.num_q
                if dataset.num_q > test_dataset.num_q
                else test_dataset.num_q
            )
            transforms[0].max_seq = max_seq_len
            datasets = split_toeic_score_for_kt(
                concat_dataset,
                cfg.data.split.mix,
                dataset.mean,
                dataset.std,
                transforms=transforms,
                seq=cfg.data.seq,
                max_seq_len=max_seq_len,
                num_q=num_q,
            )
    else:
        datasets["train"] = dataset

    if test:
        dataset = ToeicScoreDataset(
            data_path=root_path + "/" + cfg.data.version + "/test_processed.csv",
            mean=datasets["train"].mean,
            std=datasets["train"].std,
        )
        divided_list = divide_interactions_per_user(dataset, ratio=[0.1, 1.0 - 0.1])
        datasets["val"] = ToeicScoreDataset(
            dataset_dict=divided_list[1],
            mean=datasets["train"].mean,
            std=datasets["train"].std,
        )
        datasets["test"] = ToeicScoreDataset(
            dataset_dict=divided_list[0],
            mean=datasets["train"].mean,
            std=datasets["train"].std,
        )
    return datasets


