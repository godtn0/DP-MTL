import os

import pandas as pd
import pickle

import numpy as np

import torch
from torch.utils import data
from torch.utils.data import Dataset

from  datasets.data_utils import PadUptoEnd

from tqdm import tqdm


class ScoreDataset(Dataset):
    def __init__(self, theta, score) -> None:
        super().__init__()
        self.theta = theta.astype(np.float32)
        self.score = score.astype(np.float32)

    def __len__(self):
        return len(self.score)

    def __getitem__(self, index):
        return self.theta[index], self.score[index]


class ENEMWoleDataset(Dataset):
    def __init__(
        self,
        data_root,
        mapper_root,
        seq=False,
        split="train",
        max_seq_len=185,
        transforms=None,
        mean=None,
        std=None,
        num_q=185,
        demo=True,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.mapper_root = mapper_root
        self.mappers = self._load_mapper(mapper_root)
        self.max_seq_len = max_seq_len
        self.transforms = transforms
        self.mean = mean
        self.std = std
        self.seq = seq
        self.num_q = num_q
        self.demo = demo
        if seq:
            self._load_data_csv_for_seq(data_root)
        else:
            self._load_data_csv_for_sanpshot(data_root, split)

    def _load_mapper(self, mapper_root):
        mappers = {}
        mapper_key_list = ["qid_to_idx", "sid_to_idx", "idx_to_qid", "idx_to_sid"]
        for key in mapper_key_list:
            with open(mapper_root + "/" + key + ".json", "rb") as f:
                mappers[key] = pickle.load(f)
        return mappers

    def _load_data_csv_for_sanpshot(self, root_path, split):
        data_version = "enem_dep_" if self.demo else "enem_processed_"
        data_path = root_path + "/" + data_version + split + ".csv"
        self.df = pd.read_csv(data_path)

        self.df["score"] = (
            self.df["SCORE1"]
            + self.df["SCORE2"]
            + self.df["SCORE3"]
            + self.df["SCORE4"]
        )
        if self.mean is None:
            self.mean = self.df.groupby("ID-STUDENT").mean()["score"].mean()
            self.std = self.df.groupby("ID-STUDENT").mean()["score"].std()
        choice_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        self.df = self._reset_index(self.df)
        self._convert_choice_to_index(self.df, choice_to_index)
        self._redefine_id_(self.df, self.mappers["qid_to_idx"], "ID-QUESTION")
        from_cols = [
            "ID-QUESTION",
            "ID-STUDENT",
            "CORRECT?",
            "STUDENT-CHOICE",
            "CORRECT-CHOICE",
        ]
        to_cols = [
            "item_id",
            "user_id",
            "is_correct",
            "student_choice",
            "correct_choice",
        ]
        self._rename_cols(
            self.df,
            from_cols,
            to_cols,
        )
        self.drop_keys = [
            "user_id",
            "SCORE1",
            "SCORE2",
            "SCORE3",
            "SCORE4",
            "ES",
            "score",
        ]
        self._remove_cols(self.df, self.drop_keys[1:-1])
        self.user_ids = self.df["user_id"].unique()
        self.uids_to_idx = {}
        tmp = self.df.groupby("user_id")
        for uid, row in tmp:
            self.uids_to_idx[uid] = row.index

    def _load_data_csv_for_seq(self, root_path):
        data_version = "enem_dep_" if self.demo else "enem_processed_"

        train_path = root_path + "/" + data_version + "train.csv"
        val_path = root_path + "/" + data_version + "val.csv"
        test_path = root_path + "/" + data_version + "test.csv"
        self.train_df = pd.read_csv(train_path)
        self.val_df = pd.read_csv(val_path)
        self.test_df = pd.read_csv(test_path)

        self.train_df["score"] = (
            self.train_df["SCORE1"]
            + self.train_df["SCORE2"]
            + self.train_df["SCORE3"]
            + self.train_df["SCORE4"]
        )
        self.val_df["score"] = (
            self.val_df["SCORE1"]
            + self.val_df["SCORE2"]
            + self.val_df["SCORE3"]
            + self.val_df["SCORE4"]
        )
        self.test_df["score"] = (
            self.test_df["SCORE1"]
            + self.test_df["SCORE2"]
            + self.test_df["SCORE3"]
            + self.test_df["SCORE4"]
        )

        self.train_df = self._reset_index(self.train_df)
        self.val_df = self._reset_index(self.val_df)
        self.test_df = self._reset_index(self.test_df)

        self.mean = self.train_df.groupby("ID-STUDENT").mean()["score"].mean()
        self.std = self.train_df.groupby("ID-STUDENT").mean()["score"].std()

        choice_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        self._convert_choice_to_index(self.train_df, choice_to_index)
        self._convert_choice_to_index(self.val_df, choice_to_index)
        self._convert_choice_to_index(self.test_df, choice_to_index)

        self._redefine_id_(self.train_df, self.mappers["qid_to_idx"], "ID-QUESTION")
        self._redefine_id_(self.val_df, self.mappers["qid_to_idx"], "ID-QUESTION")
        self._redefine_id_(self.test_df, self.mappers["qid_to_idx"], "ID-QUESTION")

        self._redefine_id_(self.train_df, self.mappers["sid_to_idx"], "ID-STUDENT")
        self._redefine_id_(self.val_df, self.mappers["sid_to_idx"], "ID-STUDENT")
        self._redefine_id_(self.test_df, self.mappers["sid_to_idx"], "ID-STUDENT")

        from_cols = [
            "ID-QUESTION",
            "ID-STUDENT",
            "CORRECT?",
            "STUDENT-CHOICE",
            "CORRECT-CHOICE",
        ]
        to_cols = [
            "item_id",
            "user_id",
            "is_correct",
            "student_choice",
            "correct_choice",
        ]
        self._rename_cols(
            self.train_df,
            from_cols,
            to_cols,
        )
        self._rename_cols(
            self.val_df,
            from_cols,
            to_cols,
        )
        self._rename_cols(
            self.test_df,
            from_cols,
            to_cols,
        )
        self.drop_keys = [
            "user_id",
            "Unnamed: 0",
            "Unnamed: 0.1",
            "SCORE1",
            "SCORE2",
            "SCORE3",
            "SCORE4",
            "ES",
            "score",
        ]
        self._remove_cols(self.train_df, self.drop_keys[1:-1])
        self._remove_cols(self.val_df, self.drop_keys[1:-1])
        self._remove_cols(self.test_df, self.drop_keys[1:-1])
        self.user_ids = self.train_df["user_id"].unique()
        self.train_uids_to_idx = {}
        self.val_uids_to_idx = {}
        self.test_uids_to_idx = {}
        tmp = self.train_df.groupby("user_id")
        for uid, row in tmp:
            self.train_uids_to_idx[uid] = row.index
        tmp = self.val_df.groupby("user_id")
        for uid, row in tmp:
            self.val_uids_to_idx[uid] = row.index
        tmp = self.test_df.groupby("user_id")
        for uid, row in tmp:
            self.test_uids_to_idx[uid] = row.index

    def _reset_index(self, dataframe):
        new_df = dataframe.reset_index()
        new_df.pop("index")
        del dataframe
        return new_df

    def _convert_choice_to_index(self, dataframe, mapper):
        dataframe["STUDENT-CHOICE"] = dataframe["STUDENT-CHOICE"].map(
            mapper,
        )
        dataframe["CORRECT-CHOICE"] = dataframe["CORRECT-CHOICE"].map(
            mapper,
        )

    def _redefine_id_(self, dataframe, mapper, id_key):
        dataframe[id_key] = [mapper[id] for id in dataframe[id_key]]

    def _rename_cols(self, dataframe, from_cols, to_cols):
        for i, key in enumerate(from_cols):
            dataframe[to_cols[i]] = dataframe.pop(key)

    def _remove_cols(self, dataframe, cols):
        for key in cols:
            if key in dataframe.keys():
                dataframe.pop(key)

    def __len__(self):
        return len(self.user_ids)

    def _get_snap(self, index):
        user_id = self.user_ids[index]
        df = self.df.loc[self.uids_to_idx[user_id]]

        score = (df["score"].mean() - self.mean) / (self.std + 1e-7)

        sample = {
            "score": score,
            "user_id": user_id,
            "interactions": df.to_records(index=False),
        }
        if self.transforms is not None:
            interactions = sample["interactions"]
            for transform in self.transforms:
                interactions = transform(interactions)
                sample.update({"interactions": interactions})
        return sample

    def _get_seq(self, index):
        user_id = self.user_ids[index]
        train_df = self.train_df.loc[self.train_uids_to_idx[user_id]]
        val_df = self.val_df.loc[self.val_uids_to_idx[user_id]]
        test_df = self.test_df.loc[self.test_uids_to_idx[user_id]]

        score = (train_df["score"].mean() - self.mean) / (self.std + 1e-7)

        train_df.pop("score")
        train_df.pop("user_id")
        val_df.pop("score")
        val_df.pop("user_id")
        test_df.pop("score")
        test_df.pop("user_id")

        sample = {
            "score": score,
            "user_id": user_id,
            "train_interactions": train_df.to_records(index=False),
            "val_interactions": val_df.to_records(index=False),
            "test_interactions": test_df.to_records(index=False),
        }
        if self.transforms is not None:
            train_interactions = sample["train_interactions"]
            val_interactions = sample["val_interactions"]
            test_interactions = sample["test_interactions"]
            for transform in self.transforms:
                train_interactions = transform(train_interactions)
                val_interactions = transform(val_interactions)
                test_interactions = transform(test_interactions)
                sample.update({"train_interactions": train_interactions})
                sample.update({"val_interactions": val_interactions})
                sample.update({"test_interactions": test_interactions})
        return sample

    def __getitem__(self, index):
        if self.seq:
            return self._get_seq(index)
        else:
            return self._get_snap(index)


class ENEMDataset(Dataset):
    def __init__(
        self,
        data_path=None,
        mapper_root=None,
        dataset_dict=None,
        transforms=None,
        mean=None,
        std=None,
        num_q=185,
        max_seq_len=185,
    ) -> None:
        super().__init__()
        self.transforms = transforms
        self.num_q = num_q
        self.max_seq_len = max_seq_len
        self.mean = mean
        self.std = std
        if mapper_root is not None:
            self.mappers = self._load_mapper(mapper_root)
        if data_path is not None:
            self._set_up_from_data_pth(data_path)
        elif dataset_dict is not None:
            self._set_up_from_data_dict(dataset_dict, mean, std)

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
        enem_df = pd.read_csv(data_path)
        choice_to_index = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        enem_df = self._remove_invalid_choice(enem_df, list(choice_to_index.keys()))
        enem_df["STUDENT-CHOICE"] = enem_df["STUDENT-CHOICE"].map(
            {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
        )
        enem_df["CORRECT-CHOICE"] = enem_df["CORRECT-CHOICE"].map(
            {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4},
        )
        enem_df = self._redefine_id_(enem_df, self.mappers["qid_to_idx"], "ID-QUESTION")
        enem_df = self._redefine_id_(enem_df, self.mappers["sid_to_idx"], "ID-STUDENT")

        enem_df["score"] = (
            enem_df["SCORE1"]
            + enem_df["SCORE2"]
            + enem_df["SCORE3"]
            + enem_df["SCORE4"]
        )
        if self.mean is None:
            self.mean = enem_df["score"].mean()
            self.std = enem_df["score"].std()

        enem_df["item_id"] = enem_df.pop("ID-QUESTION")
        enem_df["user_id"] = enem_df.pop("ID-STUDENT")
        enem_df["is_correct"] = np.where(enem_df.pop("CORRECT?"), 1, 0)
        enem_df["student_choice"] = enem_df.pop("STUDENT-CHOICE")
        enem_df["correct_choice"] = enem_df.pop("CORRECT-CHOICE")

        unique_user_id = np.unique(enem_df["user_id"])
        self.max_seq_len = -1

        for user_id in tqdm(unique_user_id):
            df = enem_df[enem_df["user_id"] == user_id]
            if len(df) > self.max_seq_len:
                self.max_seq_len = len(df)
            score = (df["score"].mean() - self.mean) / (self.std + 1e-7)
            drop_keys = [
                "user_id",
                "SCORE1",
                "SCORE2",
                "SCORE3",
                "SCORE4",
                "ES",
                "score",
            ]
            for key in drop_keys:
                df.pop(key)
            self.sample_list.append(
                {
                    "score": score,
                    "user_id": user_id,
                    "interactions": df.to_records(index=False),
                }
            )

    def _set_up_from_data_dict(self, dict, mean=None, std=None):
        self.sample_list = dict
        self.mean = mean
        self.std = std

    def _remove_invalid_choice(self, dataframe, valid_choices):
        new_df = dataframe[dataframe["STUDENT-CHOICE"].isin(valid_choices)]
        new_df = new_df.reset_index()
        new_df.pop("index")
        return new_df

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
        user_data["train_interactions"] = data_lists[3][i]["interactions"]
        if split[2] > 0:
            user_data["val_interactions"] = data_lists[2][i]["interactions"]
        if split[0] > 0:
            user_data["test_interactions"] = data_lists[0][i]["interactions"]
        user_data["user_id"] = data_lists[0][i]["user_id"]
        user_data["score"] = data_lists[0][i]["score"]
        merged_list.append(user_data)
    return merged_list


def divide_interactions_per_user(dataset, ratio, seed=None):
    divided_list = [[] for _ in range(len(ratio))]
    for data in dataset:
        inters = data["interactions"]
        num_inter_per_user = len(inters)
        item_idx = np.arange(num_inter_per_user)

        for i, r in enumerate(ratio):
            if r == 0:
                continue
            if seed is not None:
                if i == 0:
                    np.random.seed(0)
                elif i == 1:
                    np.random.seed(seed)
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


def split_enem_for_kt(
    dataset, split, drop_ratio, transforms=None, seed=None, seq=False
):
    datasets = {}

    split = [split[1], drop_ratio, split[0], 1.0 - drop_ratio - split[0] - split[1]]
    data_split_list = divide_interactions_per_user(dataset, split, seed=seed)
    if seq:
        merged_list = merge_by_user(data_split_list, split)
        datasets["train"] = ENEMDataset(
            dataset_dict=merged_list,
            transforms=transforms,
            mean=dataset.mean,
            std=dataset.std,
        )
        del dataset
        return datasets
    if split[0] > 0:
        datasets["test"] = ENEMDataset(
            dataset_dict=data_split_list[0],
            transforms=transforms,
            mean=dataset.mean,
            std=dataset.std,
        )
    if split[2] > 0:
        datasets["val"] = ENEMDataset(
            dataset_dict=data_split_list[2],
            transforms=transforms,
            mean=dataset.mean,
            std=dataset.std,
        )
    if split[3] > 0:
        datasets["train"] = ENEMDataset(
            dataset_dict=data_split_list[3],
            transforms=transforms,
            mean=dataset.mean,
            std=dataset.std,
        )
    del dataset
    return datasets


def split_enem_for_sp(dataset, split, drop_ratio, transforms=None, seed=None):
    datasets = {}
    drop_inters = None
    if drop_ratio > 0.0 and drop_ratio < 1.0:
        drop_ratio = [0.1, 1 - drop_ratio - 0.1, drop_ratio]
        divided_list = divide_interactions_per_user(dataset, drop_ratio, seed)
        keep_inters = np.array(divided_list[1])
        test_inters = np.array(divided_list[0])
    else:
        drop_ratio = [0.1, 0.9]
        divided_list = divide_interactions_per_user(dataset, drop_ratio, seed)
        keep_inters = np.array(divided_list[1])
        test_inters = np.array(divided_list[0])
    split = [split[0], 1 - split[0]]
    num_users = len(keep_inters)
    user_idx = np.arange(num_users)

    train_split = int(num_users * split[0])
    train_split_indices = user_idx[:train_split]
    train_list = keep_inters[train_split_indices].tolist()

    datasets["train"] = ENEMDataset(
        dataset_dict=train_list,
        transforms=transforms,
        mean=dataset.mean,
        std=dataset.std,
    )

    val_split_indices = user_idx[train_split:]
    val_list = keep_inters[val_split_indices].tolist()
    datasets["val"] = ENEMDataset(
        dataset_dict=val_list,
        transforms=transforms,
        mean=dataset.mean,
        std=dataset.std,
    )

    if test_inters is not None:
        test_list = test_inters[val_split_indices].tolist()
        datasets["test"] = ENEMDataset(
            dataset_dict=test_list,
            transforms=transforms,
            mean=dataset.mean,
            std=dataset.std,
        )
    return datasets


def enem_dataset(cfg=None):
    datasets = {}
    if cfg.data.pad:
        pad_value_dict = {
            "item_id": "0",
            "is_correct": 0,
            "student_choice": -1,
            "correct_choice": -1,
        }
        transforms = [PadUptoEnd(185, pad_value_dict)]
    else:
        transforms = None
    if cfg.data.demo == False:
        if cfg.data.seq:
            datasets["train"] = ENEMWoleDataset(
                cfg.data.root,
                cfg.data.mapper,
                seq=True,
                transforms=transforms,
                demo=cfg.data.demo,
            )
            return datasets
        else:
            datasets["train"] = ENEMWoleDataset(
                cfg.data.root, cfg.data.mapper, split="train", demo=cfg.data.demo
            )
            datasets["val"] = ENEMWoleDataset(
                cfg.data.root,
                cfg.data.mapper,
                split="val",
                mean=datasets["train"].mean,
                std=datasets["train"].std,
                demo=cfg.data.demo,
            )
            datasets["test"] = ENEMWoleDataset(
                cfg.data.root,
                cfg.data.mapper,
                split="val",
                mean=datasets["train"].mean,
                std=datasets["train"].std,
                demo=cfg.data.demo,
            )
            return datasets

    data_path = cfg.data.root + "/enem_dep.csv"
    dataset = ENEMDataset(data_path, mapper_root=cfg.data.mapper)

    if cfg is not None and cfg.split:
        np.random.seed(cfg.data.drop_seed)
        if cfg.type == "sp":
            datasets = split_enem_for_sp(
                dataset,
                split=cfg.data.split.sp,
                drop_ratio=cfg.data.drop_ratio,
                transforms=transforms,
                seed=cfg.data.drop_seed,
            )
            if len(datasets) <= 2:
                datasets["test"] = datasets["val"]
        elif cfg.type == "mix":
            datasets = split_enem_for_kt(
                dataset,
                split=cfg.data.split.mix,
                drop_ratio=cfg.data.drop_ratio,
                transforms=transforms,
                seed=cfg.data.drop_seed,
                seq=cfg.data.seq,
            )
    else:
        datasets["train"] = dataset
    return datasets