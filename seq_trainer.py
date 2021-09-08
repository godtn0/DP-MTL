from collections import UserDict
import multiprocessing
import os

import pickle

from datetime import datetime

import torch
import pytorch_lightning as pl
from torch._C import device
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader
from torch.functional import Tensor

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn import linear_model
import sklearn.metrics as metrics
from scipy import stats

import wandb

from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf

from datasets.enem_dataset import (
    enem_dataset,
    ScoreDataset,
)
from datasets.data_utils import enem_collate_fn
from datasets.toeic_dataset import toeic_score_dataset
from models.irt_model import (
    IRTScoreGenerator,
    LinearRegression,
    EarlyStopping,
    SequentialIRT,
)
from research_models.utils import save_cfg

from blink.data.transforms import PadUptoEnd


class IRTTrainer:
    def __init__(self, cfg, log_path) -> None:
        self.cfg = cfg
        self.log_path = log_path
        if cfg.data.dataset_type == "toeic":
            sp = cfg.type == "sp"
            self.datasets = toeic_score_dataset(cfg, test=sp)
        elif cfg.data.dataset_type == "enem":
            import time

            start = time.time()  
            print("dataset preprocessing start...")
            self.datasets = enem_dataset(
                cfg=cfg,
            )
            print("time :", time.time() - start)  
        self.dataloaders = self._build_data_loader(self.datasets, enem_collate_fn)
        self.model = self._build_model()
        if self.cfg.train.early_stopping:
            self.early_stopping_loss = EarlyStopping(
                metric="loss",
                patience=self.cfg.early_stopping.patience,
                dir=self.log_path,
            )
            self.early_stopping_auc = EarlyStopping(
                metric="AUC",
                patience=self.cfg.early_stopping.patience,
                dir=self.log_path,
            )

    def _build_data_loader(self, datasets, collate_fn):
        data_loaders = {}
        for key in datasets:
            shuffle = True if key == "train" else False
            data_loaders[key] = DataLoader(
                datasets[key],
                batch_size=self.cfg.batch_size,
                collate_fn=collate_fn,
                shuffle=shuffle,
            )
        return data_loaders

    def _build_model(self):
        if self.cfg.data.dataset_type == "toeic":
            model = SequentialIRT(
                self.datasets["train"].num_q,
                self.cfg.train.num_d,
                self.cfg.train.num_layers,
                self.cfg.train.num_opt,
                max_seq_len=self.cfg.data.max_seq_len,
            )
        else:
            model = SequentialIRT(
                self.datasets["train"].num_q,
                self.cfg.train.num_d,
                self.cfg.train.num_layers,
                self.cfg.train.num_opt,
                max_seq_len=self.cfg.num_q,
            )
        return model

    def _build_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), self.cfg.train.lr)

    def _build_loss(self):
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.bce_loss = nn.BCELoss(reduction="sum")
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def train(self, epochs=None):
        if epochs is None:
            epochs = self.cfg.train.max_epochs
        self._build_optimizer()
        self._build_loss()
        self.model.train()
        self.model.to(device="cuda")
        early_stop_loss = False
        early_stop_auc = False

        best_metrics = {}
        for i in range(epochs):
            logs = {}
            train_losses = []
            train_kt_losses = []
            train_ct_losses = []
            train_probs = []
            train_logits = []
            train_sans = []
            train_user_choices = []
            for data in tqdm(self.dataloaders["train"]):
                N, L = (
                    data["train_interactions"]["is_correct"].size()[0],
                    data["train_interactions"]["is_correct"].size()[1],
                )
                train_logs = self._train_step(data)
                train_losses.append(train_logs["loss"])
                train_kt_losses.append(train_logs["kt_loss"])
                train_ct_losses.append(train_logs["ct_loss"])
                for i in range(N):
                    train_probs += (
                        train_logs["prob"][i][
                            : data["train_interactions"]["sequence_size"][i]
                        ]
                        .detach()
                        .numpy()
                        .tolist()
                    )
                    train_logits.append(
                        train_logs["logits"][i][
                            : data["train_interactions"]["sequence_size"][i]
                        ]
                        .detach()
                        .numpy()
                    )
                    train_sans += (
                        data["train_interactions"]["is_correct"][i][
                            : data["train_interactions"]["sequence_size"][i]
                        ]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                    train_user_choices += (
                        data["train_interactions"]["student_choice"][i][
                            : data["train_interactions"]["sequence_size"][i]
                        ]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
            logs["train_loss"] = np.average(train_losses)
            logs["train_kt_loss"] = np.average(train_kt_losses)
            logs["train_ct_loss"] = np.average(train_ct_losses)
            train_logits = np.concatenate(train_logits, axis=0)
            train_choices_logits = np.argmax(train_logits, axis=1)
            train_compare = np.where(train_choices_logits == train_user_choices, 1, 0)
            train_ct_correct = np.count_nonzero(train_compare)
            logs["train_ct_acc"] = train_ct_correct / len(train_compare)

            fpr, tpr, thresholds = metrics.roc_curve(
                train_sans, train_probs, pos_label=1
            )
            logs["train_kt_auc"] = metrics.auc(fpr, tpr)
            train_probs = np.array(train_probs)
            train_probs = np.where(train_probs > 0.5, 1, 0)
            logs["train_kt_acc"] = metrics.accuracy_score(train_probs, train_sans)

            if self.cfg.type == "mix" and self.cfg.data.split.mix[0] > 0:
                val_mix_logs = self._evaluate_ct(self.dataloaders["train"])
                if len(best_metrics) == 0:
                    for key in val_mix_logs.keys():
                        best_metrics["val/best_" + key] = val_mix_logs[key]
                else:
                    for key in val_mix_logs.keys():
                        if "loss" in key or "mae" in key:
                            if best_metrics["val/best_" + key] > val_mix_logs[key]:
                                best_metrics["val/best_" + key] = val_mix_logs[key]
                        else:
                            if best_metrics["val/best_" + key] < val_mix_logs[key]:
                                best_metrics["val/best_" + key] = val_mix_logs[key]
                key_lst = list(val_mix_logs.keys())
                for key in key_lst:
                    val_mix_logs["val/" + key] = val_mix_logs.pop(key)

                logs.update(best_metrics)
                logs.update(val_mix_logs)
            if (
                self.cfg.train.early_stopping
                and self.cfg.type != "sp"
                and self.cfg.data.split.mix[0] > 0
            ):
                if not early_stop_loss:
                    self.early_stopping_loss(logs["val/loss"], self.model)
                    early_stop_loss = self.early_stopping_loss.early_stop

                if not early_stop_auc:
                    self.early_stopping_auc(logs["val/kt_auc"], self.model)
                    early_stop_auc = self.early_stopping_auc.early_stop
                if early_stop_loss and early_stop_auc:
                    break

    def _train_step(self, data, model=None, optimizer=None):
        if model is None:
            model = self.model
        if optimizer is None:
            optimizer = self.optimizer
        key_list = data["train_interactions"].keys()
        for key in key_list:
            data["train_interactions"][key] = data["train_interactions"][key].to(
                device="cuda"
            )

        # data = data.to(device="cuda")
        optimizer.zero_grad()
        logits, prob_from_dirt, prob_from_pirt, _ = model(data["train_interactions"])
        N, L = (
            data["train_interactions"]["student_choice"].size()[0],
            data["train_interactions"]["student_choice"].size()[1],
        )
        ct_loss = self.ce_loss(
            logits.view(N * L, -1),
            data["train_interactions"]["student_choice"].view(N * L),
        )
        ####### directly cacluate propbability of correctness through sigomid ######
        kt_loss_from_dirt = self.bce_loss(
            prob_from_dirt,
            (
                data["train_interactions"]["is_correct"]
                * data["train_interactions"]["pad_mask"]
            ).float(),
        )
        ############################################################################
        kt_loss_from_pirt = self.bce_loss(
            prob_from_pirt,
            (
                data["train_interactions"]["is_correct"]
                * data["train_interactions"]["pad_mask"]
            ).float(),
        )
        if self.cfg.train.kt_only:
            kt_loss = kt_loss_from_dirt
            prob = prob_from_dirt
        else:
            kt_loss = kt_loss_from_pirt
            prob = prob_from_pirt
        num_samples = torch.sum(data["train_interactions"]["sequence_size"])
        kt_loss = kt_loss / num_samples
        loss = (
            self.cfg.train.mixer_lambda * kt_loss
            + (1 - self.cfg.train.mixer_lambda) * ct_loss
        )
        loss.backward()
        optimizer.step()
        logs = {
            "loss": loss.detach().item(),
            "kt_loss": kt_loss.detach().item(),
            "ct_loss": ct_loss.detach().item(),
            "logits": logits.cpu(),
            "prob": prob.cpu(),
        }
        return logs

    def evaluate(self):
        if self.cfg.type == "sp":
            logs = self._evaluate_sp(test=True)
        elif self.cfg.type == "mix":
            if self.early_stopping_loss.best_model_path is not None:
                self.model.load_state_dict(
                    torch.load(self.early_stopping_loss.best_model_path)
                )
            if self.cfg.data.split.mix[1] > 0:
                test_logs = self._evaluate_ct(self.dataloaders["train"], test=True)
                key_list = list(test_logs.keys())
                for key in key_list:
                    test_logs["eval/test_" + key] = test_logs.pop(key)
            sp_logs = self._evaluate_sp(test=True)

            if self.early_stopping_auc.best_model_path is not None:
                self.model.load_state_dict(
                    torch.load(self.early_stopping_auc.best_model_path)
                )
            else:
                return
            if self.cfg.data.split.mix[1] > 0:
                test_logs = self._evaluate_ct(self.dataloaders["train"], test=True)
                key_list = list(test_logs.keys())
                for key in key_list:
                    test_logs["eval/test_" + key + "_fa"] = test_logs.pop(key)
            sp_logs = self._evaluate_sp(test=True)
            key_list = list(sp_logs.keys())
            for key in key_list:
                sp_logs[key + "_fa"] = sp_logs.pop(key)

    def _evaluate_ct(self, dataloader, test=False, model=None):
        if model is None:
            model = self.model
        if test:
            val_key = "test_interactions"
        else:
            val_key = "val_interactions"
        model.to(device="cuda")
        logs = {}
        ct_losses = []
        kt_losses = []
        logits = []
        probs = []
        anss = []
        s_choices = []
        user_states = []
        user_ids = []
        for i, data in enumerate(dataloader):
            key_list = data["train_interactions"].keys()
            for key in key_list:
                data["train_interactions"][key] = data["train_interactions"][key].to(
                    device="cuda"
                )
                data[val_key][key] = data[val_key][key].to(device="cuda")

            _, _, _, user_state = model(data["train_interactions"])

            logit, prob_from_dirt, prob_from_pirt, _ = model(
                (data[val_key], user_state), eval=True
            )
            N, L = (
                data[val_key]["student_choice"].size()[0],
                data[val_key]["student_choice"].size()[1],
            )
            ct_loss = self.ce_loss(
                logit.view(N * L, -1),
                (data[val_key]["student_choice"]).view(N * L),
            )
            kt_loss_from_pirt = self.bce_loss(
                prob_from_pirt,
                (data[val_key]["is_correct"] * data[val_key]["pad_mask"]).float(),
            )
            kt_loss_from_dirt = self.bce_loss(
                prob_from_dirt,
                (data[val_key]["is_correct"] * data[val_key]["pad_mask"]).float(),
            )
            if self.cfg.train.kt_only:
                kt_loss = kt_loss_from_dirt
                prob = prob_from_dirt
            else:
                kt_loss = kt_loss_from_pirt
                prob = prob_from_pirt
            num_samples = torch.sum(data[val_key]["sequence_size"])
            kt_loss = kt_loss / num_samples
            ct_losses.append(ct_loss.cpu().item())
            kt_losses.append(kt_loss.cpu().item())
            for i in range(N):
                logits.append(
                    logit[i][: data[val_key]["sequence_size"][i]].cpu().detach().numpy()
                )
                probs += (
                    prob[i][: data[val_key]["sequence_size"][i]]
                    .cpu()
                    .detach()
                    .numpy()
                    .tolist()
                )
                s_choices += (
                    data[val_key]["student_choice"][i][
                        : data[val_key]["sequence_size"][i]
                    ]
                    .cpu()
                    .numpy()
                    .tolist()
                )
                anss += (
                    data[val_key]["is_correct"][i][: data[val_key]["sequence_size"][i]]
                    .cpu()
                    .numpy()
                    .tolist()
                )

        logs["ct_loss"] = np.average(ct_losses)
        logs["kt_loss"] = np.average(kt_losses)
        logs["loss"] = (
            self.cfg.train.mixer_lambda * logs["kt_loss"]
            + (1 - self.cfg.train.mixer_lambda) * logs["ct_loss"]
        )
        logits = np.concatenate(logits, axis=0)

        fpr, tpr, thresholds = metrics.roc_curve(anss, probs, pos_label=1)
        logs["kt_auc"] = metrics.auc(fpr, tpr)
        probs = np.array(probs)
        probs = np.where(probs > 0.5, 1, 0)
        logs["kt_acc"] = metrics.accuracy_score(probs, anss)
        choices_logits = np.argmax(logits, axis=1)
        compare = np.where(choices_logits == s_choices, 1, 0)
        ct_correct = np.count_nonzero(compare)
        logs["ct_acc"] = ct_correct / len(compare)

        return logs

    def _evaluate_sp(self, model=None, test=False):
        if model is None:
            model = self.model
        logs = {}

        user_states = []
        user_scores = []
        for i, data in enumerate(self.dataloaders["train"]):
            key_list = data["train_interactions"].keys()
            for key in key_list:
                data["train_interactions"][key] = data["train_interactions"][key].to(
                    device="cuda"
                )

            _, _, _, user_state = model(data["train_interactions"])
            user_states.append(user_state.cpu().detach().numpy())
            user_scores.append(data["score"].numpy())

        user_states = np.concatenate(user_states, axis=0)
        user_scores = np.concatenate(user_scores, axis=0)

        indices = np.arange(len(user_states))
        np.random.shuffle(indices)
        train_indices = indices[: int(0.9 * len(user_states))]
        train_user_states = user_states[train_indices]
        train_user_scores = user_scores[train_indices]
        train_sp_dataset = ScoreDataset(train_user_states, train_user_scores)
        train_sp_data_loader = DataLoader(
            train_sp_dataset, batch_size=256, shuffle=True
        )

        val_indices = indices[int(0.9 * len(user_states)) :]
        val_user_states = user_states[val_indices]
        val_user_scores = user_scores[val_indices]
        val_sp_dataset = ScoreDataset(val_user_states, val_user_scores)
        val_sp_data_loader = DataLoader(val_sp_dataset, batch_size=256, shuffle=False)

        score_predictor = LinearRegression(num_d=self.cfg.train.num_d, num_scores=1)
        score_predictor_iso = IRTScoreGenerator()
        score_predictor_sklearn = linear_model.LinearRegression().fit(
            train_user_states,
            train_user_scores * (self.datasets["train"].std + 1e-7)
            + self.datasets["train"].mean,
        )

        mse = nn.MSELoss()
        optimizer = optim.Adam(score_predictor.parameters(), lr=0.03)
        score_predictor.train()
        score_predictor.to(device="cuda")
        for epoch in range(70):
            train_sp_logs = {}
            losses = []
            for data in train_sp_data_loader:
                N = data[0].size()[0]
                optimizer.zero_grad()
                pred_score = score_predictor(data[0].to(device="cuda")).view(N)
                loss = mse(pred_score, data[1].to(device="cuda"))
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().cpu().item())
            train_sp_logs["train_sp_loss"] = np.average(losses)

            train_pred_scores = []
            train_real_scores = []
            train_thetas = []
            for data in train_sp_data_loader:
                N = data[0].size()[0]
                pred_score = score_predictor(data[0].to(device="cuda")).view(N)
                train_pred_scores += pred_score.detach().cpu().numpy().tolist()
                train_real_scores += data[1].numpy().tolist()
                train_thetas += data[0].numpy().tolist()
            train_pred_scores = (
                np.array(train_pred_scores) * (self.datasets["train"].std + 1e-7)
                + self.datasets["train"].mean
            )
            train_real_scores = (
                np.array(train_real_scores) * (self.datasets["train"].std + 1e-7)
                + self.datasets["train"].mean
            )
            train_sp_logs["train_mae"] = np.average(
                np.abs(train_pred_scores - train_real_scores)
            )

            if self.cfg.train.num_d == 1:
                theta_for_irt = np.squeeze(np.array(train_thetas))
            else:
                theta_for_irt = train_pred_scores
            score_predictor_iso.optimize([theta_for_irt, train_real_scores])
            iso_predicted_train_score = score_predictor_iso(theta_for_irt)
            train_sp_logs["train_iso_mae"] = np.average(
                np.abs(iso_predicted_train_score - train_real_scores)
            )
            train_sp_logs["train_s_corr"] = stats.spearmanr(
                theta_for_irt, train_real_scores
            )[0]

        train_pred_sklearn_scores = score_predictor_sklearn.predict(train_user_states)
        train_real_scores = (
            train_user_scores * (self.datasets["train"].std + 1e-7)
            + self.datasets["train"].mean
        )
            {
                "train_sklearn_mae": np.average(
                    np.abs(train_pred_sklearn_scores - train_real_scores)
                )
            }
        )

        pred_scores = []
        ################## evaluating score prediciton with testing dataset ##################
        for data in val_sp_data_loader:
            N = data[0].size()[0]
            pred_score = score_predictor(data[0].to(device="cuda")).view(N)
            pred_score = pred_score.detach().cpu().numpy().tolist()
            pred_scores += pred_score

        pred_scores = (
            np.array(pred_scores) * (self.datasets["train"].std + 1e-7)
            + self.datasets["train"].mean
        )
        real_scores = (
            val_user_scores * (self.datasets["train"].std + 1e-7)
            + self.datasets["train"].mean
        )
        logs["new_user/mae"] = np.average(np.abs(pred_scores - real_scores))

        if self.cfg.train.num_d == 1:
            val_theta_for_irt = val_user_states.reshape(np.shape(val_user_states)[0])
        else:
            val_theta_for_irt = pred_scores
        val_s_corr = stats.spearmanr(val_theta_for_irt, real_scores)
        logs["new_user/s_corr"] = val_s_corr[0]
        iso_predicted_val_score = score_predictor_iso(val_theta_for_irt)
        logs["new_user/iso_mae"] = np.average(
            np.abs(iso_predicted_val_score - real_scores)
        )
        sklearn_predicted_val_score = score_predictor_sklearn.predict(val_user_states)
        logs["new_user/sklearn_mae"] = np.average(
            np.abs(sklearn_predicted_val_score - real_scores)
        )
        return logs


def train(
    cfg: DictConfig,
):
    root = "/root/"
    cur_time = datetime.now().replace(microsecond=0).isoformat()
    if not os.path.isdir(os.path.join(root, "logs")):
        os.mkdir(os.path.join(root, "logs"))
    _log_dir = os.path.join(root, f"logs/run_seq_irt_{cfg.type}")
    if not os.path.isdir(_log_dir):
        os.mkdir(_log_dir)
    log_dir = os.path.join(_log_dir, cur_time)
    cfg["log_dir"] = log_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
    # if mixer lambda == 1.0 (only train with vanilla IRT training method(kt)), kt_only should be true.
    # if kt_only == True, only using the prob of correct choice (same with vanilla IRT).
    if cfg.train.mixer_lambda == 1.0:
        assert cfg.train.kt_only
    save_cfg(cfg, log_dir)
    trainer = IRTTrainer(cfg, cfg.log_dir)
    trainer.train(cfg.train.max_epochs)
    trainer.evaluate()
    dir = os.path.join(cfg.log_dir, "checkpoints/")
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, "{}.pt".format("seq_irt"))
    torch.save(trainer.model.state_dict(), path)


class NonDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NonDaemonProcess


if __name__ == "__main__":
    pl.seed_everything(0)
    root = "/root/imsi/research-poc/research_models/"
    model_type = "seq_irt"
    ############## dataset type is one of ["enem", "toeic"] ##############
    dataset_type = "enem"
    cfg_file = f"sp_configs/{model_type}_{dataset_type}.yaml"
    config = OmegaConf.load(os.path.join(root, "configs", cfg_file))
    train(config)
