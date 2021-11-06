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
from torch.utils.data import TensorDataset, dataset, DataLoader
from torch.functional import Tensor

import numpy as np

import pandas as pd

from tqdm import tqdm

from sklearn import linear_model
import sklearn.metrics as metrics
from scipy import stats


from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf

from datasets.enem_dataset import enem_dataset, ScoreDataset
from datasets.toeic_dataset import toeic_score_dataset
from models.irt_model import (
    IRTScoreGenerator,
    MultiClassIRT,
    LinearRegression,
    EarlyStopping,
    NeuralCollborativeFiltering,
)
from research_models.utils import save_cfg


def make_irt_data(
    dataset,
    user_id_mapper=None,
    question_id_mapper=None,
    score_data=False,
):
    uids = []
    qids = []
    anss = []
    student_choices = []
    correct_choices = []

    for data in dataset:
        if score_data:
            inters = data["interactions"]
        else:
            inters = data
        seq_len = len(inters)
        if user_id_mapper is not None:
            if not data["user_id"] in user_id_mapper.keys():
                user_id_mapper[data["user_id"]] = len(user_id_mapper)

            user_id = user_id_mapper[data["user_id"]]

        item_ids = []
        if question_id_mapper is not None:
            for inter in inters:
                if not inter["item_id"] in question_id_mapper.keys():
                    question_id_mapper[inter["item_id"]] = len(question_id_mapper)
                item_ids.append(question_id_mapper[inter["item_id"]])

        uids += [user_id] * seq_len
        qids += item_ids
        anss += list(inters["is_correct"])
        student_choices += list(inters["student_choice"])
        correct_choices += list(inters["correct_choice"])
    return (
        np.array(uids),
        np.array(qids),
        np.array(anss),
        np.array(student_choices),
        np.array(correct_choices),
    )


def build_data_loader(
    uids,
    qids,
    anss,
    student_choices=None,
    correct_choices=None,
    batch_size=None,
    train=True,
):
    train_tensor = torch.zeros((len(anss), 5), dtype=int)
    train_tensor[:, 0] = torch.from_numpy(uids)
    train_tensor[:, 1] = torch.from_numpy(qids)
    train_tensor[:, 2] = torch.from_numpy(anss)
    if student_choices is not None and correct_choices is not None:
        train_tensor[:, 3] = torch.from_numpy(student_choices)
        train_tensor[:, 4] = torch.from_numpy(correct_choices)
    train_dataset = TensorDataset(train_tensor)
    if batch_size is None:
        batch_size = len(train_dataset) // 64 + 1
    from torch.utils.data import DataLoader

    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train)
    return train_dataset, train_data_loader


class IRTTrainer:
    def __init__(self, cfg, log_path) -> None:
        self.cfg = cfg
        self.log_path = log_path
        if cfg.data.dataset_type == "toeic":
            sp = cfg.type == "sp"
            self.datasets = toeic_score_dataset(cfg, test=sp)
        elif cfg.data.dataset_type == "enem":
            self.datasets = enem_dataset(
                cfg=cfg,
            )
        self.user_id_mappers = {}
        self.item_id_mapper = {}
        (
            self.irt_datasets,
            self.irt_dataloaders,
            self.num_students,
            self.num_questions,
        ) = self._build_irt_data_loaders(self.datasets)

        # print("=" * 30 + "build data_loader" + "=" * 30)

        self.model = self._build_model(
            self.num_questions["train"],
            self.num_students["train"],
            self.cfg.train.model,
        )
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

        # print("=" * 30 + "build irt model" + "=" * 30)

    def _build_irt_data_loaders(self, datasets):
        num_students = {}
        num_questions = {}
        irt_datasets = {}
        irt_dataloaders = {}
        self.user_id_mappers["train"] = {}
        for key in datasets.keys():
            self.user_id_mappers[key] = {}
            if key != "train" and self.cfg.type != "sp":
                self.user_id_mappers[key] = self.user_id_mappers["train"]
            (uids, qids, anss, student_choices, correct_choices,) = make_irt_data(
                datasets[key],
                self.user_id_mappers[key],
                self.item_id_mapper,
                score_data=self.cfg.train.score_data,
            )
            num_students[key] = len(np.unique(uids))
            num_questions[key] = np.max(qids) + 1
            irt_datasets[key], irt_dataloaders[key] = build_data_loader(
                uids,
                qids,
                anss,
                student_choices,
                correct_choices,
                self.cfg.batch_size,
            )
        return irt_datasets, irt_dataloaders, num_students, num_questions

    def _build_model(self, num_questions, num_students, model_name):
        torch.manual_seed(self.cfg.train.random_seed)
        if model_name == "multic":
            model = MultiClassIRT(
                num_questions,
                num_students,
                num_opt=self.cfg.train.num_opt,
                num_d=self.cfg.train.num_d,
            )
        elif model_name == "ncf":
            model = NeuralCollborativeFiltering(
                num_q=num_questions,
                num_s=num_students,
                num_d=self.cfg.train.num_d,
                num_opt=self.cfg.train.num_opt,
                hidden_dim=self.cfg.train.hidden_dim,
                out_dim=self.cfg.train.out_dim,
                num_layers=self.cfg.train.num_layers,
            )

        return model

    def _build_optimizer(self):
        if self.cfg.train.weight_decay:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                self.cfg.train.lr,
                weight_decay=self.cfg.train.weight_decaying_lambda,
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(), self.cfg.train.lr)

    def _build_loss(self):
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def train(self, epochs=None):
        if epochs is None:
            epochs = self.cfg.train.max_epochs
        self._build_optimizer()
        self._build_loss()
        self.model.train()
        self.model.to(device="cuda")
        # print("=" * 30 + "training start" + "=" * 30)
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
            for data in tqdm(self.irt_dataloaders["train"]):
                train_logs = self._train_step(data)
                train_losses.append(train_logs["loss"])
                train_kt_losses.append(train_logs["kt_loss"])
                train_ct_losses.append(train_logs["ct_loss"])
                train_probs += train_logs["prob"].detach().numpy().tolist()
                train_logits.append(train_logs["logits"].detach().numpy())
                train_sans += data[0][:, 2].cpu().numpy().tolist()
                train_user_choices += data[0][:, 3].cpu().numpy().tolist()
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
                val_mix_logs = self._evaluate_ct(self.irt_dataloaders["val"])
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
            print(logs)

    def _train_step(self, data, model=None, optimizer=None):
        if model is None:
            model = self.model
        if optimizer is None:
            optimizer = self.optimizer
        data = data[0].to(device="cuda")
        optimizer.zero_grad()
        logits = model(data)
        ct_loss = self.ce_loss(logits, data[:, 3])
        ####### directly cacluate propbability of correctness through sigomid ######
        logit = logits[range(logits.shape[0]), data[:, 4]]
        prob_form_dirt = self.sigmoid(logit)
        kt_loss_from_dirt = self.bce_loss(prob_form_dirt, data[:, 2].float())
        ############################################################################
        probs = self.softmax(logits)
        prob_from_pirt = probs[range(probs.shape[0]), data[:, 4]]
        kt_loss_from_pirt = self.bce_loss(prob_from_pirt, data[:, 2].float())
        if self.cfg.train.kt_only:
            kt_loss = kt_loss_from_dirt
            prob = prob_form_dirt
        else:
            kt_loss = kt_loss_from_pirt
            prob = prob_from_pirt
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
                    test_logs = self._evaluate_ct(self.irt_dataloaders["test"])
                    key_list = list(test_logs.keys())
                    for key in key_list:
                        test_logs["eval/test_" + key] = test_logs.pop(key)
                sp_logs = self._evaluate_sp(test=True)

            if self.early_stopping_auc.best_model_path is not None:
                self.model.load_state_dict(
                    torch.load(self.early_stopping_auc.best_model_path)
                )
                if self.cfg.data.split.mix[1] > 0:
                    test_logs = self._evaluate_ct(self.irt_dataloaders["test"])
                    key_list = list(test_logs.keys())
                    for key in key_list:
                        test_logs["eval/test_" + key + "_fa"] = test_logs.pop(key)
                sp_logs = self._evaluate_sp(test=True)
                key_list = list(sp_logs.keys())
                for key in key_list:
                    sp_logs[key + "_fa"] = sp_logs.pop(key)

        return sp_logs, test_logs

    def _evaluate_ct(self, dataloader, model=None):
        if model is None:
            model = self.model
        model.eval()
        model.to(device="cuda")
        logs = {}
        ct_losses = []
        kt_losses = []
        logits = []
        probs = []
        anss = []
        s_choices = []

        for data in dataloader:
            data = data[0].to(device="cuda")
            logit = model(data)
            ct_loss = self.ce_loss(logit, data[:, 3])
            prob_from_pirt = self.softmax(logit)
            prob_from_pirt = prob_from_pirt[range(prob_from_pirt.shape[0]), data[:, 4]]
            kt_loss_from_pirt = self.bce_loss(prob_from_pirt, data[:, 2].float())
            logit_to_dirt = logit[range(logit.shape[0]), data[:, 4]]
            prob_form_dirt = self.sigmoid(logit_to_dirt)
            kt_loss_from_dirt = self.bce_loss(prob_form_dirt, data[:, 2].float())
            if self.cfg.train.kt_only:
                kt_loss = kt_loss_from_dirt
                prob = prob_form_dirt
            else:
                kt_loss = kt_loss_from_pirt
                prob = prob_from_pirt
            ct_losses.append(ct_loss.cpu().item())
            kt_losses.append(kt_loss.cpu().item())
            logits.append(logit.cpu().detach().numpy())
            probs += prob.cpu().detach().numpy().tolist()
            s_choices += data[:, 3].cpu()
            anss += data[:, 2].cpu()
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
        train_theta = model.theta.data.detach().cpu().numpy()
        ################## making training score dataset ##################
        train_scores = {
            "score": [],
            "user_id": [],
        }
        for data in self.datasets["train"]:
            for key in train_scores.keys():
                train_scores[key].append(data[key])

        train_user_sorting_indices = []
        for i in range(len(train_scores["user_id"])):
            train_user_sorting_indices.append(
                self.user_id_mappers["train"][train_scores["user_id"][i]]
            )
        train_scores["score"] = np.array(train_scores["score"])

        train_sorted_theta = train_theta[train_user_sorting_indices]
        if self.cfg.type == "mix":
            np.random.seed(0)
            num_users = len(train_scores["score"])
            indices = np.arange(num_users)
            np.random.shuffle(indices)

            val_scores = {
                "score": [],
                "user_id": [],
            }
            val_indices = indices[: int(num_users * 0.1)]
            val_sorted_theta = train_sorted_theta[val_indices]
            val_scores["score"] = train_scores["score"][val_indices]

            train_indices = indices[int(num_users * 0.1) :]
            train_sorted_theta = train_sorted_theta[train_indices]
            train_scores["score"] = train_scores["score"][train_indices]

            val_sp_dataset = ScoreDataset(val_sorted_theta, val_scores["score"])

            batch_size = 64

            val_sp_data_loader = DataLoader(
                val_sp_dataset, batch_size=batch_size, shuffle=False
            )

        train_sp_dataset = ScoreDataset(train_sorted_theta, train_scores["score"])
        batch_size = 256

        train_sp_data_loader = DataLoader(
            train_sp_dataset, batch_size=batch_size, shuffle=True
        )

        ################## defining score prediction model ##################
        score_predictor = LinearRegression(num_d=self.cfg.train.num_d, num_scores=1)
        score_predictor_iso = IRTScoreGenerator()
        score_predictor_sklearn = linear_model.LinearRegression().fit(
            train_sorted_theta,
            train_scores["score"] * (self.datasets["train"].std + 1e-7)
            + self.datasets["train"].mean,
        )

        mse = nn.MSELoss()
        optimizer = optim.Adam(score_predictor.parameters(), lr=0.01)
        score_predictor.train()
        score_predictor.to(device="cuda")

        ################## training score prediction model ##################
        for epoch in range(50):
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

        train_pred_sklearn_scores = score_predictor_sklearn.predict(train_sorted_theta)
        train_real_scores = (
            train_scores["score"] * (self.datasets["train"].std + 1e-7)
            + self.datasets["train"].mean
        )

        ################## evaluating score prediciton with testing dataset ##################
        pred_scores = []
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
            val_scores["score"] * (self.datasets["train"].std + 1e-7)
            + self.datasets["train"].mean
        )
        logs["new_user/mae"] = np.average(np.abs(pred_scores - real_scores))

        if self.cfg.train.num_d == 1:
            val_theta_for_irt = val_sorted_theta.reshape(np.shape(val_sorted_theta)[0])
        else:
            val_theta_for_irt = pred_scores
        val_s_corr = stats.spearmanr(val_theta_for_irt, real_scores)
        logs["new_user/s_corr"] = val_s_corr[0]
        iso_predicted_val_score = score_predictor_iso(val_theta_for_irt)
        logs["new_user/iso_mae"] = np.average(
            np.abs(iso_predicted_val_score - real_scores)
        )
        sklearn_predicted_val_score = score_predictor_sklearn.predict(val_sorted_theta)
        logs["new_user/sklearn_mae"] = np.average(
            np.abs(sklearn_predicted_val_score - real_scores)
        )
        return logs

    def save_user_params(self, save_dir=""):
        with open(save_dir + "/user_id_mapper.json", "wb") as f:
            pickle.dump(self.user_id_mappers["train"], f)

    def save_item_params(self, save_dir=""):
        with open(save_dir + "/item_id_mapper.json", "wb") as f:
            pickle.dump(self.item_id_mapper, f)


def train(
    cfg: DictConfig,
):
    root = "/root/"
    cur_time = datetime.now().replace(microsecond=0).isoformat()
    if not os.path.isdir(os.path.join(root, "logs")):
        os.mkdir(os.path.join(root, "logs"))
    _log_dir = os.path.join(root, f"logs/run_irt_{cfg.type}")
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
    trainer.save_user_params(cfg.log_dir)
    trainer.save_item_params(cfg.log_dir)
    dir = os.path.join(cfg.log_dir, "checkpoints/")
    if not os.path.exists(dir):
        os.makedirs(dir)
    path = os.path.join(dir, "{}.pt".format("irt"))
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
    root = "/root/imsi/DP-MTL"
    script_name = "snap_trainer"
    ############## dataset type is one of ["enem", "toeic"] ##############
    dataset_type = "enem"
    cfg_file = f"./irt_{dataset_type}.yaml"
    config = OmegaConf.load(os.path.join(root, "configs", cfg_file))
    config = OmegaConf.merge(config, OmegaConf.from_cli())

    train(config)