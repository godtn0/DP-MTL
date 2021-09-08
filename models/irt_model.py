import os
import torch
import torch.nn as nn

from sklearn.isotonic import IsotonicRegression

import numpy as np


def param_array(size):
    return torch.nn.Parameter(
        torch.randn(size, dtype=torch.float32), requires_grad=True
    )


class MultiClassIRT(nn.Module):
    def __init__(self, num_q, num_s, num_opt=1, num_d=4):
        super(MultiClassIRT, self).__init__()
        self.num_d = num_d
        self.num_opt = num_opt
        self.a = param_array((num_q, num_opt, num_d))
        self.b = param_array((num_q, num_opt))
        self.theta = param_array((num_s, num_d))

    def forward(self, x):
        N = x.size()[0]
        uids = x[:, 0]
        thetas = self.theta[uids].view(N, self.num_d, 1)

        qids = x[:, 1]
        a = self.a[qids]
        b = self.b[qids]

        logits = (torch.bmm(a, thetas)).view(N, self.num_opt) + b
        return logits


class IRTScoreGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.iso_reg = IsotonicRegression(increasing="auto", out_of_bounds="clip")

    def optimize(self, data):
        user_params, scores = data
        self.score_model = self.iso_reg.fit(user_params, scores)

    def forward(self, x):
        return self.score_model.predict(x)


class EarlyStopping:
    def __init__(
        self, metric, patience=3, verbose=False, delta=0, dir="checkpoint.pt"
    ) -> None:
        self.metric = metric
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.dir = dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_path = None

    def __call__(self, value, model):
        if self.metric in ["AUC", "ACC"]:
            score = value
        elif self.metric in ["loss"]:
            score = -value

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(value, model)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, value, model):
        pre_value = (
            self.best_score if self.metric in ["AUC", "ACC"] else -self.best_score
        )
        dir = os.path.join(self.dir, "checkpoints_{}/".format(self.metric))
        if not os.path.exists(dir):
            os.makedirs(dir)
        path = os.path.join(
            self.dir, "checkpoints_{}/{}.pt".format(self.metric, str(value))
        )
        torch.save(model.state_dict(), path)
        self.best_model_path = path


class LinearRegression(nn.Module):
    def __init__(self, num_scores=1, num_d=4):
        super().__init__()
        self.num_scores = num_scores
        self.num_d = num_d
        self._build_model()

    def _build_model(self):
        self.linear = nn.Linear(self.num_d, self.num_scores)

    def forward(self, x):
        return self.linear(x)


class NeuralCollborativeFiltering(nn.Module):
    def __init__(
        self, num_q, num_s, num_d, num_opt=1, hidden_dim=4, out_dim=1, num_layers=3
    ):
        super().__init__()
        self.num_q = num_q
        self.num_s = num_s
        self.num_d = num_d
        self.num_opt = num_opt
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        (
            self.ncf_layers,
            self.gmf_layer,
            self.theta,
            self.a,
            self.to_logit,
        ) = self._build_model()
        self.sigomid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def _build_model(self):
        ncf_layers = nn.ModuleList()
        in_dim = self.num_d * 2
        out_dim = self.hidden_dim
        for i in range(self.num_layers):
            ncf_layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

        theta = param_array((self.num_s, self.num_d))
        a = param_array((self.num_q, self.num_opt, self.num_d))

        gmf_layer = nn.Linear(self.num_d, out_dim)

        to_logit = nn.Linear(self.num_d + out_dim, self.out_dim)

        return ncf_layers, gmf_layer, theta, a, to_logit

    def forward(self, x):
        N = x.size()[0]
        uids = x[:, 0]
        user_embedding = self.theta[uids]

        qids = x[:, 1]
        item_embedding = self.a[qids]
        item_embedding = torch.transpose(item_embedding, 0, 1)

        for i in range(self.num_opt):
            ncf_x = torch.cat((user_embedding, item_embedding[i]), 1)
            for block in self.ncf_layers:
                ncf_x = self.relu(block(ncf_x))
            gmf_x = user_embedding * item_embedding[i]
            x = torch.cat((ncf_x, gmf_x), -1)
            x = self.to_logit(x)
            if i == 0:
                logit = x
            else:
                logit = torch.cat((logit, x), -1)
        return logit


class SequentialIRT(nn.Module):
    def __init__(
        self, num_q, num_d, num_layers, num_opt, max_seq_len=None, bidirectional=True
    ):
        super().__init__()
        self.num_q = num_q
        self.max_seq_len = max_seq_len
        if self.max_seq_len is None:
            self.max_seq_len = self.num_q
        self.num_d = num_d
        self.num_opt = num_opt
        self.bidirectional = bidirectional

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.lstm = nn.LSTM(
            num_d, num_d, num_layers, batch_first=True, bidirectional=bidirectional
        )
        self.a = param_array((num_q, num_opt, num_d))

    def forward(self, data, eval=False):
        if eval:
            x = data[0]
        else:
            x = data
        N = x["item_id"].size()[0]
        out_dim = 2 if self.bidirectional else 1
        qids = x["item_id"]
        q_embed = self.a[qids]
        q_embed_t = torch.transpose(q_embed, -2, -1)
        one_hot_student_choice = torch.nn.functional.one_hot(
            x["student_choice"] * x["pad_mask"]
        ).view(N, self.max_seq_len, self.num_opt, 1)
        one_hot_student_choice = one_hot_student_choice.type(torch.float)
        q_selected_emb = torch.matmul(q_embed_t, one_hot_student_choice).view(
            N, self.max_seq_len, -1
        )
        if eval:
            user_state = data[1]
        else:
            user_states, _ = self.lstm(q_selected_emb)
            user_states = torch.mean(
                user_states.view(N, self.max_seq_len, out_dim, self.num_d), 2
            )
            user_states = user_states * x["pad_mask"].view(N, self.max_seq_len, 1)
            user_state = torch.sum(user_states, -2) / x["sequence_size"].view(N, 1)
        q_embed_for_logit = torch.transpose(q_embed, 0, 2)
        logits = q_embed_for_logit * user_state
        logits = torch.sum(torch.transpose(logits, 0, 2), -1)

        logits = logits * x["pad_mask"].view(N, self.max_seq_len, 1)

        one_hot_correct_choice = torch.nn.functional.one_hot(
            x["correct_choice"] * x["pad_mask"]
        ).view(N, self.max_seq_len, self.num_opt, 1)
        one_hot_correct_choice = one_hot_correct_choice.type(torch.float)
        logit_from_dirt = torch.matmul(
            logits.view(N, self.max_seq_len, 1, self.num_opt), one_hot_correct_choice
        ).view(N, self.max_seq_len)
        prob_from_dirt = self.sigmoid(logit_from_dirt)
        prob_from_dirt = prob_from_dirt * x["pad_mask"]

        probs = self.softmax(logits)
        prob_from_pirt = torch.matmul(
            probs.view(N, self.max_seq_len, 1, self.num_opt), one_hot_correct_choice
        ).view(N, self.max_seq_len)
        prob_from_pirt = prob_from_pirt * x["pad_mask"]

        return logits, prob_from_dirt, prob_from_pirt, user_state
