import torch
import torch.nn.functional as F


class BPRLoss(torch.nn.Module):
    """BPRLoss（Bayesian Personalized Ranking Loss）

    """

    def __init__(self):
        super().__init__()

    def forward(self, pos_score, neg_score):
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        return loss


class TLLoss(torch.nn.Module):
    """ Triplet Logistic Loss

    """

    def __init__(self):
        super().__init__()

    def forward(self, pos_score, neg_score):
        loss = F.logsigmoid(neg_score - pos_score + 1).mean()
        return loss


class THLoss(torch.nn.Module):
    """Triplet Hinge Loss

    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.m = margin

    def forward(self, pos_score, neg_score):
        # relu(x) = max{0,x}
        loss = F.relu(neg_score - pos_score + self.m).mean()
        return loss
