import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss as CELoss
from torch.nn.functional import linear, normalize


class CrossEntropyLoss(CELoss):
    """Rewrite CrossEntropyLoss to support init with kwargs"""

    def __init__(self, feat_dim: int, num_classes: int, lambda_c: float = 1.0, **kwargs):
        super(CrossEntropyLoss, self).__init__(**kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = input[0]
        return super().forward(out, target)


class CenterLoss(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, lambda_c: float = 1.0):
        super(CenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        batch_size = feat.size()[0]
        expanded_centers = self.centers.index_select(dim=0, index=label)
        intra_distances = feat.dist(expanded_centers)
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
        return loss


class ContrastiveCenterLoss(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, lambda_c: float = 1.0):
        super(ContrastiveCenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        batch_size = feat.size()[0]
        expanded_centers = self.centers.expand(batch_size, -1, -1)
        expanded_feat = feat.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers = (expanded_feat - expanded_centers).pow(2).sum(dim=-1)
        distances_same = distance_centers.gather(1, label.unsqueeze(1))
        intra_distances = distances_same.sum()
        inter_distances = distance_centers.sum().sub(intra_distances)
        epsilon = 1e-6
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances / (inter_distances + epsilon) / 0.1

        return loss


class CrossEntropyLoss_ContrastiveCenterLoss(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, lambda_c: float = 1.0):
        super(CrossEntropyLoss_ContrastiveCenterLoss, self).__init__()
        self.cc_loss = ContrastiveCenterLoss(feat_dim, num_classes, lambda_c)
        self.ce_loss = CELoss()

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        logits = feat[0]
        feat_fusion = feat[1]

        ce_loss = self.ce_loss(logits, label)
        cc_loss = self.cc_loss(feat_fusion, label)
        total_loss = ce_loss + cc_loss
        return total_loss


class CrossEntropyLoss_CenterLoss(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, lambda_c: float = 1.0):
        super(CrossEntropyLoss_CenterLoss, self).__init__()
        self.c_loss = CenterLoss(feat_dim, num_classes, lambda_c)
        self.ce_loss = CELoss()

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        logits = feat[0]
        feat_fusion = feat[1]

        ce_loss = self.ce_loss(logits, label)
        c_loss = self.c_loss(feat_fusion, label)
        total_loss = ce_loss + c_loss
        return total_loss


class ContrastiveCenterLossSER(ContrastiveCenterLoss):
    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        feat_fusion = feat[1]
        loss = super().forward(feat_fusion, label)
        return loss


class CenterLossSER(CenterLoss):
    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        feat_fusion = feat[1]
        loss = super().forward(feat_fusion, label)
        return loss


class CombinedMarginLoss(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float,
        m1: float,
        m2: float,
        m3: float,
    ):
        """Combined margin loss for SphereFace, CosFace, ArcFace

        Args:
            in_features (int): the size of feature vector
            out_features (int): the number of classes
            s (float): scale factor
            m1 (float): margin for SphereFace
            m2 (float): margin for ArcFace, m1 must be 1.0
            m3 (float): margin for CosFace, m1 must be 1.0
        """
        super(CombinedMarginLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        self.weight = torch.nn.Parameter(torch.normal(0, 0.01, (out_features, in_features)))

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

        # CrossEntropyLoss
        self.ce_loss = CELoss()

    def forward(self, embbedings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        norm_embeddings = normalize(embbedings)
        norm_weight_activated = normalize(weight)
        logits = linear(norm_embeddings, norm_weight_activated)
        logits = logits.clamp(-1, 1)

        index_positive = torch.where(labels != -1)[0]
        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise ValueError("Unsupported margin values.")

        loss = self.ce_loss(logits, labels)
        return loss, logits


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0.0, alpha: float = None, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = input[0]
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
