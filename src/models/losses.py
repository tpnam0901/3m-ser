import torch
from torch import nn
from torch.nn import CrossEntropyLoss as CELoss


class CrossEntropyLoss(CELoss):
    def forward(self, input, target):
        out = input[0]
        return super().forward(out, target)


class CenterLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, lambda_c=1.0):
        super(CenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.size()[0]
        expanded_centers = self.centers.index_select(dim=0, index=label)
        intra_distances = feat.dist(expanded_centers)
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
        return loss


class ContrastiveCenterLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, lambda_c=1.0):
        super(ContrastiveCenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, feat, label):
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
    def __init__(self, feat_dim, num_classes, lambda_c=1.0):
        super(CrossEntropyLoss_ContrastiveCenterLoss, self).__init__()
        self.cc_loss = ContrastiveCenterLoss(feat_dim, num_classes, lambda_c)
        self.ce_loss = CELoss()

    def forward(self, feat, label):
        logits = feat[0]
        feat_fusion = feat[1]
        feat_text = feat[2]
        feat_audio = feat[3]

        ce_loss = self.ce_loss(logits, label)
        cc_loss = self.cc_loss(feat_fusion, label) + self.cc_loss(feat_text, label) + self.cc_loss(feat_audio, label)
        total_loss = ce_loss + cc_loss
        return total_loss


class CrossEntropyLoss_ContrastiveCenterLossForServer(nn.Module):
    def __init__(self, fusion_dim, text_dim, audio_dim, num_classes, lambda_c=1.0):
        super(CrossEntropyLoss_ContrastiveCenterLossForServer, self).__init__()
        self.ccf_loss = ContrastiveCenterLoss(fusion_dim, num_classes, lambda_c)
        self.cct_loss = ContrastiveCenterLoss(text_dim, num_classes, lambda_c)
        self.cca_loss = ContrastiveCenterLoss(audio_dim, num_classes, lambda_c)
        self.ce_loss = CELoss()

    def forward(self, feat, label):
        logits = feat[0]
        feat_fusion = feat[1]
        feat_text = feat[2]
        feat_audio = feat[3]
        ce_loss = self.ce_loss(logits, label)
        cc_loss = self.ccf_loss(feat_fusion, label) + self.cct_loss(feat_text, label) + self.cca_loss(feat_audio, label)
        total_loss = ce_loss + cc_loss
        return total_loss


class CrossEntropyLoss_CenterLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, lambda_c=1.0):
        super(CrossEntropyLoss_CenterLoss, self).__init__()
        self.c_loss = CenterLoss(feat_dim, num_classes, lambda_c)
        self.ce_loss = CELoss()

    def forward(self, feat, label):
        logits = feat[0]
        feat_fusion = feat[1]
        feat_text = feat[2]
        feat_audio = feat[3]

        ce_loss = self.ce_loss(logits, label)
        c_loss = self.c_loss(feat_fusion, label) + self.c_loss(feat_text, label) + self.c_loss(feat_audio, label)
        total_loss = ce_loss + c_loss
        return total_loss
