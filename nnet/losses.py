import torch


class AveragePoolingClassLoss(torch.nn.Module):
    """
    Computes the loss of the image class probabilities computed from the segmentation map.

    Here we implement a very simple weak loss:
    - For each image we compute the probability of the classes as the softmax of the logits.
    - We then average the probabilities of each class to compute the mean probabilities `probs=[pc0, pc1, ..., pc4]`
    - Finally we compute the Binary Cross Entropy between the class labels and the probabilities`BCE(probs, gt)`.

    """
    def __init__(self, n_classes=5, drop_background=True):
        super().__init__()
        self.loss = torch.nn.BCELoss()
        self.n_classes = n_classes
        self.drop_background = drop_background

    def average_pooling(self, logits):
        batch_agg = []
        if self.drop_background:
            pred = torch.nn.Softmax(dim=1)(logits[:, :-1, :, :])
        else:
            pred = torch.nn.Softmax(dim=1)(logits[:, :, :, :])
        for i_pred in pred:
            max_v, max_i = torch.max(i_pred, axis=0)
            agg = torch.zeros(self.n_classes)
            for c in max_i.unique():
                agg[c] = torch.mean(i_pred[c][max_i == c])
            batch_agg.append(agg)
        return torch.stack(batch_agg).to(logits.device)

    def forward(self, segmentation_logits, class_gt):
        average_class_pred = self.average_pooling(segmentation_logits)
        loss = self.loss(average_class_pred, class_gt)
        return loss
