import re

import torch.nn as nn



class BaseObject(nn.Module):

    def __init__(self, name=None):
        super().__init__()
        self._name = name

    @property
    def __name__(self):
        if self._name is None:
            name = self.__class__.__name__
            s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
            return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        else:
            return self._name



class Metric(BaseObject):
    pass



class Accuracy(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, y_pr, y_gt):
        _, predicted = y_pr.max(1)
        total = y_gt.size(0)
        correct = predicted.eq(y_gt).sum()
        accuracy = 1.0 * correct / total

        return accuracy
