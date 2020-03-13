
import sys
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm as tqdm
from utils.meter import AverageValueMeter
from utils.metric import Accuracy



class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs



class DistillEpoch(Epoch):

    def __init__(self, model, teacher, optimizer, teacher_weight=0.9, temperature=2, eps=1e-18, device='cpu', verbose=True):
        loss = nn.CrossEntropyLoss()
        loss.__name__ = 'cross_entropy'                
        super().__init__(
            model=model,
            loss=loss,
            metrics=[Accuracy()],
            stage_name='distill',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.teacher = teacher.to(device)
        self.temperature = temperature
        self.teacher_outputs = {}
        self.teacher_weight = teacher_weight
        self.eps = eps
        self.student_loss = 0.0
        self.distill_loss = 0.0
        self.teacher_loss = 0.0
        self.average_N = 5

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()

        # Compute student loss
        output = self.model.forward(x)
        student_loss = self.loss(output, y)

        # Compute teacher output
        x_unknown = torch.cat([torch.unsqueeze(x_item, dim=0) for x_item in x if x_item.cpu().detach().numpy().tostring() not in self.teacher_outputs])
        y_unknown = self.teacher.forward(x_unknown).detach()
        unknown_idx = 0
        teacher_output = []
        for x_item in x:
            x_val = x_item.cpu().detach().numpy().tostring()
            if x_val in self.teacher_outputs:
                teacher_output.append(self.teacher_outputs[x_val])
            else:
                teacher_output.append(torch.unsqueeze(y_unknown[unknown_idx], dim=0))
                unknown_idx += 1
        teacher_output = torch.cat(teacher_output, dim=0)

        self.teacher_outputs.update(
            {
                x_item.cpu().detach().numpy().tostring(): torch.unsqueeze(y_item, dim=0) 
                for x_item, y_item in zip(x_unknown, y_unknown)
            }
        )

        teacher_output = teacher_output.to(self.device)

        # Compute soft student output and target
        soft_output = output / self.temperature
        soft_target = teacher_output / self.temperature

        # Compute distillation loss
        distill_loss = nn.KLDivLoss(reduction='sum')(
            F.log_softmax(soft_output, dim=1),
            F.softmax(soft_target, dim=1)
        ) / soft_output.shape[0] * self.temperature * self.temperature

        avg_k = 2 / (self.average_N + 1)
        self.student_loss = student_loss.cpu().detach() * avg_k + self.student_loss * (1 - avg_k)
        self.distill_loss = distill_loss.cpu().detach() * avg_k + self.distill_loss * (1 - avg_k) 

        if isinstance(self.teacher_weight, str) and self.teacher_weight == 'auto':
            teacher_loss = self.loss(teacher_output, y)
            self.teacher_loss = teacher_loss.cpu().detach() * avg_k + self.teacher_loss * (1 - avg_k) 
            teacher_weight = 1.0 * (student_loss - teacher_loss) / (self.student_loss - self.teacher_loss)
            teacher_weight = torch.sigmoid(teacher_weight * 3)

            distill_loss *= self.student_loss / self.distill_loss
        else:
            teacher_weight = self.teacher_weight
        loss = (1.0 - teacher_weight) * student_loss + teacher_weight * distill_loss

        loss.backward()
        self.optimizer.step()
        
        return student_loss, output



class TrainEpoch(Epoch):

    def __init__(self, model, optimizer, device='cpu', verbose=True):
        loss = nn.CrossEntropyLoss()
        loss.__name__ = 'cross_entropy'        
        super().__init__(
            model=model,
            loss=loss,
            metrics=[Accuracy()],
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction



class ValidEpoch(Epoch):

    def __init__(self, model, device='cpu', verbose=True):
        loss = nn.CrossEntropyLoss()
        loss.__name__ = 'cross_entropy'
        super().__init__(
            model=model,
            loss=loss,
            metrics=[Accuracy()],
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction


