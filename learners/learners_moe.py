import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert:
    def __init__(
            self, 
            model,
            criterion,
            metric,
            device,
            optimizer,
            lr_scheduler=None,
            is_binary_classification=False
    ):
    
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.metric = metric
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.is_binary_classification = is_binary_classification
        self.model_dim = int(self.get_param_tensor().shape[0])
    
    def get_param_tensor(self):
        param_list = []

        for param in self.model.parameters():
            param_list.append(param.data.view(-1, ))

        return torch.cat(param_list)

class LearnersMoE(object):
    """
    Iterable Ensemble of experts.

    Attributes
    ----------
    moe
    learners_weights
    device
    metric

    Methods
    ----------
    __init__
    __iter__
    __len__
    compute_gradients_and_loss
    optimizer_step
    fit_epochs
    evaluate
    gather_losses
    free_memory
    free_gradients

    """
    def __init__(self, moe, learners_weights):
        self.moe = moe
        self.learners_weights = learners_weights
        self.device = self.moe[-1].device
        self.metric = self.moe[-1].metric
        self.is_binary_classification = self.moe[-1].is_binary_classification
        self.model_dim = self.moe[-1].model_dim
        

    def fit_batch(self, batch, weights=None):
        """
        """
        for expert in self.moe:
            expert.model.train()

        x, y, indices = batch
        x = x.to(self.device).type(torch.float32)
        y = y.to(self.device)

        for expert in self.moe:
            expert.optimizer.zero_grad()

        loss = 0.0
        h = self.moe[-1].model(x)  
        for expert_id, expert in enumerate(self.moe[:-1]):
            expert_pred = expert.model(h)
            loss_expert = expert.criterion(expert_pred, y)
            if weights is not None:
                weights = weights.to(self.device)
                loss += (loss_expert.T @ weights[expert_id][indices]) / loss_expert.size(0)
            else:
                loss += loss_expert.mean()
        loss.backward()

        for expert in self.moe:
            expert.optimizer.step()
            if expert.lr_scheduler is not None:
                expert.lr_scheduler.step()


    def fit_epoch(self, iterator, weights=None):
        """
        """
        for expert in self.moe:
            expert.model.train()


        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            for expert in self.moe:
                expert.optimizer.zero_grad()

            loss = 0.0
            h = self.moe[-1].model(x)  
            for expert_id, expert in enumerate(self.moe[:-1]):
                expert_pred = expert.model(h)
                loss_expert = expert.criterion(expert_pred, y)
                if weights is not None:
                    weights = weights.to(self.device)
                    loss += (loss_expert.T @ weights[expert_id][indices]) / loss_expert.size(0)
                else:
                    loss += loss_expert.mean()
            loss.backward()

            for expert in self.moe:
                expert.optimizer.step()



    def fit_epochs(self, iterator, n_epochs, weights=None):
        """
        """
        for step in range(n_epochs):
            self.fit_epoch(iterator, weights)

            for expert in self.moe:
                if expert.lr_scheduler is not None:
                    expert.lr_scheduler.step()
                
    def gather_losses(self, iterator):
        """
        gathers losses for all elements of iterator

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor with losses of all elements of the iterator.dataset

        """
        for expert in self.moe:
            expert.model.eval()
        n_samples = len(iterator.dataset)
        n_experts = len(self.moe[:-1])
        all_losses = torch.zeros(n_experts, n_samples, device=self.device)

        with torch.no_grad():
            for (x, y, indices) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                h = self.moe[-1].model(x)  
                for expert_id, expert in enumerate(self.moe[:-1]):
                    expert_pred = expert.model(h)
                    all_losses[expert_id][indices] = expert.criterion(expert_pred, y).squeeze()

        return all_losses


    def evaluate_iterator(self, iterator):
        """
        """
        criterion = nn.NLLLoss(reduction="none")

        for expert in self.moe:
            expert.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for (x, y, _) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)
                n_samples += y.size(0)

                y_pred = 0.
                h = self.moe[-1].model(x)  
                for expert_id, expert in enumerate(self.moe[:-1]):
                    y_pred += self.learners_weights[expert_id] * F.softmax(expert.model(h), dim=1)

                y_pred = torch.clamp(y_pred, min=0., max=1.)

                global_loss += criterion(torch.log(y_pred), y).sum().item()

                global_metric += self.metric(y_pred, y).item()

            return global_loss / n_samples, global_metric / n_samples

    def __iter__(self):
        return LearnersMoEIterator(self)

    def __len__(self):
        return len(self.moe)

    def __getitem__(self, idx):
        return self.moe[idx]
    
class LearnersMoEIterator(object):
    """
    LearnersEnsemble iterator class

    Attributes
    ----------
    _learners_ensemble
    _index

    Methods
    ----------
    __init__
    __next__

    """
    def __init__(self, learners_moe):
        self._learners_moe = learners_moe.moe
        self._index = 0

    def __next__(self):
        while self._index < len(self._learners_moe):
            result = self._learners_moe[self._index]
            self._index += 1

            return result

        raise StopIteration
    
class LearnerspFedMoE(LearnersMoE):
    def fit_epoch(self, iterator, weights=None):
        """
        """
        for expert in self.moe:
            expert.model.train()


        for x, y, indices in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            for expert in self.moe:
                expert.optimizer.zero_grad()

            loss = 0.0
            weights = self.moe[-1].model(x) #[b, n]
            for expert_id, expert in enumerate(self.moe[:-1]):
                expert_pred = expert.model(x)
                loss_expert = expert.criterion(expert_pred, y)
                loss += (loss_expert * weights[:, expert_id]).mean()

            loss.backward()

            for expert in self.moe:
                expert.optimizer.step()

    def evaluate_iterator(self, iterator):
        """
        """
        criterion = nn.NLLLoss(reduction="none")

        for expert in self.moe:
            expert.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for (x, y, _) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)
                n_samples += y.size(0)

                y_pred = 0.
                weights = self.moe[-1].model(x) #[b, n]
                for expert_id, expert in enumerate(self.moe[:-1]):
                    y_pred += weights[:,expert_id].unsqueeze(1) * F.softmax(expert.model(x), dim=1)

                y_pred = torch.clamp(y_pred, min=0., max=1.)

                global_loss += criterion(torch.log(y_pred), y).sum().item()

                global_metric += self.metric(y_pred, y).item()

            return global_loss / n_samples, global_metric / n_samples