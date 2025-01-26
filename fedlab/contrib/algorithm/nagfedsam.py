import copy

import torch
import numpy as np
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from .fedavg import FedAvgServerHandler
from .minimizers import NagSAM, SAM
from ...utils import SerializationTool


##################
#
#      Server
#
##################


# class FedSAMServerHandler(SyncServerHandler):
class NagFedSamServerHandler(FedAvgServerHandler):
    @property
    def downlink_package(self):
        return [self.model_parameters, self.deltas]

    def setup_optim(self, eta_l, eta_g=1, isLocal=True):
        self.deltas = {cid: None for cid in range(self.num_clients)}
        self.eta_l = eta_l
        self.eta_g = eta_g
        self.isLocal = isLocal
        # self.K = K

    def global_update(self, buffer):
        # global_grad = self.calc_nesterov(buffer)  # grad = theta_prev - theta_current
        super().global_update(buffer)
        # serialized_parameters = self.model_parameters - global_grad * self.eta_g
        # self.set_model(serialized_parameters)
        self.set_nesterov(buffer)
        # self.set_momentum()
        # parameters_list = [ele[0] for ele in buffer]
        # weights = torch.tensor([ele[1] for ele in buffer]).to(self.device)
        # serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        # SerializationTool.deserialize_model(self._model, serialized_parameters)

    def set_nesterov(self, buffer):
        for ele in buffer:
            if self.isLocal:
                self.deltas[ele[2]] = ele[0]
                # self.deltas[ele[2]].div_(self.deltas[ele[2]].norm(2)).mul_(self.g_rho)
            else:
                self.deltas[ele[2]] = self.model_parameters
                # self.deltas[ele[2]].div_(self.deltas[ele[2]].norm(2)).mul_(self.g_rho)

    def calc_nesterov(self, buffer):
        # parameters_list = [ele[0] for ele in buffer]
        # weights = torch.tensor(weights)
        # weights = weights / torch.sum(weights)
        # S = len(buffer)

        eta_l = self.eta_l
        weights = [ele[1] for ele in buffer]
        K = np.array(weights).mean()

        # K = self.K
        # K = buffer[0][2]    # number of epoch
        gradient_list = [
            torch.sub(ele[0], self.model_parameters) for idx, ele in enumerate(buffer)
        ]

        delta = torch.mean(torch.stack(gradient_list, dim=0), dim=0)
        delta.div_(-1*eta_l*K)
        return delta

##################
#
#      Client
#
##################


class NagFedSamSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, rho, g_rho, beta, isNAG,
                 cuda=True, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.rho = rho
        self.beta = beta
        self.isNAG = isNAG
        self.personal = personal
        self.g_rho = g_rho

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        deltas = payload[1]

        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(id, model_parameters, data_loader, deltas[id])
            self.cache.append(pack)

    def train(self, id, model_parameters, train_loader, delta):
        if self.isNAG:
            self.set_model(model_parameters)
            delta_i = model_parameters - delta
            delta_i.div_(delta_i.norm(2)).mul_(self.g_rho)
            minimizer = NagSAM(self.optimizer, self.model, self.rho, self.beta, delta)
        else:
            perturb_parameters = copy.deepcopy(model_parameters)
            if delta is not None:
                perturb_parameters.add_(delta)
            self.set_model(perturb_parameters)
            minimizer = SAM(self.optimizer, self.model, self.rho)

        # data_size = 0
        num_update = 0
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                # Ascent Step
                output = self.model(data)
                loss = self.criterion(output, target)

                loss.backward()
                minimizer.ascent_step()

                # Descent Step
                self.criterion(self.model(data), target).backward()
                minimizer.descent_step()

                # data_size += len(target)
                num_update += 1

        return [self.model_parameters, num_update, id]
