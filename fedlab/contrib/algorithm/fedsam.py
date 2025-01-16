import copy

import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from .fedavg import FedAvgServerHandler
from .minimizers import SAM
from ...utils import Aggregators
from ...utils.model import create_zero_list


##################
#
#      Server
#
##################


# class FedSAMServerHandler(SyncServerHandler):
class FedSamServerHandler(FedAvgServerHandler):
    pass

    # super().__init__()
    """FedAvg server handler."""
    # @property
    # def downlink_package(self):
    #     return [self.model_parameters] #, self.global_c]
    #
    # def setup_optim(self, lr):
    #     self.lr = lr
        # self.global_c = torch.zeros_like(self.model_parameters)

    def global_update(self, buffer):
        # unpack
        super().global_update(buffer)
        g0s = [ele[2] for ele in buffer]
        g1s = [ele[3] for ele in buffer]

        g0g = Aggregators.fedavg_aggregate(g0s)
        g1g = Aggregators.fedavg_aggregate(g1s)

        div_g0 = sum([torch.norm(g0-g0g) for g0 in g0s])
        div_g1 = sum([torch.norm(g1-g1g) for g1 in g1s])


##################
#
#      Client
#
##################


class FedSamSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, rho, cuda=True, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.rho = rho

    # def setup_optim(self, epochs, batch_size, lr):
    #     super().setup_optim(epochs, batch_size, lr)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        # self.SAM = SAM(self.optimizer, self.model, self.rho)

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            # optimizer = torch.optim.SGD(model_parameters, lr=self.lr)
            minimizer = SAM(self.optimizer, self.model, self.rho)
            pack = self.train(id, model_parameters, minimizer, data_loader)
            self.cache.append(pack)

    def train(self, id, model_parameters, minimizer, train_loader):
        self.set_model(model_parameters)
        g0 = create_zero_list(self.model)
        g1 = copy.deepcopy(g0)

        data_size = 0
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                g = create_zero_list(self.model)
                # Ascent Step
                output = self.model(data)
                loss = self.criterion(output, target)

                loss.backward()
                for i, param in enumerate(self.model.parameters()):
                    g[i] = param.grad.data.clone()
                    g0[i] += g[i]
                minimizer.ascent_step()

                # Descent Step
                self.criterion(self.model(data), target).backward()
                for i, param in enumerate(self.model.parameters()):
                    g1[i] += (param.grad.data.clone() - g[i])

                minimizer.descent_step()

                data_size += len(target)

        return [self.model_parameters, data_size, g0, g1]
