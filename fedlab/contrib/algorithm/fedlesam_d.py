import copy

import torch
from fedlab.utils.model import create_zero_list
from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from .fedavg import FedAvgServerHandler
from .minimizers import SAM, LESAM, LESAM_D
from ...utils import Aggregators


##################
#
#      Server
#
##################


# class FedSAMServerHandler(SyncServerHandler):
class FedLESAMDServerHandler(FedAvgServerHandler):
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

    # def global_update(self, buffer):
    #     # unpack
    #     dys = [ele[0] for ele in buffer]
    #     dcs = [ele[1] for ele in buffer]
    #
    #     dx = Aggregators.fedavg_aggregate(dys)
    #     dc = Aggregators.fedavg_aggregate(dcs)
    #
    #     next_model = self.model_parameters + self.lr * dx
    #     self.set_model(next_model)

    # self.global_c += 1.0 * len(dcs) / self.num_clients * dc


##################
#
#      Client
#
##################


class FedLESAMDSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, rho, cuda=True, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.rho = rho
        old_model = copy.deepcopy(model)
        for param in old_model.parameters():
            param.data.zero_()  # 使用 zero_() 方法将所有参数值设为0
        self.old_parameters = old_model.parameters()

    # def setup_optim(self, epochs, batch_size, lr):
    #     super().setup_optim(epochs, batch_size, lr)
    # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
    # self.SAM = SAM(self.optimizer, self.model, self.rho)

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(id, model_parameters, data_loader)
            self.cache.append(pack)

    def train(self, id, model_parameters, train_loader):
        self.set_model(model_parameters)
        optimizer = LESAM_D(self.model.parameters(), self.optimizer, rho=self.rho)
        global_update = [pb - pa for pb, pa in zip(self.old_parameters, model_parameters)]
        self.old_parameters = copy.deepcopy(model_parameters)

        data_size = 0
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.to(self.device)
                    target = target.to(self.device).reshape(-1).long()

                # for i, param in enumerate(self.model.parameters()):
                #     diff_vect = torch.norm(self.old_parameters[i] - before_parameters[i])
                #     param.data += self.rho * diff_vect / torch.norm(diff_vect, p=2)

                # Ascent Step
                # output = self.model(data)
                # loss = self.criterion(output, target)
                optimizer.paras = [data, target, self.criterion, self.model]
                optimizer.step(global_update)

                # Descent Step
                # self.optimizer.zero_grad()
                # loss.backward()
                self.optimizer.step()

                data_size += len(target)

        return [self.model_parameters, data_size]
