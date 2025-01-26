import copy

import torch
from fedlab.utils.model import create_zero_list
from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from .fedavg import FedAvgServerHandler
from .minimizers import SAM, LESAM
from ...utils import Aggregators

##################
#
#      Server
#
##################


# class FedSAMServerHandler(SyncServerHandler):
class FedLESAMServerHandler(FedAvgServerHandler):
    pass

    # super().__init__()
    """FedAvg server handler."""
    @property
    def downlink_package(self):
        return [self.model_parameters, self.deltas]

    def setup_optim(self, isLocal=True):
        self.isLocal = isLocal
        self.deltas = {cid: None for cid in range(self.num_clients)}

        # self.global_c += 1.0 * len(dcs) / self.num_clients * dc
    def global_update(self, buffer, upload_res=False):
        super().global_update(buffer)
        for ele in buffer:
            cid = ele[2]
            if self.isLocal:
                self.deltas[cid] = ele[1]
            else:
                self.deltas[cid] = self.model_parameters

##################
#
#      Client
#
##################

class FedLESAMSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, rho, isNAG=False, cuda=True, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.rho = rho
        self.isNAG = isNAG

    # def setup_optim(self, epochs, batch_size, lr):
    #     super().setup_optim(epochs, batch_size, lr)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        # self.SAM = SAM(self.optimizer, self.model, self.rho)

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        deltas = payload[1]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(id, model_parameters, data_loader, deltas[id])
            self.cache.append(pack)

    def train(self, id, model_parameters, train_loader, delta):
        self.set_model(model_parameters)
        minimizer = LESAM(self.optimizer, self.model, self.rho)
        if delta is None:
            perturb = -model_parameters
        else:
            perturb = delta - model_parameters
        perturb.div_(perturb.norm(2))

        data_size = 0
        for _ in range(self.epochs):
            for i, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data = data.to(self.device)
                    target = target.to(self.device).reshape(-1).long()

                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                if self.isNAG and i != 0:
                    nag_perturb = self.model_parameters - model_parameters
                    nag_perturb.div_(nag_perturb.norm(2))
                    nag_perturb.add_(perturb)  # 可换成梯度解耦
                    minimizer.ascent_step(nag_perturb)
                else:
                    minimizer.ascent_step(perturb)
                self.criterion(self.model(data), target).backward()
                minimizer.descent_step()
                data_size += len(target)

        self.old_parameters = copy.deepcopy(model_parameters)
        return [self.model_parameters, data_size, id]
