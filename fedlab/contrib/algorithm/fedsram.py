import copy

import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from .fedavg import FedAvgServerHandler
from .minimizers import SAM
from ...utils import Aggregators, SerializationTool
from ...utils.accumulators import NormSolver
from ...utils.logger import log_metric
from ...utils.model import create_zero_list, create_model_param_list


##################
#
#      Server
#
##################


# class FedSAMServerHandler(SyncServerHandler):
class FedSRamServerHandler(FedAvgServerHandler):
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
        # 将列表转换为张量，并移动到 GPU
        gfs = torch.stack([ele[2] for ele in buffer]).to(self.device)
        gps = torch.stack([ele[3] for ele in buffer]).to(self.device)
        # 计算差值
        ghs = gps - gfs
        ghfs = ghs - gfs
        # 聚合
        gfa = Aggregators.fedavg_aggregate_tensor(gfs, device=self.device)
        gpa = Aggregators.fedavg_aggregate_tensor(gps, device=self.device)
        gha = Aggregators.fedavg_aggregate_tensor(ghs, device=self.device)
        ghfa = Aggregators.fedavg_aggregate_tensor(ghfs, device=self.device)
        # 计算范数（batch-wise）
        gf_div = torch.norm(gfs - gfa.unsqueeze(1), p=2, dim=1).sum().item()
        gp_div = torch.norm(gps - gpa.unsqueeze(1), p=2, dim=1).sum().item()
        gh_div = torch.norm(ghs - gha.unsqueeze(1), p=2, dim=1).sum().item()
        ghf_div = torch.norm(ghfs - ghfa.unsqueeze(1), p=2, dim=1).sum().item()
        # 记录度量
        log_metric(
            ["Divergence of Update Gradient", "Divergence of Perturb Gradient",
             "Divergence of Surrogate Gradient", "Divergence of Residual Gradient"],
            [gf_div, gp_div, gh_div, ghf_div],
            self.round
        )

        parameters_list = [ele[0] for ele in buffer]
        weights, val = NormSolver.find_norm_element_FW(parameters_list, False)
        # weights = torch.tensor([ele[1] for ele in buffer]).to(self.device)
        serialized_parameters = Aggregators.fedavg_aggregate(parameters_list, weights)
        SerializationTool.deserialize_model(self._model, serialized_parameters)


##################
#
#      Client
#
##################


class FedSRamSerialClientTrainer(SGDSerialClientTrainer):
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
        # minimizer = SAM(self.optimizer, self.model, self.rho)
        # train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.set_model(model_parameters)
        gf = torch.zeros_like(model_parameters).flatten()
        gp = copy.deepcopy(gf)

        data_size = 0
        for _ in range(self.epochs):
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                # Ascent Step
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                gf += minimizer.ascent_step() * len(target)
                self.criterion(self.model(data), target).backward()
                gp += minimizer.descent_step() * len(target)
                data_size += len(target)
        gf /= data_size
        gp /= data_size
        # print(torch.norm(gf, p=2), torch.norm(gp, p=2))
        return [self.model_parameters, data_size, gf, gp]
