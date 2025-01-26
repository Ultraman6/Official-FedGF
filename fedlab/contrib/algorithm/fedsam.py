import copy
import math
from multiprocessing import Process

import numpy as np
import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from .fedavg import FedAvgServerHandler
from .minimizers import SAM, LESAM
from ...utils import Aggregators
from ...utils.logger import log_metric
from ...utils.model import create_zero_list, create_model_param_list


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

    # def global_update(self, buffer):
    #     # unpack
    #     super().global_update(buffer)
    #     # 将列表转换为张量，并移动到 GPU
    #     gfs, ghs = [ele[2] for ele in buffer], [ele[3] for ele in buffer]
    #     ghfs_client, ghfs_sample = [ele[4] for ele in buffer], [ele[5] for ele in buffer]
    #     # gfs = torch.stack([ele[2] for ele in buffer]).to(self.device)
    #     # ghs = torch.stack([ele[3] for ele in buffer]).to(self.device)
    #
    #     # ghf_ns = [ele[2] for ele in buffer]
    #     # gfs = torch.stack([ele[2] for ele in buffer]).to(self.device)
    #     # gps = torch.stack([ele[3] for ele in buffer]).to(self.device)
    #     # 计算差值
    #     # ghs = gps - gfs
    #     # ghfs = ghs - gfs
    #     # 聚合
    #     gfa = Aggregators.fedavg_aggregate(gfs)
    #     gfa_norm = torch.norm(gfa, p=2).item()
    #     # gfa = Aggregators.fedavg_aggregate_tensor(gfs, device=self.device)
    #     # gpa = Aggregators.fedavg_aggregate_tensor(gps, device=self.device)
    #     gha = Aggregators.fedavg_aggregate(ghs)
    #     gha_norm = torch.norm(gha, p=2).item()
    #     # gha = Aggregators.fedavg_aggregate_tensor(ghs, device=self.device)
    #     # ghfa = Aggregators.fedavg_aggregate_tensor(ghfs, device=self.device)
    #
    #     gfas = [(ele[2] - gfa) / gfa_norm for ele in buffer]
    #     ghas = [(ele[3] - gha) / gha_norm for ele in buffer]
    #     # 计算范数（batch-wise）
    #     gf_norms = [torch.norm(gf - gfa, p=2).item() for gf in gfs]
    #     gh_norms = [torch.norm(gh - gha, p=2).item() for gh in ghs]
    #     gf_div = np.mean(gf_norms)
    #     gh_div = np.mean(gh_norms)
    #     # gf_div = torch.norm(gfs - gfa.unsqueeze(1), p=2, dim=1).sum().item()
    #     # gp_div = torch.norm(gps - gpa.unsqueeze(1), p=2, dim=1).sum().item()
    #     # gh_div = torch.norm(ghs - gha.unsqueeze(1), p=2, dim=1).sum().item()
    #     # ghf_div = torch.norm(ghfs - ghfa.unsqueeze(1), p=2, dim=1).sum().item()
    #     gfh_cov = np.mean([torch.dot(gf_, gh_).item() for gf_, gh_ in zip(gfas, ghas)])
    #     gfh_cov_abs = np.mean([abs(torch.dot(gf_, gh_).item()) for gf_, gh_ in zip(gfas, ghas)])
    #     gfh_cov_norm = np.mean([gf_n * gh_n for gf_n, gh_n in zip(gf_norms, gh_norms)])
    #     # 记录度量
    #     # log_metric(
    #     #     ["Divergence of Update Gradient", "Divergence of Perturb Gradient",
    #     #      "Divergence of Surrogate Gradient", "Divergence of Residual Gradient"],
    #     #     [gf_div, gp_div, gh_div, ghf_div],
    #     #     self.round
    #     # )
    #     # log_metric(
    #     #     ["Divergence of Residual"],
    #     #     [ghf_div],
    #     #     self.round,
    #     #     True
    #     # )
    #     log_metric(
    #         ["Div of Update", "Div of Perturb", "Div of Update standard", "Div of Perturb standard",
    #          "Cov of Update and Perturb", "Abs ov of Update and Perturb", "Norm ov of Update and Perturb",
    #          "Mean of Residual Client", "Mean of Residual Sample", 'Std of Residual Client', 'Std of Residual Sample'],
    #         [gf_div, gh_div, gf_div / gfa_norm , gh_div / gha_norm,
    #          gfh_cov, gfh_cov_abs, gfh_cov_norm,
    #          np.mean(ghfs_client), np.mean(ghfs_sample), np.std(ghfs_client), np.std(ghfs_sample)],
    #         self.round,
    #         True
    #     )
    #     # # 计算范数（单独的聚合梯度）
    #     # gf_norm = torch.norm(gfa, p=2).item()
    #     # gp_norm = torch.norm(gpa, p=2).item()
    #     # gh_norm = torch.norm(gha, p=2).item()
    #     # ghf_norm = torch.norm(ghfa, p=2).item()
    #     # # 记录范数
    #     # log_metric(
    #     #     ["Norm of Update Gradient", "Norm of Perturb Gradient",
    #     #      "Norm of Surrogate Gradient", "Norm of Residual Gradient"],
    #     #     [gf_norm, gp_norm, gh_norm, ghf_norm],
    #     #     self.round
    #     # )


##################
#
#      Client
#
##################


class FedSamSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, rho, isNAG=False, cuda=True, device=None, logger=None,
                 personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.rho = rho
        self.isNAG = isNAG

    # def setup_optim(self, epochs, batch_size, lr):
    #     super().setup_optim(epochs, batch_size, lr)
    # optimizer = tSAMorch.optim.SGD(self.model.parameters(), lr=self.lr)
    # self.SAM = (self.optimizer, self.model, self.rho)

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            # optimizer = torch.optim.SGD(model_parameters, lr=self
            if self.isNAG:
                minimizer = LESAM(self.optimizer, self.model, self.rho)
            else:
                minimizer = SAM(self.optimizer, self.model, self.rho)
            pack = self.train(id, model_parameters, minimizer, data_loader)
            self.cache.append(pack)

    def train(self, id, model_parameters, minimizer, train_loader):
        # minimizer = SAM(self.optimizer, self.model, self.rho)
        # train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.set_model(model_parameters)
        gf = torch.zeros_like(model_parameters).flatten()
        # gp = copy.deepcopy(gf)
        gh = copy.deepcopy(gf)
        ghf = copy.deepcopy(gf)
        # ghfs = []
        ghf_sample = 0.0
        data_size = 0
        for _ in range(self.epochs):
            for i, (data, target) in enumerate(train_loader):
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)
                # Ascent Step
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                if self.isNAG:
                    if i == 0:
                        minimizer.optimizer.step()
                    else:
                        nag_perturb = model_parameters - self.model_parameters
                        nag_perturb.div_(nag_perturb.norm(2))
                        minimizer.ascent_step(nag_perturb)
                        self.criterion(self.model(data), target).backward()
                        minimizer.descent_step()
                else:   
                    _gf = minimizer.ascent_step()
                    self.criterion(self.model(data), target).backward()
                    _gp = minimizer.descent_step()
                    gf += _gf * len(target)
                    gh += (_gp - _gf) * len(target)
                    ghf += (_gp - 2 * _gf) * len(target)
                    # ghfs.append(_gp - 2 * _gf)
                    ghf_sample += (torch.norm(_gp - 2 * _gf, p=2)).item() * len(target)
                data_size += len(target)

        gf /= data_size
        # gp /= data_size
        gh /= data_size
        ghf /= data_size
        ghf_client = torch.norm(ghf, p=2).item()
        ghf_sample /= data_size

        # print(ghf_n)
        # print(torch.norm(gf, p=2), torch.norm(gp, p=2))
        return [self.model_parameters, data_size, gf, gh, ghf_client, ghf_sample]
