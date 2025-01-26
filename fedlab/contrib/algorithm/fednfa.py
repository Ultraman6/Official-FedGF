import copy
import math
from multiprocessing import Process

import numpy as np
import torch

from .basic_server import SyncServerHandler
from .basic_client import SGDClientTrainer, SGDSerialClientTrainer
from .fedavg import FedAvgServerHandler
from .minimizers import SAM, LESAM
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
class FedNfaServerHandler(FedAvgServerHandler):
    pass

    # super().__init__()
    """FedAvg server handler."""

    @property
    def downlink_package(self):
        return [self.model_parameters, self.deltas]
    #
    def setup_optim(self, g_rho, mode='v1', isLocal=False):
        self.g_rho = g_rho
        self.mode = mode
        self.isLocal = isLocal
        self.deltas = {cid: None for cid in range(self.num_clients)}

    def hold_deltas(self, ms, ids):
        for id, m in zip(ids, ms):
            self.deltas[id] = m

    def global_update(self, buffer):
        # unpack
        wis = [ele[0] for ele in buffer]
        ids = [ele[2] for ele in buffer]
        ws = torch.tensor([ele[1] for ele in buffer]).to(self.device)
        fis = [self.model_parameters - wi for wi in wis]
        f = Aggregators.fedavg_aggregate(fis, ws)

        if self.mode == 'v':
            SerializationTool.deserialize_model(self._model, self.model_parameters - f)

        elif self.mode == 'v0':
            tf = f.div(f.norm(2)).mul(self.g_rho)
            SerializationTool.deserialize_model(self._model, self.model_parameters - f + tf)

        elif self.mode == 'v1':
            f_fis = [f - fi for fi in fis]
            pf_fis = [g.div_(g.norm(2) + 1e-8).mul_(self.g_rho) for g in f_fis]
            # nws, val = NormSolver.find_norm_element_FW(pf_fis, True)
            pf_fi = Aggregators.fedavg_aggregate(pf_fis, ws)
            SerializationTool.deserialize_model(self._model, self.model_parameters - f)

            self.hold_deltas(pf_fis, ids)

        elif self.mode == 'v2':
            f_fis = [f - fi for fi in fis]
            pf_fis = [g.div_(g.norm(2)).mul_(self.g_rho) for g in f_fis]
            pf_fi = Aggregators.fedavg_aggregate(pf_fis, ws)

            ppf_fis = [pf_fi - pfi for pfi in pf_fis]
            nws, val = NormSolver.find_norm_element_FW(ppf_fis, True)

            ppf_fi = Aggregators.fedavg_aggregate(ppf_fis, nws)
            SerializationTool.deserialize_model(self._model, self.model_parameters - f - ppf_fi)

            self.hold_deltas(ppf_fis, ids)

        elif self.mode == 'v4':
            f_fis = [f - fi for fi in fis]
            pf_fis = [g.div_(g.norm(2)).mul_(self.g_rho) for g in f_fis]
            nws, val = NormSolver.find_norm_element_FW(pf_fis, False)
            pf_fi = Aggregators.fedavg_aggregate(pf_fis, nws)
            pf_fis.append(pf_fi)
            # pf_fi = Aggregators.fedavg_aggregate(pf_fis, ws)
            pnws, val = NormSolver.find_norm_element_FW(pf_fis, True)
            npf_fi = Aggregators.fedavg_aggregate(pf_fis, pnws)
            SerializationTool.deserialize_model(self._model, self.model_parameters - f - npf_fi)

        else:
            raise NotImplementedError

##################
#
#      Client
#
##################


class FedNfaSerialClientTrainer(SGDSerialClientTrainer):
    def __init__(self, model, num_clients, rho,  cuda=True, device=None, logger=None,
                 personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, logger, personal)
        self.rho = rho

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        deltas = payload[1]
        for id in id_list:
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            minimizer = SAM(self.optimizer, self.model, self.rho)
            pack = self.train(id, model_parameters, minimizer, data_loader, deltas[id])
            self.cache.append(pack)

    def train(self, id, model_parameters, minimizer, train_loader, delta=None):
        if delta is None:
            mp = copy.deepcopy(model_parameters)
            dp = self.model_parameters - mp
            mp += dp.div_(dp.norm(2) + 1e-8).mul_(self.rho)
        else:
            mp = model_parameters - delta
        self.set_model(mp)

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
                minimizer.ascent_step()
                self.criterion(self.model(data), target).backward()
                minimizer.descent_step()
                data_size += len(target)

        return [self.model_parameters, data_size, id]
