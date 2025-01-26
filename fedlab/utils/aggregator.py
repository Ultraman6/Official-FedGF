# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


class Aggregators(object):
    """Define the algorithm of parameters aggregation"""

    @staticmethod
    def fedavg_aggregate(serialized_params_list, weights=None):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Merge all tensors following FedAvg.
            weights (list, numpy.array or torch.Tensor, optional): Weights for each params, the length of weights need to be same as length of ``serialized_params_list``

        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list)).cuda()

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device='cuda:0')

        weights = weights / torch.sum(weights)
        assert torch.all(weights >= 0), "weights should be non-negative values"
        serialized_parameters = torch.sum(
            torch.stack(serialized_params_list, dim=-1) * weights, dim=-1)

        return serialized_parameters

    @staticmethod
    def fedavg_aggregate_tensor(serialized_params_tensor, weights=None, device='cuda:0'):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_tensor (torch.Tensor): A 2D tensor where each column is a serialized model parameter (i.e., a gradient vector).
            weights (torch.Tensor or None, optional): Weights for each parameter set,
                the length of weights must match the number of columns in `serialized_params_tensor`.
            device (str, optional): Device to use for the computation. Default is 'cuda:0'.

        Returns:
            torch.Tensor: Aggregated tensor after weighted average.
        """
        # Ensure the input tensor is on the correct device
        serialized_params_tensor = serialized_params_tensor.to(device)

        # If no weights are provided, use equal weights
        if weights is None:
            weights = torch.ones(serialized_params_tensor.size(1), device=device)

        # If weights are not a tensor, convert them to a tensor
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, device=device)

        # Normalize the weights so they sum to 1
        weights = weights / torch.sum(weights)

        # Ensure weights are non-negative
        assert torch.all(weights >= 0), "Weights should be non-negative values."

        # Perform the weighted sum across the columns (dimension 1)
        aggregated_params = torch.matmul(serialized_params_tensor, weights)

        return aggregated_params

    @staticmethod
    def fedasync_aggregate(server_param, new_param, alpha):
        """FedAsync aggregator
        
        Paper: https://arxiv.org/abs/1903.03934
        """
        serialized_parameters = torch.mul(1 - alpha, server_param) + \
                                torch.mul(alpha, new_param)
        return serialized_parameters
