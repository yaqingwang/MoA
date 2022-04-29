from torch import nn
from transformers.activations import get_activation

import torch
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
import torch.nn.functional as F
import copy
import typing

def percentile(t, q):
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    return t.view(-1).kthvalue(k).values.item()


class GetSubnet(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class MixtureSoup(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1, inference_level=0, \
                 weight_strategy="soft", sparsity=0.75):
        super(MixtureSoup, self).__init__()

        self.deepspeed_experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])
        self.num_local_experts = num_local_experts
        self.inference_level = inference_level
        self.weight_strategy = weight_strategy
        self.sparsity = sparsity
        self.expert_score_weight = torch.nn.Parameter(torch.zeros(self.num_local_experts), requires_grad=False)

    def get_expert_by_idx(self, idx):
        return self.deepspeed_experts[idx]

    def expert_soup_forward(self, input):
        output = F.linear(input,
                          self.parameter_dict["weight"],
                          self.parameter_dict["bias"])
        return output

    def expert_soup(self):

        weight = F.softmax(self.expert_score_weight)
        print(weight)
        if self.weight_strategy == "hard":
            subnet = GetSubnet.apply(weight, self.sparsity)
            weight = subnet / subnet.sum()

        self.parameter_dict = {"weight": 0, "bias": 0}
        for idx in range(self.num_local_experts):
            single_expert = self.deepspeed_experts[idx]
            for s_name, s_param in single_expert.named_parameters():
                if "weight" in s_name:
                    p_name = "weight"
                    self.parameter_dict[p_name] = self.parameter_dict[p_name] + (weight[idx] * s_param)
                else:
                    p_name = "bias"
                    self.parameter_dict[p_name] = self.parameter_dict[p_name] + (weight[idx] * s_param)

    def forward(self, *input: Tensor):
        expert_output = None
        if self.deepspeed_experts[0].training:
            expert_idx = torch.randint(low=0, high=self.num_local_experts, size=(1,)).item()  # selected expert
            if self.expert_score_weight.requires_grad:
                self.expert_soup()
                expert_output = self.expert_soup_forward(input[0])
            else:
                expert_output = self.get_expert_by_idx(expert_idx)(input[0])

        else:
            result = []
            for expert_idx in range(self.num_local_experts):
                temp = self.get_expert_by_idx(expert_idx)(input[0])
                result.append(temp)
            result = torch.stack(result, dim=0)

            if self.inference_level == 0:  # token level
                mask = torch.randint(0, self.num_local_experts,
                                     size=(result.size(1), result.size(2)), device=result.device)
                for i in range(self.num_local_experts):
                    expert_mask = mask.eq(i)
                    result[i] *= expert_mask.unsqueeze(-1)
                expert_output = result.sum(0)
            elif self.inference_level == 1:  # sentence level
                mask = torch.randint(0, self.num_local_experts,
                                     size=(result.size(1),), device=result.device)
                for i in range(self.num_local_experts):
                    expert_mask = mask.eq(i)
                    result[i] *= expert_mask.unsqueeze(-1).unsqueeze(-1)
                expert_output = result.sum(0)
            elif self.inference_level == 3:  # ensemble
                expert_output = result.mean(0)
            elif self.inference_level == 4:
                self.expert_soup()
                expert_output = self.expert_soup_forward(input[0])

        return expert_output


class ExpertSoup(nn.Module):
    def __init__(self, dim, r, act=None, num_expert=4, inference_level=0, sharing_down=0, sharing_up=0,
                 weight_strategy="soft", sparsity=0.75):
        super().__init__()

        self.act = act
        if sharing_down == 1:
            self.MoA_A = MixtureSoup(nn.Linear(dim, r), 1, inference_level)
        else:
            self.MoA_A = MixtureSoup(nn.Linear(dim, r), num_expert, inference_level, weight_strategy, sparsity)
        if act is not None:
            self.act = get_activation(act)

        if sharing_up == 1:
            self.MoA_B = MixtureSoup(nn.Linear(r, dim), 1, inference_level)
        else:
            self.MoA_B = MixtureSoup(nn.Linear(r, dim), num_expert, inference_level, weight_strategy, sparsity)

    def forward(self, x, residual):
        result = self.MoA_A(x)
        if self.act is not None:
            result = self.act(result)
        result = self.MoA_B(result)
        return result + residual