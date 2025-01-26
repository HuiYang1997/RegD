import math
import torch
from torch.nn.functional import relu, softplus


class PluginRegD:
    def __init__(self, dim, classNum, margin=0, use_softplus=False, mode='box'):
        self.dim = dim
        self.classNum = classNum
        self.margin = margin
        self.sub_apex_dist = 0.0001
        self.use_softplus = use_softplus
        self.mode = mode

    def depthFactor(self, offset1, offset2):
        if self.mode == 'box':
            return torch.norm(offset1, dim=-1) * torch.norm(offset2, dim=-1)
        elif self.mode == 'ball':
            return offset1 * offset2
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
    
    def our_dist(self, euc, r1, r2):
        if self.mode == 'box':
            # boundary dist
            diff = euc + r1 - r2
            dist_1 = torch.max(diff, dim=-1)[0]
            dist_2 = torch.norm(relu(diff), dim=-1)
            dist_boundary = torch.where(dist_1 < 0, dist_1, dist_2)

            # depth dist
            dstDepth = 1.0*(euc.abs().sum(dim=-1) + (r1-r2).abs().sum(dim=-1)) / self.depthFactor(r1, r2)
        elif self.mode == 'ball':
            # boundary dist
            dist_boundary = euc + r1 - r2

            # depth dist
            dstDepth = 1.0*(euc.square() + (r1-r2).square()) / self.depthFactor(r1, r2)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        our_dist = dist_boundary + torch.where(dist_boundary > 0 , dstDepth, torch.zeros_like(dstDepth))
        # our_dist = relu(our_dist + self.margin)

        return our_dist.unsqueeze(-1)
    
    def get_embedding(self, box):
        center = box[..., :self.dim]
        if self.mode == 'ball':
            offset = box[..., -1]
        else:
            offset = box[..., self.dim:]

        if self.use_softplus:
            offset = softplus(offset)
        else:
            offset = torch.abs(offset)
        return center, offset

    def loss_cross(self, energy):
        """return averaged cross-entropy loss"""
        pos_energy = torch.clamp(energy[..., 0:1], min=0.0)
        neg_energy = torch.clamp(energy[..., 1:] - self.margin, max=0.0)
        exp_neg_energ = torch.exp(-torch.cat([pos_energy, neg_energy], dim=-1))
        prob_pos = exp_neg_energ[..., 0] / torch.sum(exp_neg_energ, axis=-1)
        return - torch.log(prob_pos).mean()

    def energy(self, parent, children, relation=None):
        """distance energy to move children inside the parent cone"""
        parent_center, parent_offset = self.get_embedding(parent)
        children_center, children_offset = self.get_embedding(children)

        if relation is not None:
           children_center = children_center + relation.unsqueeze(1)

        parent_offset = parent_offset * math.exp(-self.sub_apex_dist)

        if self.mode == 'box':
            euc = torch.abs(parent_center - children_center)

            our_dist = self.our_dist(euc, children_offset, parent_offset)
        else:
            euc = torch.norm(parent_center - children_center, dim=-1)
            our_dist = self.our_dist(euc, children_offset, parent_offset)

        return our_dist
    
    def extending_with_neg_samples(self, input, num_neg=10):
        # produce negativa smaples randomly by replacing the last element with a random class
        # input: tensor of shape (batch_size, 2 or 3)
        # size: number of negative samples to create
        # return: tensor of shape (batch_size, size)
        n = input.shape[1]
        
        # duplicate the input tensor size times along the  dimension 1
        new_input = input.repeat(1, num_neg//n + 1)[:, :num_neg+n]
        # replace the last element with a random class
        new_input[:, n:] = torch.randint(0, self.classNum, (new_input.shape[0], num_neg))

        return new_input
    

    def nf1Loss_ours(self, input, embedding):
        new_input = self.extending_with_neg_samples(input)

        e = embedding(new_input)
        p2 = e.narrow(1, 1, e.size(1) - 1)
        p1 = e.narrow(1, 0, 1).expand_as(p2)
        parent, children = p2, p1
        return self.loss_cross(self.energy(parent, children).squeeze(-1))

    def nf3Loss_ours(self, input, embedding, relation_embedding):
        # c\sqsubseteq \exists r. d
        r_emb = relation_embedding(input[:, 1])

        # select the 0,2 rows in the dim 1 of the input tensor
        input_cd = input[:, [0, 2]]
        new_input = self.extending_with_neg_samples(input_cd)

        e = embedding(new_input)

        p2 = e.narrow(1, 1, e.size(1) - 1)
        p1 = e.narrow(1, 0, 1).expand_as(p2) # add r to c
        parent, children = p1, p2
        return self.loss_cross(self.energy(parent, children, r_emb).squeeze(-1))