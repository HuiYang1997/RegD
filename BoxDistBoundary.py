import torch
import math
from torch.nn import LeakyReLU
from torch.nn import functional as F


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class depthBox(torch.nn.Module):
    """
    :param num_entities: Number of entities to create box embeddings for (eg. number of nodes).
    :param dim: Embedding dimension (i.e. boxes will be in RR^dim).
    """

    def __init__(self, num_entities: int, dim: int, margin: float, depth_type: str, device: str, mode: str,
                 add_denominator: str, hybrid: int, lamda: float = 1.0):
        super().__init__()
        self.curvature = 1.0
        self.sub_apex_dist = 0.0001
        self.device = device
        self.hybrid = hybrid
        self.tmp = 0
        self.lamda = lamda

        self.mode = mode
        self.add_denominator = add_denominator

        self.dim = dim
        self.margin = margin
        self.depth_type = depth_type

        # initialize embeddings
        self.emb = torch.nn.Embedding(num_entities, dim * 2).to(device)
        torch.nn.init.uniform_(self.emb.weight, a=-1, b=1)
        self.emb.weight.data /= torch.linalg.norm(self.emb.weight.data, axis=1).reshape(
            -1, 1)

        # initialize denominator
        self.denominator = torch.nn.Sequential(
            torch.nn.Linear(dim, 2*self.dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*self.dim, 1),
            torch.nn.Softplus(),
        ).to(self.device)

    def depthdistance_box(self, parent_center, parent_offset, children_center, children_offset):
        factor = ((parent_center - children_center).square().sum(dim=-1)
                  + (parent_offset - children_offset).square().sum(dim=-1))

        denominator = self.denominator(parent_center) * self.denominator(children_center)

        dist = factor / ((parent_offset.norm(dim=-1)) * (children_offset.norm(dim=-1)))
        if self.add_denominator == 'in':
            dist = dist / (denominator.squeeze(-1))
        elif self.add_denominator == 'out':
            dist = denominator.squeeze(-1) * dist
        elif self.add_denominator == 'non':
            pass
        else:
            raise ValueError('add_denominator must be either in, out or non')

        return dist

    def depthdistance(self, parent_center, parent_offset_z, children_center, children_offset_z):
        factor = (parent_center - children_center).square().sum(dim=-1) + (parent_offset_z - children_offset_z).square()

        denominator = self.denominator(parent_center) * self.denominator(children_center)

        dist = factor / (parent_offset_z * children_offset_z)
        if self.add_denominator == 'in':
            dist = dist / (denominator.squeeze(-1))
        elif self.add_denominator == 'out':
            dist = (denominator.squeeze(-1)) * dist
        elif self.add_denominator == 'non':
            pass
        else:
            raise ValueError('add_denominator must be either in, out or non')

        assert (parent_offset_z > 0).all()
        assert (children_offset_z > 0).all()
        assert (factor >= 0).all()

        # dist = torch.arccosh(1 + dist)
        return dist

    def dist_to_boundary(self, a, children_offset_z, mode='inclusion'):
        if mode == 'ShadowCone':
            dist = a / children_offset_z
            return torch.arcsinh(dist) + math.asinh(1)
        elif mode == 'inclusion':
            diff = a + children_offset_z
            if self.mode == 'ball':
                assert self.hybrid == 0
                return diff
            elif self.mode == 'box':
                # t = torch.max(diff, dim=-1)[0]
                t = torch.norm(F.relu(diff), dim=-1)
                t1 = torch.max(diff, dim=-1)[0]
                assert self.hybrid == 0
                return torch.where(t > 0, t, t1)
            else:
                raise ValueError('mode must be either box or ball')
        else:
            raise ValueError('mode must be either inclusion or ShadowCone')

    def LSE(self, x, y):
        if self.tmp == 0:
            return torch.maximum(x, y)
        else:
            return self.tmp * torch.log(torch.exp(x / self.tmp) + torch.exp(y / self.tmp))

    def volume(self, min, max):
        return torch.log(self.LSE(max - min, torch.zeros_like(min)) + 1e-6).sum(dim=-1)

    def Intersection(self, min1, max1, min2, max2):
        min_int = self.LSE(min1, min2)
        max_int = -self.LSE(-max1, -max2)
        return min_int, max_int

    def get_embedding(self, box):
        center = box[..., :self.dim]
        offset = F.softplus(box[..., self.dim:])
        return center, offset

    def energy(self, parent, children):
        """distance energy to move children inside the parent cone"""
        parent_center, parent_offset = self.get_embedding(parent)
        children_center, children_offset = self.get_embedding(children)

        if self.training:
            parent_offset = parent_offset * math.exp(-self.sub_apex_dist)

        if self.mode == 'box':
            dist_apex = self.depthdistance_box(parent_center, parent_offset, children_center, children_offset)

            dist = torch.abs(parent_center - children_center)
            a = (dist - parent_offset)
            if self.hybrid == 1:
                # compute the distance based on volume
                min_parent = parent_center - parent_offset
                max_parent = parent_center + parent_offset
                min_children = children_center - children_offset
                max_children = children_center + children_offset

                V_children = self.volume(min_children, max_children)
                V_intersection = self.volume(*self.Intersection(min_parent, max_parent, min_children, max_children))

                dist_to_boundary = V_children - V_intersection
            else:
                dist_to_boundary = self.dist_to_boundary(a, children_offset, mode='inclusion')

        elif self.mode == 'ball':
            children_offset_norm = children_offset[..., 0]
            parent_offset_norm = parent_offset[..., 0]

            dist = torch.norm(parent_center - children_center, dim=-1)
            a = (dist - parent_offset_norm)

            dist_apex = self.depthdistance(parent_center, parent_offset_norm, children_center, children_offset_norm)

            if self.hybrid == 1:
                dist_to_boundary = self.dist_to_boundary(a, children_offset_norm, mode='ShadowCone')
                # a = a + children_offset_norm
                # # use 1-dimension approximation
                # vol_intersection = children_offset_norm + parent_offset_norm - dist
                # vol_intersection = torch.where(a < 0, 2 * children_offset_norm, vol_intersection)
                # vol_intersection = torch.where(children_offset_norm > parent_offset_norm + dist, 2 * parent_offset_norm,
                #                                vol_intersection)
                #
                # vol_intersection = torch.log(F.relu(vol_intersection) + 1e-6)
                # vol_children = torch.log(2 * children_offset_norm + 1e-6)
                #
                # dist_to_boundary = vol_children - vol_intersection

                # use 2-dimenison approxiamtion
                # sum_r = children_offset_norm + parent_offset_norm
                # diff_r = torch.abs(children_offset_norm - parent_offset_norm)
                # a1 = (sum_r-dist).square()
                # a2 = dist.square() + 2*dist*sum_r-3*diff_r.square()
                #
                # intersection = torch.where(dist < sum_r, pi*(a1+a2)/(12*dist), torch.zeros_like(dist))
                #
                # vol_intersection = torch.log(intersection + 1e-6)
                # vol_children = torch.log(4 * pi * (children_offset_norm**3) / 3 + 1e-6)
                # vol_parent = torch.log(4 * pi * (parent_offset_norm**3) / 3 + 1e-6)
                # dist_to_boundary = torch.where(dist > diff_r, vol_children - vol_intersection, vol_children - vol_parent)
            else:
                dist_to_boundary = self.dist_to_boundary(a, children_offset_norm, mode='inclusion')

        else:
            raise ValueError('mode must be either box or ball')

        if self.training:
            return dist_to_boundary + self.lamda * torch.where(dist_to_boundary < 0, torch.zeros_like(dist_to_boundary),
                                                           dist_apex)
        else:
            return dist_to_boundary + self.lamda * dist_apex

    def partial(self, parent, children, ratio=1):
        """check whether children lies in the partial cone defined by parent, i.e., parent > children"""
        parent_center, parent_offset = self.get_embedding(parent)
        children_center, children_offset = self.get_embedding(children)

        euc = torch.abs(parent_center - children_center)
        if self.mode == 'box':
            inclusion_loss = euc - ratio * (parent_offset - children_offset)
            return torch.max(inclusion_loss, dim=-1)[0]
        elif self.mode == 'ball':
            inclusion = euc.norm(dim=-1, p=2)
            inclusion -= ratio * parent_offset.norm(dim=-1, p=2)
            inclusion += ratio * children_offset.norm(dim=-1, p=2)
            return inclusion
        else:
            raise ValueError('mode must be either box or ball')

    def forward(self, inputs, reverse=False):
        """
        input:
            contrastive_batch: batch_size * [relation[0]=0, relation[1]=1, 2, 3]
        output:
            energy between relation[0] and rest: batch_size * (num_pos + num_neg)
        now for training use only
        """
        e = self.emb(inputs)
        p2 = e.narrow(1, 1, e.size(1) - 1)
        p1 = e.narrow(1, 0, 1).expand_as(p2)
        if reverse:
            parent, children = p2, p1
        else:
            parent, children = p1, p2
        return self.energy(parent, children).squeeze(-1)

    def loss(self, energy):
        pos_energy = torch.clamp(energy[..., 0], min=0.0)
        neg_energy = torch.clamp(self.margin - energy[..., 1:], min=0.0).sum(dim=-1)
        return (pos_energy + neg_energy).sum()

    def loss_cross(self, energy):
        """return averaged cross-entropy loss"""
        pos_energy = torch.clamp(energy[..., 0:1], min=0.0)
        neg_energy = torch.clamp(energy[..., 1:] - self.margin, max=0.0)
        exp_neg_energ = torch.exp(-torch.cat([pos_energy, neg_energy], dim=-1))
        prob_pos = exp_neg_energ[..., 0] / torch.sum(exp_neg_energ, axis=-1)
        return - torch.log(prob_pos).mean()
