import math
import torch
from torch.nn.functional import relu, softplus


class PluginTBox:
    def __init__(self, tmp=0.1):
        self.tmp = tmp

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

    def our_dist(self, children_center, children_offset, parent_center, parent_offset):
        min_parent = parent_center - parent_offset
        max_parent = parent_center + parent_offset
        min_children = children_center - children_offset
        max_children = children_center + children_offset

        V_child = self.volume(min_children, max_children)
        V_intersection = self.volume(*self.Intersection(min_parent, max_parent, min_children, max_children))
        return V_child - V_intersection

