import torch
from torch.nn import functional as F


class OE(torch.nn.Module):
    """
    :param num_entities: Number of entities to create box embeddings for (eg. number of nodes).
    :param dim: Embedding dimension (i.e. boxes will be in RR^dim).
    """

    def __init__(self, num_entities: int, dim: int, margin: float, t: float = 0.01):
        super().__init__()
        self.dim = dim
        self.tmp = t

        self.margin = margin
        self.emb = torch.nn.Embedding(num_entities, dim)
        torch.nn.init.uniform_(self.emb.weight, a=-1, b=1)
        self.emb.weight.data /= torch.linalg.norm(self.emb.weight.data, axis=1).reshape(
            -1, 1)

    def partial(self, parent, children):
        """check whether children lies in the partial cone defined by parent, i.e., parent > children"""
        parent_emb = self.get_embedding(parent)
        children_emb = self.get_embedding(children)

        inclusion_loss = children_emb - parent_emb
        return (torch.all(inclusion_loss <= 0, dim=-1))

    def get_embedding(self, box):
        return box

    def energy(self, parent, children):
        parent_emb = self.get_embedding(parent)
        children_emb = self.get_embedding(children)

        diff = children_emb - parent_emb

        return torch.norm(F.relu(diff), p=2, dim=-1)

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
