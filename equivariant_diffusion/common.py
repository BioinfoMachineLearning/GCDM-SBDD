import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Bool, Float, Int64, jaxtyped
from torch_geometric.data import Batch
from typing import Callable, Literal, Optional, Tuple, Union


ActivationType = Literal["relu", "elu", "leaky_relu", "tanh", "sigmoid", "none", "silu", "swish"]


@beartype
def centralize(
    batch: Batch,
    key: str,
    batch_index: torch.Tensor,
    node_mask: Optional[Bool[torch.Tensor, "n_nodes"]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        # derive centroid of each batch element
        entities_centroid = torch_scatter.scatter(
            batch[key][node_mask],
            batch_index[node_mask],
            dim=0,
            reduce="mean"
        )  # e.g., [batch_size, 3]

        # center entities using corresponding centroids
        entities_centered = batch[key] - (entities_centroid[batch_index] * node_mask.float().unsqueeze(-1))
        masked_values = (
            torch.ones_like(batch[key]) * torch.inf
        )
        values = batch[key][node_mask]
        masked_values[node_mask] = (values - entities_centroid[batch_index][node_mask])
        entities_centered = masked_values

    else:
        # derive centroid of each batch element, and center entities using corresponding centroids
        entities_centroid = torch_scatter.scatter(batch[key], batch_index, dim=0, reduce="mean")  # e.g., [batch_size, 3]
        entities_centered = batch[key] - entities_centroid[batch_index]

    return entities_centroid, entities_centered


@jaxtyped
@beartype
def decentralize(
    batch: Batch,
    key: str,
    batch_index: torch.Tensor,
    entities_centroid: torch.Tensor,
    node_mask: Optional[Bool[torch.Tensor, "n_nodes"]] = None
) -> torch.Tensor:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        masked_values = torch.ones_like(batch[key]) * torch.inf
        masked_values[node_mask] = (batch[key][node_mask] + entities_centroid[batch_index])
        entities_centered = masked_values
    else:
        entities_centered = batch[key] + entities_centroid[batch_index]
    return entities_centered


@jaxtyped
@beartype
def localize(
    pos: Float[torch.Tensor, "batch_num_nodes 3"],
    edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
    norm_pos_diff: bool = True,
    node_mask: Optional[Bool[torch.Tensor, "n_nodes"]] = None
) -> Float[torch.Tensor, "batch_num_edges 3 3"]:
    row, col = edge_index[0], edge_index[1]

    if node_mask is not None:
        edge_mask = node_mask[row] & node_mask[col]

        pos_diff = torch.ones((edge_index.shape[1], 3), device=edge_index.device) * torch.inf
        pos_diff[edge_mask] = pos[row][edge_mask] - pos[col][edge_mask]

        pos_cross = torch.ones((edge_index.shape[1], 3), device=edge_index.device) * torch.inf
        pos_cross[edge_mask] = torch.cross(pos[row][edge_mask], pos[col][edge_mask])
    else:
        pos_diff = pos[row] - pos[col]
        pos_cross = torch.cross(pos[row], pos[col])

    if norm_pos_diff:
        # derive and apply normalization factor for `pos_diff`
        if node_mask is not None:
            norm = torch.ones((edge_index.shape[1], 1), device=pos_diff.device)
            norm[edge_mask] = (
                torch.sqrt(torch.sum((pos_diff[edge_mask] ** 2), dim=1).unsqueeze(1))
            ) + 1
        else:
            norm = torch.sqrt(torch.sum(pos_diff ** 2, dim=1).unsqueeze(1)) + 1
        pos_diff = pos_diff / norm

        # derive and apply normalization factor for `pos_cross`
        if node_mask is not None:
            cross_norm = torch.ones((edge_index.shape[1], 1), device=pos_cross.device)
            cross_norm[edge_mask] = (
                torch.sqrt(torch.sum((pos_cross[edge_mask]) ** 2, dim=1).unsqueeze(1))
            ) + 1
        else:
            cross_norm = (torch.sqrt(torch.sum(pos_cross ** 2, dim=1).unsqueeze(1))) + 1
        pos_cross = pos_cross / cross_norm

    if node_mask is not None:
        pos_vertical = torch.ones((edge_index.shape[1], 3), device=edge_index.device) * torch.inf
        pos_vertical[edge_mask] = torch.cross(pos_diff[edge_mask], pos_cross[edge_mask])
    else:
        pos_vertical = torch.cross(pos_diff, pos_cross)

    f_ij = torch.cat((pos_diff.unsqueeze(1), pos_cross.unsqueeze(1), pos_vertical.unsqueeze(1)), dim=1)
    return f_ij


def get_activations(act_name: ActivationType, return_functional: bool = False) -> Union[nn.Module, Callable]:
    """Maps activation name (str) to activation function module."""
    if act_name == "relu":
        return F.relu if return_functional else nn.ReLU()
    elif act_name == "elu":
        return F.elu if return_functional else nn.ELU()
    elif act_name == "leaky_relu":
        return F.leaky_relu if return_functional else nn.LeakyReLU()
    elif act_name == "tanh":
        return F.tanh if return_functional else nn.Tanh()
    elif act_name == "sigmoid":
        return F.sigmoid if return_functional else nn.Sigmoid()
    elif act_name == "none":
        return nn.Identity()
    elif act_name in {"silu", "swish"}:
        return F.silu if return_functional else nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {act_name}")


@jaxtyped
@beartype
def safe_norm(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
    keepdim: bool = False,
    sqrt: bool = True
) -> torch.Tensor:
    norm = torch.sum(x ** 2, dim=dim, keepdim=keepdim)
    if sqrt:
        norm = torch.sqrt(norm + eps)
    return norm + eps


@jaxtyped
@beartype
def is_identity(nonlinearity: Optional[Union[Callable, nn.Module]] = None) -> bool:
    return nonlinearity is None or isinstance(nonlinearity, nn.Identity)
