###########################################################################################
# Implementation of Geometric-Complete Perceptron layers
#
# Papers:
# (1) Geometry-Complete Perceptron Networks for 3D Molecular Graphs,
#     by A Morehead, J Cheng
# (2) Geometry-Complete Diffusion for 3D Molecule Generation,
#     by A Morehead, J Cheng
#
# Orginal repositories:
# (1) https://github.com/BioinfoMachineLearning/GCPNet
# (2) https://github.com/BioinfoMachineLearning/Bio-Diffusion
###########################################################################################

import hydra
import torch
import torch_scatter

import numpy as np

from beartype import beartype
from copy import copy
from functools import partial
from jaxtyping import Bool, Float, Int64, jaxtyped
from omegaconf import OmegaConf, DictConfig
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from typing import Any, Dict, Optional, Tuple, Union

from equivariant_diffusion.common import centralize, decentralize, get_activations, is_identity, localize, safe_norm
from equivariant_diffusion.wrappers import ScalarVector


class VectorDropout(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate: float):
        super().__init__()
        self.drop_rate = drop_rate

    @jaxtyped
    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = x[0].device
        if not self.training:
            return x
        mask = torch.bernoulli((1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class GCPDropout(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, drop_rate: float, use_gcp_dropout: bool = True):
        super().__init__()
        self.scalar_dropout = nn.Dropout(drop_rate) if use_gcp_dropout else nn.Identity()
        self.vector_dropout = VectorDropout(drop_rate) if use_gcp_dropout else nn.Identity()

    @jaxtyped
    @beartype
    def forward(self, x: Union[torch.Tensor, ScalarVector]) -> Union[torch.Tensor, ScalarVector]:
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (x.scalar.shape[0] == 0 or x.vector.shape[0] == 0):
            return x
        elif isinstance(x, torch.Tensor):
            return self.scalar_dropout(x)
        return ScalarVector(self.scalar_dropout(x[0]), self.vector_dropout(x[1]))


class GCPLayerNorm(nn.Module):
    """
    Adapted from https://github.com/drorlab/gvp-pytorch
    """

    def __init__(self, dims: ScalarVector, eps: float = 1e-8, use_gcp_norm: bool = True):
        super().__init__()
        self.scalar_dims, self.vector_dims = dims
        self.scalar_norm = nn.LayerNorm(self.scalar_dims) if use_gcp_norm else nn.Identity()
        self.use_gcp_norm = use_gcp_norm
        self.eps = eps

    @staticmethod
    @jaxtyped
    @beartype
    def norm_vector(v: torch.Tensor, use_gcp_norm: bool = True, eps: float = 1e-8) -> torch.Tensor:
        v_norm = v
        if use_gcp_norm:
            vector_norm = torch.clamp(torch.sum(torch.square(v), dim=-1, keepdim=True), min=eps)
            vector_norm = torch.sqrt(torch.mean(vector_norm, dim=-2, keepdim=True))
            v_norm = v / vector_norm
        return v_norm

    @jaxtyped
    @beartype
    def forward(self, x: Union[torch.Tensor, ScalarVector]) -> Union[torch.Tensor, ScalarVector]:
        if isinstance(x, torch.Tensor) and x.shape[0] == 0:
            return x
        elif isinstance(x, ScalarVector) and (x.scalar.shape[0] == 0 or x.vector.shape[0] == 0):
            return x
        elif not self.vector_dims:
            return self.scalar_norm(x)
        s, v = x
        return ScalarVector(self.scalar_norm(s), self.norm_vector(v, use_gcp_norm=self.use_gcp_norm, eps=self.eps))


class GCP(nn.Module):
    def __init__(
            self,
            input_dims: ScalarVector,
            output_dims: ScalarVector,
            nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
            scalar_out_nonlinearity: Optional[str] = "silu",
            scalar_gate: int = 0,
            vector_gate: bool = True,
            feedforward_out: bool = False,
            bottleneck: int = 1,
            scalarization_vectorization_output_dim: int = 3,
            enable_e3_equivariance: bool = False,
            **kwargs
    ):
        super().__init__()

        if nonlinearities is None:
            nonlinearities = ("none", "none")

        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.scalar_nonlinearity, self.vector_nonlinearity = (
            get_activations(nonlinearities[0], return_functional=True),
            get_activations(nonlinearities[1], return_functional=True)
        )
        self.scalar_gate, self.vector_gate = scalar_gate, vector_gate
        self.enable_e3_equivariance = enable_e3_equivariance

        if self.scalar_gate > 0:
            self.norm = nn.LayerNorm(self.scalar_output_dim)

        if self.vector_input_dim:
            assert (
                self.vector_input_dim % bottleneck == 0
            ), f"Input channel of vector ({self.vector_input_dim}) must be divisible with bottleneck factor ({bottleneck})"

            self.hidden_dim = self.vector_input_dim // bottleneck if bottleneck > 1 else max(self.vector_input_dim,
                                                                                             self.vector_output_dim)

            scalar_vector_frame_dim = (scalarization_vectorization_output_dim * 3)
            self.vector_down = nn.Linear(self.vector_input_dim, self.hidden_dim, bias=False)
            self.scalar_out = nn.Sequential(
                nn.Linear(self.hidden_dim + self.scalar_input_dim + scalar_vector_frame_dim, self.scalar_output_dim),
                get_activations(scalar_out_nonlinearity),
                nn.Linear(self.scalar_output_dim, self.scalar_output_dim)
            ) if feedforward_out else nn.Linear(self.hidden_dim + self.scalar_input_dim + scalar_vector_frame_dim, self.scalar_output_dim)

            self.vector_down_frames = nn.Linear(
                self.vector_input_dim, scalarization_vectorization_output_dim, bias=False)

            if self.vector_output_dim:
                self.vector_up = nn.Linear(self.hidden_dim, self.vector_output_dim, bias=False)
                if self.vector_gate:
                    self.vector_out_scale = nn.Linear(self.scalar_output_dim, self.vector_output_dim)
        else:
            self.scalar_out = nn.Sequential(
                nn.Linear(self.scalar_input_dim, self.scalar_output_dim),
                get_activations(scalar_out_nonlinearity),
                nn.Linear(self.scalar_output_dim, self.scalar_output_dim)
            ) if feedforward_out else nn.Linear(self.scalar_input_dim, self.scalar_output_dim)

    @jaxtyped
    @beartype
    def create_zero_vector(
        self,
        scalar_rep: Float[torch.Tensor, "batch_num_entities merged_scalar_dim"]
    ) -> Float[torch.Tensor, "batch_num_entities o 3"]:
        return torch.zeros(scalar_rep.shape[0], self.vector_output_dim, 3, device=scalar_rep.device)
    
    @staticmethod
    @jaxtyped
    @beartype
    def scalarize(
        vector_rep: Float[torch.Tensor, "batch_num_entities 3 3"],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_inputs: bool,
        enable_e3_equivariance: bool,
        dim_size: int,
        node_mask: Optional[Bool[torch.Tensor, "n_nodes"]] = None
    ) -> Float[torch.Tensor, "effective_batch_num_entities 9"]:
        row, col = edge_index[0], edge_index[1]

        # gather source node features for each `entity` (i.e., node or edge)
        # note: edge inputs are already ordered according to source nodes
        vector_rep_i = vector_rep[row] if node_inputs else vector_rep

        # project equivariant values onto corresponding local frames
        if vector_rep_i.ndim == 2:
            vector_rep_i = vector_rep_i.unsqueeze(-1)
        elif vector_rep_i.ndim == 3:
            vector_rep_i = vector_rep_i.transpose(-1, -2)

        if node_mask is not None:
            edge_mask = node_mask[row] & node_mask[col]
            local_scalar_rep_i = torch.zeros((edge_index.shape[1], 3, 3), device=edge_index.device)
            local_scalar_rep_i[edge_mask] = torch.matmul(
                frames[edge_mask], vector_rep_i[edge_mask]
            )
            local_scalar_rep_i = local_scalar_rep_i.transpose(-1, -2)
        else:
            local_scalar_rep_i = torch.matmul(frames, vector_rep_i).transpose(-1, -2)

        # potentially enable E(3)-equivariance and, thereby, chirality-invariance
        if enable_e3_equivariance:
            # avoid corrupting gradients with an in-place operation
            local_scalar_rep_i_copy = local_scalar_rep_i.clone()
            local_scalar_rep_i_copy[:, :, 1] = torch.abs(local_scalar_rep_i[:, :, 1])
            local_scalar_rep_i = local_scalar_rep_i_copy

        # reshape frame-derived geometric scalars
        local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], 9)

        if node_inputs:
            # for node inputs, summarize all edge-wise geometric scalars using an average
            return torch_scatter.scatter(
                local_scalar_rep_i,
                # summarize according to source node indices due to the directional nature of GCP's equivariant frames
                row,
                dim=0,
                dim_size=dim_size,
                reduce="mean"
            )

        return local_scalar_rep_i

    @jaxtyped
    @beartype
    def vectorize(
        self,
        scalar_rep: Float[torch.Tensor, "batch_num_entities merged_scalar_dim"],
        vector_hidden_rep: Float[torch.Tensor, "batch_num_entities 3 n"]
    ) -> Float[torch.Tensor, "batch_num_entities o 3"]:
        vector_rep = self.vector_up(vector_hidden_rep)
        vector_rep = vector_rep.transpose(-1, -2)

        if self.vector_gate:
            gate = self.vector_out_scale(self.vector_nonlinearity(scalar_rep))
            vector_rep = vector_rep * torch.sigmoid(gate).unsqueeze(-1)
        elif not is_identity(self.vector_nonlinearity):
            vector_rep = vector_rep * self.vector_nonlinearity(safe_norm(vector_rep, dim=-1, keepdim=True))

        return vector_rep

    @jaxtyped
    @beartype
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                Float[torch.Tensor, "batch_num_entities scalar_dim"],
                Float[torch.Tensor, "batch_num_entities m vector_dim"]
            ],
            Float[torch.Tensor, "batch_num_entities merged_scalar_dim"]
        ],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_inputs: bool = False,
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None
    ) -> Union[
        Tuple[
            Float[torch.Tensor, "batch_num_entities new_scalar_dim"],
            Float[torch.Tensor, "batch_num_entities n vector_dim"]
        ],
        Float[torch.Tensor, "batch_num_entities new_scalar_dim"]
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)

            # curate direction-robust and (by default) chirality-aware scalar geometric features
            vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
            scalar_hidden_rep = self.scalarize(
                vector_down_frames_hidden_rep.transpose(-1, -2),
                edge_index,
                frames,
                node_inputs=node_inputs,
                enable_e3_equivariance=self.enable_e3_equivariance,
                dim_size=vector_down_frames_hidden_rep.shape[0],
                node_mask=node_mask
            )
            merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        else:
            # bypass updating scalar features using vector information
            merged = s_maybe_v

        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using scalar information
            return self.scalar_nonlinearity(scalar_rep)
        elif self.vector_output_dim and not self.vector_input_dim:
            # instantiate vector features that are learnable in proceeding GCP layers
            vector_rep = self.create_zero_vector(scalar_rep)
        else:
            # update vector features using either row-wise scalar gating with complete local frames or row-wise self-scalar gating
            vector_rep = self.vectorize(scalar_rep, vector_hidden_rep)

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        return ScalarVector(scalar_rep, vector_rep)


class GCPEmbedding(nn.Module):
    def __init__(
        self,
        edge_input_dims: ScalarVector,
        node_input_dims: ScalarVector,
        edge_hidden_dims: ScalarVector,
        node_hidden_dims: ScalarVector,
        num_atom_types: int = 0,
        nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
        cfg: DictConfig = None,
        pre_norm: bool = True,
        use_gcp_norm: bool = True
    ):
        super().__init__()

        if num_atom_types > 0:
            self.atom_embedding = nn.Embedding(num_atom_types, num_atom_types)
        else:
            self.atom_embedding = None

        self.pre_norm = pre_norm
        if pre_norm:
            self.edge_normalization = GCPLayerNorm(edge_input_dims, use_gcp_norm=use_gcp_norm)
            self.node_normalization = GCPLayerNorm(node_input_dims, use_gcp_norm=use_gcp_norm)
        else:
            self.edge_normalization = GCPLayerNorm(edge_hidden_dims, use_gcp_norm=use_gcp_norm)
            self.node_normalization = GCPLayerNorm(node_hidden_dims, use_gcp_norm=use_gcp_norm)

        self.edge_embedding = GCP(
            edge_input_dims,
            edge_hidden_dims,
            nonlinearities=nonlinearities,
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance
        )

        self.node_embedding = GCP(
            node_input_dims,
            node_hidden_dims,
            nonlinearities=("none", "none"),
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance
        )

    @jaxtyped
    @beartype
    def forward(
        self,
        batch: Batch
    ) -> Tuple[
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_nodes h_hidden_dim"],
                Float[torch.Tensor, "batch_num_nodes m chi_hidden_dim"]
            ],
            Float[torch.Tensor, "batch_num_nodes h_hidden_dim"]
        ],
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_edges e_hidden_dim"],
                Float[torch.Tensor, "batch_num_edges x xi_hidden_dim"]
            ],
            Float[torch.Tensor, "batch_num_edges e_hidden_dim"]
        ]
    ]:
        if self.atom_embedding is not None:
            node_rep = ScalarVector(self.atom_embedding(batch.h), batch.chi)
        else:
            node_rep = ScalarVector(batch.h, batch.chi)

        edge_rep = ScalarVector(batch.e, batch.xi)

        edge_rep = edge_rep.scalar if not self.edge_embedding.vector_input_dim else edge_rep
        node_rep = node_rep.scalar if not self.node_embedding.vector_input_dim else node_rep

        if self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        edge_rep = self.edge_embedding(
            edge_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=False,
            node_mask=getattr(batch, "mask", None)
        )
        node_rep = self.node_embedding(
            node_rep,
            batch.edge_index,
            batch.f_ij,
            node_inputs=True,
            node_mask=getattr(batch, "mask", None)
        )

        if not self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)

        return node_rep, edge_rep
    

@beartype
def get_GCP_with_custom_cfg(input_dims: Any, output_dims: Any, cfg: DictConfig, **kwargs):
    cfg_dict = copy(OmegaConf.to_container(cfg, throw_on_missing=True))
    cfg_dict["nonlinearities"] = cfg.nonlinearities
    del cfg_dict["scalar_nonlinearity"]
    del cfg_dict["vector_nonlinearity"]

    for key in kwargs:
        cfg_dict[key] = kwargs[key]

    return GCP(input_dims, output_dims, **cfg_dict)


class GCPMessagePassing(nn.Module):
    def __init__(
        self,
        input_dims: ScalarVector,
        output_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: DictConfig,
        mp_cfg: DictConfig,
        reduce_function: str = "sum",
        use_scalar_message_attention: bool = True
    ):
        super().__init__()

        # hyperparameters
        self.scalar_input_dim, self.vector_input_dim = input_dims
        self.scalar_output_dim, self.vector_output_dim = output_dims
        self.edge_scalar_dim, self.edge_vector_dim = edge_dims
        self.conv_cfg = mp_cfg
        self.self_message = self.conv_cfg.self_message
        self.reduce_function = reduce_function
        self.use_scalar_message_attention = use_scalar_message_attention

        scalars_in_dim = 2 * self.scalar_input_dim + self.edge_scalar_dim
        vectors_in_dim = 2 * self.vector_input_dim + self.edge_vector_dim

        # config instantiations
        soft_cfg = copy(cfg)
        soft_cfg.bottleneck = cfg.default_bottleneck

        primary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=soft_cfg)
        secondary_cfg_GCP = partial(get_GCP_with_custom_cfg, cfg=cfg)

        # PyTorch modules #
        module_list = [
            primary_cfg_GCP(
                (scalars_in_dim, vectors_in_dim),
                output_dims,
                nonlinearities=cfg.nonlinearities,
                enable_e3_equivariance=cfg.enable_e3_equivariance
            )
        ]

        for _ in range(self.conv_cfg.num_message_layers - 2):
            module_list.append(
                secondary_cfg_GCP(
                    output_dims,
                    output_dims,
                    enable_e3_equivariance=cfg.enable_e3_equivariance
                )
            )

        if self.conv_cfg.num_message_layers > 1:
            module_list.append(
                primary_cfg_GCP(
                    output_dims,
                    output_dims,
                    nonlinearities=cfg.nonlinearities,
                    enable_e3_equivariance=cfg.enable_e3_equivariance
                )
            )

        self.message_fusion = nn.ModuleList(module_list)

        # learnable scalar message gating
        if use_scalar_message_attention:
            self.scalar_message_attention = nn.Sequential(
                nn.Linear(output_dims.scalar, 1),
                nn.Sigmoid()
            )

    @jaxtyped
    @beartype
    def message(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None
    ) -> Float[torch.Tensor, "batch_num_edges message_dim"]:
        row, col = edge_index
        vector = node_rep.vector.reshape(node_rep.vector.shape[0], node_rep.vector.shape[1] * node_rep.vector.shape[2])
        vector_reshaped = ScalarVector(node_rep.scalar, vector)

        s_row, v_row = vector_reshaped.idx(row)
        s_col, v_col = vector_reshaped.idx(col)

        v_row = v_row.reshape(v_row.shape[0], v_row.shape[1] // 3, 3)
        v_col = v_col.reshape(v_col.shape[0], v_col.shape[1] // 3, 3)

        message = ScalarVector(s_row, v_row).concat((edge_rep, ScalarVector(s_col, v_col)))
        
        message_residual = self.message_fusion[0](message, edge_index, frames, node_inputs=False, node_mask=node_mask)
        for module in self.message_fusion[1:]:
            # exchange geometric messages while maintaining residual connection to original message
            new_message = module(message_residual, edge_index, frames, node_inputs=False, node_mask=node_mask)
            message_residual = message_residual + new_message

        # learn to gate scalar messages
        if self.use_scalar_message_attention:
            message_residual_attn = self.scalar_message_attention(message_residual.scalar)
            message_residual = ScalarVector(message_residual.scalar * message_residual_attn, message_residual.vector)

        return message_residual.flatten()

    @jaxtyped
    @beartype
    def aggregate(
        self,
        message: Float[torch.Tensor, "batch_num_edges message_dim"],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        dim_size: int
    ) -> Float[torch.Tensor, "batch_num_nodes aggregate_dim"]:
        row, col = edge_index
        aggregate = torch_scatter.scatter(message, row, dim=0, dim_size=dim_size, reduce=self.reduce_function)
        return aggregate

    @jaxtyped
    @beartype
    def forward(
        self,
        node_rep: ScalarVector,
        edge_rep: ScalarVector,
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None
    ) -> ScalarVector:
        message = self.message(node_rep, edge_rep, edge_index, frames, node_mask=node_mask)
        aggregate = self.aggregate(message, edge_index, dim_size=node_rep.scalar.shape[0])
        return ScalarVector.recover(aggregate, self.vector_output_dim)


class GCPInteractions(nn.Module):
    def __init__(
        self,
        node_dims: ScalarVector,
        edge_dims: ScalarVector,
        cfg: DictConfig,
        layer_cfg: DictConfig,
        dropout: float = 0.0,
        nonlinearities: Optional[Tuple[Any, Any]] = None
    ):
        super().__init__()

        # hyperparameters #
        if nonlinearities is None:
            nonlinearities = cfg.nonlinearities
        self.pre_norm = layer_cfg.pre_norm
        self.predict_node_positions = getattr(cfg, "predict_node_positions", False)
        self.node_positions_weight = getattr(cfg, "node_positions_weight", 1.0)
        self.update_positions_with_vector_sum = getattr(cfg, "update_positions_with_vector_sum", False)
        reduce_function = "sum"

        # PyTorch modules #

        # geometry-complete message-passing neural network
        message_function = GCPMessagePassing

        self.interaction = message_function(
            node_dims,
            node_dims,
            edge_dims,
            cfg=cfg,
            mp_cfg=layer_cfg.mp_cfg,
            reduce_function=reduce_function,
            use_scalar_message_attention=layer_cfg.use_scalar_message_attention
        )

        # config instantiations
        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_GCP = partial(get_GCP_with_custom_cfg, cfg=ff_cfg)

        self.gcp_norm = nn.ModuleList([GCPLayerNorm(node_dims, use_gcp_norm=layer_cfg.use_gcp_norm)])
        self.gcp_dropout = nn.ModuleList([GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout)])

        # build out feedforward (FF) network modules
        hidden_dims = (
            (node_dims.scalar, node_dims.vector)
            if layer_cfg.num_feedforward_layers == 1
            else (4 * node_dims.scalar, 2 * node_dims.vector)
        )
        ff_interaction_layers = [
            ff_GCP(
                (node_dims.scalar * 2, node_dims.vector * 2),
                hidden_dims,
                nonlinearities=("none", "none") if layer_cfg.num_feedforward_layers == 1 else cfg.nonlinearities,
                feedforward_out=layer_cfg.num_feedforward_layers == 1,
                enable_e3_equivariance=cfg.enable_e3_equivariance
            )
        ]

        interaction_layers = [
            ff_GCP(hidden_dims, hidden_dims, enable_e3_equivariance=cfg.enable_e3_equivariance)
            for _ in range(layer_cfg.num_feedforward_layers - 2)
        ]
        ff_interaction_layers.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_interaction_layers.append(
                ff_GCP(
                    hidden_dims, node_dims,
                    nonlinearities=("none", "none"),
                    feedforward_out=True,
                    enable_e3_equivariance=cfg.enable_e3_equivariance
                )
            )

        self.feedforward_network = nn.ModuleList(ff_interaction_layers)

        # potentially build out node position update modules
        if self.predict_node_positions:
            # node position update GCP(s)
            position_output_dims = (
                node_dims
                if getattr(cfg, "update_positions_with_vector_sum", False)
                else (node_dims.scalar, 1)
            )
            self.node_position_update_gcp = ff_GCP(
                node_dims, position_output_dims,
                nonlinearities=cfg.nonlinearities,
                enable_e3_equivariance=cfg.enable_e3_equivariance
            )

    @jaxtyped
    @beartype
    def derive_x_update(
        self,
        node_rep: ScalarVector,
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        f_ij: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
        fixed_pos_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
        fix_pos: bool = False
    ) -> Float[torch.Tensor, "batch_num_nodes 3"]:
        # use vector-valued features to derive node position updates
        node_rep_update = self.node_position_update_gcp(
            node_rep,
            edge_index,
            f_ij,
            node_inputs=True,
            node_mask=node_mask
        )
        if self.update_positions_with_vector_sum:
            x_vector_update = node_rep_update.vector.sum(1)
        else:
            x_vector_update = node_rep_update.vector.squeeze(1)

        # (up/down)weight position updates
        x_update = x_vector_update * self.node_positions_weight

        # mask out updates to fixed positions
        if not fix_pos and fixed_pos_mask is not None:
            x_update = x_update * fixed_pos_mask.unsqueeze(-1)

        return x_update

    @jaxtyped
    @beartype
    def forward(
        self,
        node_rep: Tuple[Float[torch.Tensor, "batch_num_nodes node_hidden_dim"], Float[torch.Tensor, "batch_num_nodes m 3"]],
        edge_rep: Tuple[Float[torch.Tensor, "batch_num_edges edge_hidden_dim"], Float[torch.Tensor, "batch_num_edges x 3"]],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
        fixed_pos_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
        node_pos: Optional[Float[torch.Tensor, "batch_num_nodes 3"]] = None,
        fix_pos: bool = False
    ) -> Tuple[
            Tuple[
                Float[torch.Tensor, "batch_num_nodes hidden_dim"],
                Float[torch.Tensor, "batch_num_nodes n 3"]
            ],
            Optional[Float[torch.Tensor, "batch_num_nodes 3"]]
        ]:
        node_rep = ScalarVector(node_rep[0], node_rep[1])
        edge_rep = ScalarVector(edge_rep[0], edge_rep[1])

        # apply GCP normalization (1)
        if self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # forward propagate with interaction module
        hidden_residual = self.interaction(
            node_rep, edge_rep, edge_index, frames, node_mask=node_mask
        )

        # aggregate input and hidden features
        hidden_residual = ScalarVector(*hidden_residual.concat((node_rep,)))

        # propagate with feedforward layers
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                edge_index,
                frames,
                node_inputs=True,
                node_mask=node_mask
            )

        # apply GCP dropout
        node_rep = node_rep + self.gcp_dropout[0](hidden_residual)

        # apply GCP normalization (2)
        if not self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # update only unmasked node representations and residuals
        if node_mask is not None:
            node_rep = node_rep.mask(node_mask.float())

        # bypass updating node positions
        if not self.predict_node_positions:
            return node_rep, node_pos

        # update node positions
        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frames, node_mask=node_mask, fixed_pos_mask=fixed_pos_mask, fix_pos=fix_pos
        )

        # update only unmasked node positions
        if node_mask is not None:
            node_pos = node_pos * node_mask.float().unsqueeze(-1)

        return node_rep, node_pos


class GCPNetDynamics(torch.nn.Module):
    def __init__(
        self,
        gcpnet_cfg: DictConfig,
        num_atom_features: int,
        num_residue_features: int,
        num_joint_features: int,
        num_dims: int,
        shared_feature_space_nonlinearity: nn.Module,
        condition_on_time: bool = True,
        update_pocket_coords: bool = True,
        edge_cutoff: Optional[float] = None,
    ):
        super().__init__()

        model_cfg = gcpnet_cfg.model_cfg
        module_cfg = gcpnet_cfg.module_cfg
        layer_cfg = gcpnet_cfg.layer_cfg

        self.predict_node_pos = module_cfg.predict_node_positions
        self.predict_node_rep = module_cfg.predict_node_rep
        self.condition_on_time = condition_on_time
        self.update_pocket_coords = update_pocket_coords
        self.edge_cutoff = edge_cutoff

        # Feature dimensionalities
        self.num_dims = num_dims
        edge_input_dims = ScalarVector(model_cfg.e_input_dim, model_cfg.xi_input_dim)
        node_input_dims = ScalarVector(num_joint_features + condition_on_time, model_cfg.chi_input_dim)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)

        # Position-wise operations
        self.centralize = partial(centralize, key="pos")
        self.localize = partial(localize, norm_pos_diff=module_cfg.norm_pos_diff)
        self.decentralize = partial(decentralize, key="pos")

        # Shared feature space MLPs
        self.atom_encoder = nn.Sequential(
            nn.Linear(num_atom_features, 2 * num_atom_features),
            shared_feature_space_nonlinearity,
            nn.Linear(2 * num_atom_features, num_joint_features)
        )
        self.atom_decoder = nn.Sequential(
            nn.Linear(num_joint_features, 2 * num_atom_features),
            shared_feature_space_nonlinearity,
            nn.Linear(2 * num_atom_features, num_atom_features)
        )
        self.residue_encoder = nn.Sequential(
            nn.Linear(num_residue_features, 2 * num_residue_features),
            shared_feature_space_nonlinearity,
            nn.Linear(2 * num_residue_features, num_joint_features)
        )
        if update_pocket_coords:
            self.residue_decoder = nn.Sequential(
                nn.Linear(num_joint_features, 2 * num_residue_features),
                shared_feature_space_nonlinearity,
                nn.Linear(2 * num_residue_features, num_residue_features)
            )

        # Input embeddings
        self.gcp_embedding = GCPEmbedding(
            edge_input_dims,
            node_input_dims,
            self.edge_dims,
            self.node_dims,
            cfg=module_cfg
        )

        # Message-passing layers
        self.interaction_layers = nn.ModuleList(
            GCPInteractions(
                self.node_dims,
                self.edge_dims,
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=model_cfg.dropout
            ) for _ in range(model_cfg.num_layers)
        )

        if self.predict_node_rep:
            # Predictions
            self.invariant_node_projection = GCP(
                # Note: `GCPNet` defaults to providing SE(3) equivariance
                # It is possible to provide E(3) equivariance by instead setting `module_cfg.enable_e3_equivariance=true`
                self.node_dims,
                (node_input_dims.scalar, 0),
                nonlinearities=tuple(module_cfg.nonlinearities),
                scalar_gate=module_cfg.scalar_gate,
                vector_gate=module_cfg.vector_gate,
                enable_e3_equivariance=module_cfg.enable_e3_equivariance,
                node_inputs=True
            )

    @staticmethod
    @jaxtyped
    @beartype
    def get_edges(
        batch_mask: torch.Tensor,
        x: torch.Tensor,
        edge_cutoff: Optional[float] = None,
    ) -> torch.Tensor:
        adj = batch_mask[:, None] == batch_mask[None, :]
        if edge_cutoff is not None:
            adj = adj & (torch.cdist(x, x) <= edge_cutoff)
        edges = torch.stack(torch.where(adj), dim=0)
        return edges

    
    @staticmethod
    @jaxtyped
    @beartype
    def normalize(
        tensor: torch.Tensor,
        dim: int = -1
    ) -> torch.Tensor:
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
        )

    @staticmethod
    @jaxtyped
    @beartype
    def compute_orientations(X: torch.Tensor) -> torch.Tensor:
        forward = GCPNetDynamics.normalize(X[1:] - X[:-1])
        backward = GCPNetDynamics.normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat((forward.unsqueeze(-2), backward.unsqueeze(-2)), dim=-2)
    
    @staticmethod
    @jaxtyped
    @beartype
    def compute_node_features(
        batch: Batch,
        coords_key: str = "pos"
    ) -> Tuple[
        Union[Dict[str, torch.Tensor], Optional[torch.Tensor]],
        torch.Tensor
    ]:
        # construct invariant node features
        if hasattr(batch, "h"):
            node_s = batch.h
        else:
            node_s = None

        # build equivariant node features
        coords = batch[coords_key]
        orientations = GCPNetDynamics.compute_orientations(coords)
        node_v = torch.nan_to_num(orientations)

        return node_s, node_v
    
    @staticmethod
    @jaxtyped
    @beartype
    def compute_edge_features(
        batch: Batch,
        coords_key: str = "pos"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = batch[coords_key]
        E_vectors = coords[batch.edge_index[0]] - coords[batch.edge_index[1]]
        radial = torch.sum(E_vectors ** 2, dim=1).unsqueeze(-1)

        edge_s = radial
        edge_v = GCPNetDynamics.normalize(E_vectors).unsqueeze(-2)

        edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

        return edge_s, edge_v

    @jaxtyped
    @beartype
    def forward(
        self,
        xh_atoms: torch.Tensor,
        xh_residues: torch.Tensor,
        t: torch.Tensor,
        mask_atoms: torch.Tensor,
        mask_residues: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Separate atom features
        x_atoms = xh_atoms[:, :self.num_dims].clone()
        h_atoms = xh_atoms[:, self.num_dims:].clone()
        x_residues = xh_residues[:, :self.num_dims].clone()
        h_residues = xh_residues[:, self.num_dims:].clone()

        # Embed atom features and residue features in a shared space
        h_atoms = self.atom_encoder(h_atoms)
        h_residues = self.residue_encoder(h_residues)

        # Combine the two node types
        x = torch.cat((x_atoms, x_residues), dim=0)
        h = torch.cat((h_atoms, h_residues), dim=0)
        x_init = x.clone()
        h_init = h.clone()
        mask = torch.cat([mask_atoms, mask_residues])

        # Construct `Batch` from individual features
        batch = Batch(pos=x, h=h, batch=mask)

        # Get edges of a complete graph
        batch.edge_index = self.get_edges(batch.batch, batch.pos, edge_cutoff=self.edge_cutoff)

        # install noisy node positions into graph
        # Centralize node positions to make them translation-invariant
        batch.pos = x_init
        x_init = self.centralize(batch, batch_index=batch.batch)[-1].clone()
        batch.pos = x_init
        
        # Install noisy scalar and vector-valued features for nodes and edges into graph
        batch.h, batch.chi = h_init, self.compute_node_features(batch)[-1]
        batch.e, batch.xi = self.compute_edge_features(batch)

        # Condition model's predictions on the current time step
        if self.condition_on_time:
            if np.prod(t.size()) == 1:
                # Note: Here, `t` is the same for all elements in batch
                h_time = torch.empty_like(
                    batch.h[:, 0:1],
                    device=batch.pos.device
                ).fill_(t.item())
            else:
                # Note: Here, `t` is different over the batch dimension
                h_time = t[mask]
            batch.h = torch.cat([batch.h, h_time], dim=1)

        # Determine which positions to update
        pos_update_mask = (
            None
            if self.update_pocket_coords
            else torch.cat((torch.ones_like(mask_atoms), torch.zeros_like(mask_residues))).bool()
        )

        # Begin GCPNet forward pass #

        # Craft complete local frames corresponding to each edge
        batch.f_ij = self.localize(batch.pos, batch.edge_index)

        # Embed node and edge input features
        (h, chi), (e, xi) = self.gcp_embedding(batch)

        # Update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            (h, chi), batch.pos = layer(
                (h, chi),
                (e, xi),
                batch.edge_index,
                batch.f_ij,
                node_pos=batch.pos,
                fixed_pos_mask=pos_update_mask,
            )

        # Summarize intermediate node representations as final predictions
        if self.predict_node_rep:
            batch.h = self.invariant_node_projection(
                ScalarVector(h, chi),
                batch.edge_index,
                batch.f_ij,
                node_inputs=True
            )

        # Record final version of each feature in `Batch` object
        batch.chi, batch.e, batch.xi = chi, e, xi

        # End GCPNet forward pass #

        vel = (batch.pos - x_init)  # Use delta(x) to estimate noise
        h_final = batch.h

        if self.condition_on_time:
            # Slice off last dimension which represents time
            h_final = h_final[:, :-1]

        # Decode atom and residue features
        h_final_atoms = self.atom_decoder(h_final[:len(mask_atoms)])
        h_final_residues = (
            self.residue_decoder(h_final[len(mask_atoms):])
            if self.update_pocket_coords
            else h_final[len(mask_atoms):]
        )

        # Detect and nullify any invalid node position predictions
        if vel.isnan().any():
            print(f"Detected NaN in `vel` -> resetting GCPNet `vel` output for time step(s) {t} to zero.")
            vel = torch.zeros_like(vel)

        if self.update_pocket_coords:
            # In case of unconditional joint distribution, include this as in
            # the original code
            batch.vel = vel
            _, vel = centralize(batch, key="vel", batch_index=batch.batch)
            del batch.vel

        return torch.cat([vel[:len(mask_atoms)], h_final_atoms], dim=-1), \
               torch.cat([vel[len(mask_atoms):], h_final_residues], dim=-1)
    

@hydra.main(version_base="1.3", config_path="configs", config_name="gcpnet.yaml")
def main(cfg: DictConfig):
    enc = hydra.utils.instantiate(cfg)
    print(enc)


if __name__ == "__main__":
    main()
