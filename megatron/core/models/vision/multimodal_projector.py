from megatron.core import tensor_parallel
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor
import torch.nn.functional as F
import copy
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
try:
    import apex

    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    HAVE_APEX = True
    LNImpl = FusedLayerNorm
except ImportError:
    import warnings

    from megatron.core.transformer.torch_layer_norm import WrappedTorchLayerNorm

    warnings.warn(f'Apex is not installed. Falling back to Torch LayerNorm')
    LNImpl = WrappedTorchLayerNorm


class MultimodalProjector(MegatronModule):
    """
    MultimodalProjector will take the encoded input with input_size hidden state and project
    it into the hidden size of the language model for multimodal training. When projector is
    type affine linear_fc1 from submodules is used.

    Args:
        transformer_config (TransformerConfig): Transformer config
        submodules (MLPSubmodules): Specifies MLP submodules for mlp type projector
        projector_type (str): Projector type
        input_size (int): Input size from feature encoder
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        projector_type: str,
        input_size: int,
    ):

        
        super().__init__(config=copy.deepcopy(config))
        self.projector_type = projector_type

        assert submodules is not None, "MLPSubmodules must be provided"
        
        if self.projector_type == "affine":
            self.encoder = build_module(
                submodules.linear_fc1,
                input_size,
                config.hidden_size,
                config=config,
                init_method=config.init_method,
                gather_output=True,
                bias=True,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=None,
            )
        elif self.projector_type == "linear+mlp":
            self.linear_proj = build_module(
                    ColumnParallelLinear,  
                    input_size,
                    config.hidden_size,
                    config=config,
                    init_method=config.init_method,
                    gather_output=True,
                    bias=config.add_bias_linear,
                    skip_bias_add=True,
                    is_expert=False,
                    tp_comm_buffer_name=None,
                )
            # we use layernorm here!
            self.config.normalization = "LayerNorm"
            self.input_layernorm = build_module(
                LNImpl,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        
            self.mlp = MLP(config=self.config, submodules=submodules, input_size=config.hidden_size)
        else:
            print(f'#########{self.projector_type}')
            raise



    def forward(self, hidden_states):
        # Run encoder.
        if self.projector_type == "affine":
            x, encoder_output_bias = self.encoder(hidden_states)
        elif self.projector_type == "linear+mlp":
            x, _ = self.linear_proj(hidden_states)

            x = self.input_layernorm(x)

            x = F.gelu(x)

            x, encoder_output_bias = self.mlp(x)

        if encoder_output_bias is not None:
            x = x + encoder_output_bias

        return x
