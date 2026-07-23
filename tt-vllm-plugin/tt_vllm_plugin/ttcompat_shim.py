"""Compatibility shim: expose symbols the tt-vllm-plugin imports from
`vllm.utils` that fork vllm(22be241) moved elsewhere or changed shape.

The plugin is authored against pypi vllm(<0.11), where these live under
`vllm.utils`. Fork vllm relocated them:
  - STR_DTYPE_TO_TORCH_DTYPE  -> vllm.utils.torch_utils
  - cdiv                      -> vllm.utils.math_utils
  - LayerBlockType (Enum-ish) -> vllm.config.model defines it as a typing Literal;
                                 plugin uses LayerBlockType.attention, so provide a
                                 tiny stand-in exposing `.attention` = "attention".
Numerical/model logic is untouched; this only restores import symbols.
"""
import vllm.utils as _u

try:
    _u.STR_DTYPE_TO_TORCH_DTYPE
except AttributeError:
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE as _s
    _u.STR_DTYPE_TO_TORCH_DTYPE = _s

try:
    _u.cdiv
except AttributeError:
    from vllm.utils.math_utils import cdiv as _c
    _u.cdiv = _c

try:
    _u.LayerBlockType
except AttributeError:
    class _LayerBlockType:
        attention = "attention"
        linear_attention = "linear_attention"
        mamba = "mamba"
    _u.LayerBlockType = _LayerBlockType
