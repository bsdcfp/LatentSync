# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from tryon.sgm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, TryOnUNetModel

sd3_common_transformer_block_config = {
    "dummy_input": {
        "hidden_states": (2, 4096, 1536),
        "encoder_hidden_states": (2, 333, 1536),
        "temb": (2, 1536),
    },
    "output_names": ["encoder_hidden_states_out", "hidden_states_out"],
    "dynamic_axes": {
        "hidden_states": {0: "batch_size"},
        "encoder_hidden_states": {0: "batch_size"},
        "temb": {0: "steps"},
    },
}

context_size = [
    (2, 320, 96, 72),
    (2, 320, 96, 72),
    (2, 640, 48, 36),
    (2, 640, 48, 36),
    (2, 1280, 24, 18),
    (2, 1280, 24, 18),
    (2, 1280, 12, 9),
    (2, 1280, 12, 9),
    (2, 1280, 12, 9),
    (2, 1280, 12, 9),
    (2, 1280, 12, 9),
    (2, 1280, 24, 18),
    (2, 1280, 24, 18),
    (2, 1280, 24, 18),
    (2, 1280, 48, 36),
    (2, 640, 48, 36),
    (2, 640, 48, 36),
    (2, 640, 96, 72),
    (2, 320, 96, 72),
    (2, 320, 96, 72),
    (2, 320, 96, 72),
]

ONNX_CONFIG = {
    UNet2DConditionModel: {
        "down_blocks.0": {
            "dummy_input": {
                "hidden_states": (2, 320, 128, 128),
                "temb": (2, 1280),
            },
            "output_names": ["sample", "res_samples_0", "res_samples_1", "res_samples_2"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
            },
        },
        "down_blocks.1": {
            "dummy_input": {
                "hidden_states": (2, 320, 64, 64),
                "temb": (2, 1280),
                "encoder_hidden_states": (2, 77, 2048),
            },
            "output_names": ["sample", "res_samples_0", "res_samples_1", "res_samples_2"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size"},
            },
        },
        "down_blocks.2": {
            "dummy_input": {
                "hidden_states": (2, 640, 32, 32),
                "temb": (2, 1280),
                "encoder_hidden_states": (2, 77, 2048),
            },
            "output_names": ["sample", "res_samples_0", "res_samples_1"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size"},
            },
        },
        "mid_block": {
            "dummy_input": {
                "hidden_states": (2, 1280, 32, 32),
                "temb": (2, 1280),
                "encoder_hidden_states": (2, 77, 2048),
            },
            "output_names": ["sample"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size"},
            },
        },
        "up_blocks.0": {
            "dummy_input": {
                "hidden_states": (2, 1280, 32, 32),
                "res_hidden_states_0": (2, 640, 32, 32),
                "res_hidden_states_1": (2, 1280, 32, 32),
                "res_hidden_states_2": (2, 1280, 32, 32),
                "temb": (2, 1280),
                "encoder_hidden_states": (2, 77, 2048),
            },
            "output_names": ["sample"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size"},
                "res_hidden_states_0": {0: "batch_size"},
                "res_hidden_states_1": {0: "batch_size"},
                "res_hidden_states_2": {0: "batch_size"},
            },
        },
        "up_blocks.1": {
            "dummy_input": {
                "hidden_states": (2, 1280, 64, 64),
                "res_hidden_states_0": (2, 320, 64, 64),
                "res_hidden_states_1": (2, 640, 64, 64),
                "res_hidden_states_2": (2, 640, 64, 64),
                "temb": (2, 1280),
                "encoder_hidden_states": (2, 77, 2048),
            },
            "output_names": ["sample"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "encoder_hidden_states": {0: "batch_size"},
                "res_hidden_states_0": {0: "batch_size"},
                "res_hidden_states_1": {0: "batch_size"},
                "res_hidden_states_2": {0: "batch_size"},
            },
        },
        "up_blocks.2": {
            "dummy_input": {
                "hidden_states": (2, 640, 128, 128),
                "res_hidden_states_0": (2, 320, 128, 128),
                "res_hidden_states_1": (2, 320, 128, 128),
                "res_hidden_states_2": (2, 320, 128, 128),
                "temb": (2, 1280),
            },
            "output_names": ["sample"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                "temb": {0: "steps"},
                "res_hidden_states_0": {0: "batch_size"},
                "res_hidden_states_1": {0: "batch_size"},
                "res_hidden_states_2": {0: "batch_size"},
            },
        },
    },
    # SD3Transformer2DModel: {
    #     **{f"transformer_blocks.{i}": sd3_common_transformer_block_config for i in range(23)},
    #     "transformer_blocks.23": {
    #         "dummy_input": {
    #             "hidden_states": (2, 4096, 1536),
    #             "encoder_hidden_states": (2, 333, 1536),
    #             "temb": (2, 1536),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "encoder_hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #         },
    #     },
    # },
    TryOnUNetModel: {
        "blocks": {
            "dummy_input": {
                "hidden_states": (2, 12, 96, 72),
                "temb": (2, 1280),
                **{
                    f"context_{i}": (2, *item[1:])
                    for i, item in enumerate(context_size)
                },
                "clip_context": (2, 1, 1280),
            },
            "output_names": ["hidden_states_out"],
            "dynamic_axes": {
                "hidden_states": {0: "batch_size"},
                **{"context_%d" % i: {0: "batch_size"} for i in range(len(context_size))},
                "clip_context": {0: "batch_size"},
                "temb": {0: "steps"},
            },
        },
    },
    # TryOnUNetModel: {
    #     "input_blocks.0": {
    #         "dummy_input": {
    #             "hidden_states": (2, 12, 96, 72),
    #             # "temb": (2, 1280),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             # "temb": {0: "steps"},
    #         },
    #     },
    #     "input_blocks.1": {
    #         "dummy_input": {
    #             "hidden_states": (2, 320, 96, 72),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 320, 96, 72),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "input_blocks.2": {
    #         "dummy_input": {
    #             "hidden_states": (2, 320, 96, 72),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 320, 96, 72),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "input_blocks.3": {
    #         "dummy_input": {
    #             "hidden_states": (2, 320, 96, 72),
    #             # "temb": (2, 1280),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             # "temb": {0: "steps"},
    #         },
    #     },
    #     "input_blocks.4": {
    #         "dummy_input": {
    #             "hidden_states": (2, 320, 48, 36),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 640, 48, 36),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "input_blocks.5": {
    #         "dummy_input": {
    #             "hidden_states": (2, 640, 48, 36),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 640, 48, 36),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "input_blocks.6": {
    #         "dummy_input": {
    #             "hidden_states": (2, 640, 48, 36),
    #             # "temb": (2, 1280),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             # "temb": {0: "steps"},
    #         },
    #     },
    #     "input_blocks.7": {
    #         "dummy_input": {
    #             "hidden_states": (2, 640, 24, 18),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 1280, 24, 18),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "input_blocks.8": {
    #         "dummy_input": {
    #             "hidden_states": (2, 1280, 24, 18),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 1280, 24, 18),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "input_blocks.9": {
    #         "dummy_input": {
    #             "hidden_states": (2, 1280, 24, 18),
    #             # "temb": (2, 1280),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             # "temb": {0: "steps"},
    #         },
    #     },
    #     "input_blocks.10": {
    #         "dummy_input": {
    #             "hidden_states": (2, 1280, 12, 9),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 1280, 12, 9),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "input_blocks.11": {
    #         "dummy_input": {
    #             "hidden_states": (2, 1280, 12, 9),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 1280, 12, 9),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "middle_block": {
    #         "dummy_input": {
    #             "hidden_states": (2, 1280, 12, 9),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 1280, 12, 9),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.0": {
    #         "dummy_input": {
    #             "hidden_states": (2, 2560, 12, 9),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 1280, 12, 9),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.1": {
    #         "dummy_input": {
    #             "hidden_states": (2, 2560, 12, 9),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 1280, 12, 9),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.2": {
    #         "dummy_input": {
    #             "hidden_states": (2, 2560, 12, 9),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 1280, 24, 18),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.3": {
    #         "dummy_input": {
    #             "hidden_states": (2, 2560, 24, 18),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 1280, 24, 18),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.4": {
    #         "dummy_input": {
    #             "hidden_states": (2, 2560, 24, 18),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 1280, 24, 18),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.5": {
    #         "dummy_input": {
    #             "hidden_states": (2, 1920, 24, 18),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 1280, 48, 36),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.6": {
    #         "dummy_input": {
    #             "hidden_states": (2, 1920, 48, 36),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 640, 48, 36),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.7": {
    #         "dummy_input": {
    #             "hidden_states": (2, 1280, 48, 36),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 640, 48, 36),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.8": {
    #         "dummy_input": {
    #             "hidden_states": (2, 960, 48, 36),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 640, 96, 72),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.9": {
    #         "dummy_input": {
    #             "hidden_states": (2, 960, 96, 72),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 320, 96, 72),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.10": {
    #         "dummy_input": {
    #             "hidden_states": (2, 640, 96, 72),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 320, 96, 72),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    #     "output_blocks.11": {
    #         "dummy_input": {
    #             "hidden_states": (2, 640, 96, 72),
    #             "temb": (2, 1280),
    #             "context0": (2, 1, 1280),
    #             "context1": (2, 320, 96, 72),
    #         },
    #         "output_names": ["hidden_states_out"],
    #         "dynamic_axes": {
    #             "hidden_states": {0: "batch_size"},
    #             "temb": {0: "steps"},
    #             "context0": {0: "batch_size"},
    #             "context1": {0: "batch_size"},
    #         },
    #     },
    # },

}
