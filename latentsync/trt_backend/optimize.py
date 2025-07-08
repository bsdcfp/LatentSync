import onnx_graphsurgeon as gs
from polygraphy.backend.onnx.loader import fold_constants
from onnx import shape_inference
import tempfile
import onnx
import os
import re


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        return gs.export_onnx(self.graph) if return_onnx else self.graph

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(
            gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True
        )
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            temp_dir = tempfile.TemporaryDirectory().name
            os.makedirs(temp_dir, exist_ok=True)
            onnx_orig_path = os.path.join(temp_dir, "model.onnx")
            onnx_inferred_path = os.path.join(temp_dir, "inferred.onnx")
            onnx.save_model(
                onnx_graph,
                onnx_orig_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                convert_attribute=False,
            )
            onnx.shape_inference.infer_shapes_path(onnx_orig_path, onnx_inferred_path)
            onnx_graph = onnx.load(onnx_inferred_path)
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def clip_add_hidden_states(self, return_onnx=False):
        hidden_layers = -1
        onnx_graph = gs.export_onnx(self.graph)
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                name = onnx_graph.graph.node[i].output[j]
                if "layers" in name:
                    hidden_layers = max(
                        int(name.split(".")[1].split("/")[0]), hidden_layers
                    )
        for i in range(len(onnx_graph.graph.node)):
            for j in range(len(onnx_graph.graph.node[i].output)):
                if onnx_graph.graph.node[i].output[
                    j
                ] == "/text_model/encoder/layers.{}/Add_1_output_0".format(
                    hidden_layers - 1
                ):
                    onnx_graph.graph.node[i].output[j] = "hidden_states"
            for j in range(len(onnx_graph.graph.node[i].input)):
                if onnx_graph.graph.node[i].input[
                    j
                ] == "/text_model/encoder/layers.{}/Add_1_output_0".format(
                    hidden_layers - 1
                ):
                    onnx_graph.graph.node[i].input[j] = "hidden_states"
        if return_onnx:
            return onnx_graph

    def fuse_mha_qkv_int8_sq(self):
        tensors = self.graph.tensors()
        keys = tensors.keys()

        # mha  : fuse QKV QDQ nodes
        # mhca : fuse KV QDQ nodes
        q_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_q/input_quantizer/DequantizeLinear_output_0"
        )
        k_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_k/input_quantizer/DequantizeLinear_output_0"
        )
        v_pat = (
            "/down_blocks.\\d+/attentions.\\d+/transformer_blocks"
            ".\\d+/attn\\d+/to_v/input_quantizer/DequantizeLinear_output_0"
        )

        qs = list(
            sorted(
                map(
                    lambda x: x.group(0),  # type: ignore
                    filter(
                        lambda x: x is not None, [re.match(q_pat, key) for key in keys]
                    ),
                )
            )
        )
        ks = list(
            sorted(
                map(
                    lambda x: x.group(0),  # type: ignore
                    filter(
                        lambda x: x is not None, [re.match(k_pat, key) for key in keys]
                    ),
                )
            )
        )
        vs = list(
            sorted(
                map(
                    lambda x: x.group(0),  # type: ignore
                    filter(
                        lambda x: x is not None, [re.match(v_pat, key) for key in keys]
                    ),
                )
            )
        )

        removed = 0
        assert len(qs) == len(ks) == len(vs), "Failed to collect tensors"
        for q, k, v in zip(qs, ks, vs):
            is_mha = all(["attn1" in tensor for tensor in [q, k, v]])
            is_mhca = all(["attn2" in tensor for tensor in [q, k, v]])
            assert (is_mha or is_mhca) and (not (is_mha and is_mhca))

            if is_mha:
                tensors[k].outputs[0].inputs[0] = tensors[q]
                tensors[v].outputs[0].inputs[0] = tensors[q]
                del tensors[k]
                del tensors[v]
                removed += 2
            else:  # is_mhca
                tensors[k].outputs[0].inputs[0] = tensors[v]
                del tensors[k]
                removed += 1
        print(f"Removed {removed} QDQ nodes")
        return removed


def optimize(name, onnx_graph, return_onnx=True, **kwargs):
    opt = Optimizer(onnx_graph, verbose=True)
    opt.info(name + ": original")
    opt.cleanup()
    opt.info(name + ": cleanup")
    opt.fold_constants()
    opt.info(name + ": fold constants")
    opt.infer_shapes()
    opt.info(name + ": shape inference")
    if kwargs.get("fuse_mha_qkv_int8", False):
        opt.fuse_mha_qkv_int8_sq()
        opt.info(name + ": fuse QKV nodes")
    onnx_opt_graph = opt.cleanup(return_onnx=return_onnx)
    opt.info(name + ": finished")
    return onnx_opt_graph
