from copy import deepcopy
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import polygraphy.backend.onnx.loader

def change_inout_type(graph):
    inputs = graph.inputs
    for input in inputs:
        if input.dtype == "float32":
            input.dtype = "float16"

    outputs = graph.outputs
    for output in outputs:
        if output.dtype == "float32":
            output.dtype = "float16"

def only_change_inout_type(src_onnx, dst_onnx):
    graph = gs.import_onnx(onnx.load(src_onnx))
    change_inout_type(graph)
    graph.cleanup()
    onnx.save(gs.export_onnx(graph), dst_onnx)

def add_groupnorm(graph):
    cnt = 0
    for node in graph.nodes:
        if node.op == "Reshape" and node.o().op == "InstanceNormalization" and node.o().o().op == "Reshape" \
                and node.o().o().o().op == "Mul" and node.o().o().o().o().op == "Add":

            last_node = node.o().o().o().o()

            instance_norm = node.o()
            instance_norm_scale = instance_norm.inputs[1]
            instance_norm_bias = instance_norm.inputs[2]
            epsilon = instance_norm.attrs["epsilon"]
            mul_node = node.o().o().o()
            add_node = node.o().o().o().o()

            scale = np.ascontiguousarray(np.array(deepcopy(instance_norm_scale.values.tolist()), dtype=np.float32))
            bias = np.ascontiguousarray(np.array(deepcopy(instance_norm_bias.values.tolist()), dtype=np.float32))
            gamma = np.ascontiguousarray(np.array(deepcopy(mul_node.inputs[1].values.tolist()), dtype=np.float32))
            beta = np.ascontiguousarray(np.array(deepcopy(add_node.inputs[1].values.tolist()), dtype=np.float32))

            with_swish = True if len(add_node.o().o().inputs) == 2 and "Sigmoid" in add_node.o().o().inputs[1].name and node.o().o().o().o().o().o().op == "Mul" else False
            if with_swish:
                last_node = node.o().o().o().o().o().o()


            print(with_swish)
            
            constant_gamma = gs.Constant("gamma_{}".format(cnt), gamma.reshape(-1))
            constant_beta = gs.Constant("beta_{}".format(cnt), beta.reshape(-1))
            x = node.inputs[0]
            group_norm_v = gs.Variable("group_norm_{}".format(cnt), np.dtype(np.float32), x.shape)
            group_norm = gs.Node("GroupNorm", "GroupNorm_{}".format(cnt),
                                attrs={"epsilon": epsilon, "bSwish": with_swish},
                                inputs=[x, constant_gamma, constant_beta],
                                outputs=[group_norm_v])
            cnt += 1
            for n in graph.nodes:
                if last_node.outputs[0] in n.inputs:
                    index = n.inputs.index(last_node.outputs[0])
                    n.inputs[index] = group_norm.outputs[0]
            last_node.outputs = []
            graph.nodes.append(group_norm)

    print("find groupnorm: ", cnt)

def opt_graph_with_gn_silu(src_onnx, dst_onnx):
    onnx_graph = polygraphy.backend.onnx.loader.fold_constants(onnx.load(src_onnx), allow_onnxruntime_shape_inference=True)
    graph = gs.import_onnx(onnx_graph)

    change_inout_type(graph)
    add_groupnorm(graph)
    graph.cleanup()

    onnx.save(gs.export_onnx(graph), 
              dst_onnx,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location="weights.opt.pb",
              convert_attribute=False)

# src_onnx = "/workspace/models/onnx/unet/model.sim.onnx"
# dst_onnx = "/workspace/models/onnx_opt/unet/model.onnx"
# opt_unet(src_onnx, dst_onnx)
