# -*- coding: utf-8 -*-
import onnx
from onnx import shape_inference
import netron
import argparse
import json

# for operators, see https://github.com/onnx/onnx/blob/master/docs/Operators.md
# for proto, see https://github.com/onnx/onnx/blob/master/onnx/onnx.proto


def get_tensor_dim(edge, graph):
    dims, values = [], []
    in_values = False

    if edge == 'output':
        values = graph.output
    else:
        values = graph.value_info

    for v in values:
        if v.name == edge:
            for i in v.type.tensor_type.shape.dim:
                dims.append(i.dim_value)
                in_values = True
            break
    if not in_values:
        for i in graph.initializer:
            if i.name == edge:
                for d in i.dims:
                    dims.append(d)
                break

    return dims


def get_metric_for_op(op, pathslist, test_set, block_corpus):
    MAX_SPC = 1000

    metrics = [1]  # otc
    metrics.append(len(test_set[op]['input']) / len(block_corpus[op]['in_degree']))  # idc
    metrics.append(len(test_set[op]['output']))  # odc
    metrics.append(len(test_set[op]['edgetype']) / len(block_corpus))  # sec
    metrics.append((len(test_set[op]['attrs'])+len(test_set[op]['dims']))) # / MAX_SPC)  # spc
    metrics.append(test_set[op]['cnt'])
    return metrics


def get_all_paths(start, edges, graph, inputs):

    output, path, curpaths = [], [], []
    if start == -1:
        path.append(('output', 'Output'))
        curpaths.append(path)
        return curpaths
    elif start < 0:
        name = inputs[-(start+2)]
        path.append((name, 'Input'))
        output = [name]
    else:
        path.append((graph.node[start].name, graph.node[start].op_type))
        output = graph.node[start].output

    for edgename in output:
        for edge in edges[edgename]:
            newpaths = get_all_paths(edge[1], edges, graph, inputs)
            for newpath in newpaths:
                curpaths.append(path+newpath)

    return curpaths



def get_edges_with_dims(model):
    inputs = []
    for init in model.graph.initializer:
        inputs.append(init.name)
    edges = {'output':[]}
    for name in inputs:
        edges[name] = []
    dims = {}

    for i in range(len(model.graph.node)):
        end_node = model.graph.node[i]
        for edge in end_node.input:
            if edge not in dims and 'concat' not in end_node.name and 'flatten' not in end_node.name:
                dims[edge] = get_tensor_dim(edge, model.graph)
            if edge in inputs:
                edges[edge].append((-inputs.index(edge)-2, i))
                continue

            for j in range(len(model.graph.node)):
                start_node = model.graph.node[j]
                if edge in start_node.output:
                    if edge in edges:
                        edges[edge].append((j, i))
                    else:
                        edges[edge] = [(j, i)]

        if 'output' in end_node.output:
            edges['output'].append((i, -1))


    return edges, dims, inputs


def parse_graph(model_no, model, test_set, block_corpus):
    INFINITE_INPUT = 10000
    MAX_OUTPUT = 10000

    inferred_model = shape_inference.infer_shapes(model)
    edges, dims, inputs = get_edges_with_dims(inferred_model)
    paths = []
    for i in range(len(inputs)):
        paths += get_all_paths(-i-2, edges, model.graph, inputs)


    for node in model.graph.node:
        if 'flatten' in node.name or 'concat' in node.name:
            continue
        if node.op_type in block_corpus:
            if node.op_type not in test_set:
                test_set[node.op_type] = {'input':[], 'output':[], 'edgetype':[], 'attrs':[], 'dims':[], 'path':{}, 'cnt':0}
            test_set[node.op_type]['cnt'] += 1
            for output in node.output:
                if output == 'output':
                    continue
                for edge in edges[output]:
                    if inferred_model.graph.node[edge[1]].op_type not in test_set[node.op_type]['edgetype'] and 'flatten' not in inferred_model.graph.node[edge[1]].name and 'concat' not in inferred_model.graph.node[edge[1]].name:
                        test_set[node.op_type]['edgetype'].append(inferred_model.graph.node[edge[1]].op_type)
            if 'concat' in node.name:
                continue
            if node.attribute and node.attribute not in test_set[node.op_type]['attrs']:
                test_set[node.op_type]['attrs'].append(node.attribute)
            for input in node.input:
                if dims[input] not in test_set[node.op_type]['dims']:
                    test_set[node.op_type]['dims'].append(dims[input])


        suffix = []
        for i in node.output:
            if i == 'output':
                continue
            for j in range(len(edges[i]) - 1):
                suffix.append(i)
        node.output.extend(suffix)


    for node in model.graph.node:
        if 'flatten' in node.name or 'concat' in node.name:
            continue
        if node.op_type in block_corpus:
            if 'concat' not in node.name and len(node.input) not in test_set[node.op_type]['input']:
                test_set[node.op_type]['input'].append(len(node.input))
            if len(node.output) not in test_set[node.op_type]['output']:
                test_set[node.op_type]['output'].append(len(node.output))

    return paths, edges, test_set, dims

def get_graph_metric(model, edges):
    cnt = 0
    ops = []
    pairs = []
    for i in range(len(model.graph.node)):
        fromnode = model.graph.node[i]
        if 'flatten' in fromnode.name or 'concat' in fromnode.name:
            continue
        cnt += 1
        if fromnode.op_type not in ops:
            ops.append(fromnode.op_type)
        for tensors in edges.values():
            for edge in tensors:
                if edge[0] == i and edge[1] >=0:
                    tonode = model.graph.node[edge[1]]
                    if 'flatten' not in tonode.name and 'concat' not in tonode.name:
                        pair = (fromnode.op_type, tonode.op_type)
                        if pair not in pairs:
                            pairs.append(pair)
    return cnt, ops, pairs
        

def get_coverage(modellist, test_set, g_metrics):
    pathslist = []
    with open("./bc.json", 'r') as load_f:
        block_corpus = json.load(load_f)
    w = [1, 1, 1, 1, 1, 1, 1]
    w_op = [1, 1, 1, 1, 1, 1, 1]
    w_g = [1, 1, 1, 1, 1, 1, 1]


    for i in range(len(modellist)):
        model = modellist[i]
        model.graph.output[0].name = 'output'
        model.graph.node[-1].output[0] = 'output'

        paths, edges, test_set, dims = parse_graph(i, model, test_set, block_corpus)
        opcnt, optypes, opedges = get_graph_metric(model, edges)
        g_opcnt = opcnt
        g_type = len(optypes)  
        g_edge = len(opedges)
        def get_p_and_s(dims):
            dd = list(dims.values())
            ddd = []
            for x in dd:
                if x not in ddd:
                    ddd.append(x)
            return ddd

        g_s_and_p = len(get_p_and_s(dims))
        g_type1 = len(optypes)/opcnt
        g_edge1 = len(opedges)/(len(optypes)*len(optypes))
        g_edge2 = len(opedges)/(len(block_corpus)*len(block_corpus))
        g_s_and_p1 = g_s_and_p/opcnt
        g_s_and_p2 = g_s_and_p/len(optypes)
        def parse_path(paths):
            maxlen, avglen = 0, 0
            for path in paths:
                maxlen = max(len(path)-4, maxlen)
                avglen += (len(path)-4)
            return len(paths), maxlen, avglen/len(paths)
        g_pnum, g_maxlen, g_avglen = parse_path(paths)
        g_metrics.append((g_opcnt, g_type, g_edge, g_s_and_p, g_pnum, g_maxlen, g_avglen, g_type1, g_edge1, g_edge2, g_s_and_p1, g_s_and_p2))

        pathslist.append(paths)

    total = [0., 0., 0., 0., 0.]
    metricOP = {}
    for key in block_corpus:
        if key in test_set:
            metrics = get_metric_for_op(key, pathslist, test_set, block_corpus)
            total = [total[i]+metrics[i] for i in range(len(metrics)-1)]
            metricOP[key] = metrics
        else:
            metricOP[key] = [0, '-', '-', '-', '-', '0']


    metricI = [total[i]/len(block_corpus) for i in range(len(total))]
    return metricOP, metricI, g_metrics, test_set