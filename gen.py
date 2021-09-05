import numpy as np

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import onnxruntime as rt

# from tvm import relay
# from tvm.relay import testing
# import tvm
# from tvm import te
# from tvm.contrib import graph_runtime

import os
import collections
import time
import argparse

# np.random.seed(1)

error_num = 0
error_config = None
err_message_set = set()

DEBUG = True

NO_INPUT = True

TEST_GPU = False

global_tensor_num = 0

parser = argparse.ArgumentParser()
parser.add_argument('--minnode', type=int,  help="MIN_NODE", default=1)
parser.add_argument('--maxnode', type=int,  help="MAX_NODE", default=25)
parser.add_argument('--pickrate', type=float,  help="pickExistRate", default=0.95)
parser.add_argument('--file', type=str, help='res_file', default='./res_dice_time.txt')
args = parser.parse_args()

ITER_NUM = 10000
MIN_NODE = args.minnode
MAX_NODE = args.maxnode
MIN_TENSOR_DIM = 1
MAX_TENSOR_DIM = 5
MAX_TENSOR_DIM_LEN = 5
MAX_MULTI_INPUTS = 6
np.random.seed(hash(args.file) % 10000)
totOp = 0
totSize = 0
ok = 0
okUpper = 100000000000
pickExistRate = args.pickrate

graph_num = 0

import criterion

def work():
	tensor_list = []
	tensor_map = {}
	tensor_init = {}
	tensor_init_type = {}
	node_list = []
	input_tensor = []
	inputs_feed = {}
	output_tensor = []
	init_tensor = []

	id_ops = ['Identity', 'Abs', 'Neg', 'Reciprocal', 'Floor', 'Ceil', 'Softsign',\
			  'Sigmoid', 'HardSigmoid', 'Relu', 'LeakyRelu', \
			  'Sin', 'Cos', 'Softmax', 'MaxPool', 'AveragePool', 'LpPool',\
			  'SpaceToDepth', 'Erf', 'Sign', \
			  'Flatten'
			  ]
	id_ops += ['Round']

	id_ops.append('Exp')
	id_ops.append('Selu')
	id_ops.append('Sqrt')
	extra_t_ops = ['MatMul', 'Add', 'Sub', 'Mul', 'Div', 'Concat']
	extra_t_ops += ['PRelu']

	extra_t_ops += ['Gemm']

	extra_t_ops += ['Conv']

	multi_extra_t_ops = ['Sum', 'Max', 'Min', 'Mean']
	extra_t_ops += multi_extra_t_ops

	reduce_ops = ["ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSumSquare", \
	              "ReduceL1", "ReduceL2", "ReduceLogSumExp"]

	multi_out_ops = []

	other_ops = []

	ops = []
	ops += id_ops
	ops += extra_t_ops
	ops += reduce_ops
	ops += other_ops
	ops += multi_out_ops

	network_depth = np.random.randint(MIN_NODE, MAX_NODE + 1)


	ops_seq = None
	


	global global_tensor_num
	global_tensor_num = 0

	def rand_shape():
		fixed_dim = np.random.randint(MIN_TENSOR_DIM, MAX_TENSOR_DIM + 1)
		fixed_shape = [np.random.randint(1, MAX_TENSOR_DIM_LEN + 1) for i in range(fixed_dim)]
		return fixed_shape

	def newOrReuse(shape, withF=None):
		if np.random.randint(0, 100000) < 100000 * pickExistRate:
			if withF is None:
				t = match(shape)
			else:
				t = matchWithF(withF)
			if t is not None:
				return t
		if withF is not None:
			lc = list(filter(withF, [i for i in range(MIN_TENSOR_DIM, MAX_TENSOR_DIM + 1)]))
			if len(lc) == 0:
				raise Exception("can not generate shape for this operation %s" % new_node_type)
			l = lc[np.random.randint(0, len(lc))]
			shape = [-1 for i in range(l)]

		for i in range(len(shape)):
			if shape[i] == -1:
				shape[i] = np.random.randint(1, MAX_TENSOR_DIM_LEN + 1)
		ret = new_tensor(shape)
		pass_value(ret)
		return ret

	def new_tensor(shape, data_type=TensorProto.FLOAT, data_value=None):
		global global_tensor_num
		global_tensor_num += 1
		cur_name = 'node' + str(global_tensor_num)
		cur_tensor = helper.make_tensor_value_info(cur_name, data_type, shape)
		if (data_value is None) and (-1 not in shape):
			if data_type == TensorProto.FLOAT:
				cur_value = np.random.random(tuple(shape))
				cur_value = cur_value * 2 - 1
				cur_value = cur_value.astype(np.float32)
		else:
			cur_value = data_value
		tensor_list.append(cur_name)
		tensor_map[cur_name] = cur_tensor
		tensor_init[cur_name] = cur_value
		tensor_init_type[cur_name] = data_type
		return cur_name

	def tensor_shape(t_name):
		return list(tensor_init[t_name].shape)

	def pass_value(t):
		t_value = tensor_init[t]
		t_type = tensor_init_type[t]
		t = tensor_map[t]

		if NO_INPUT:
			init_tensor.append(helper.make_tensor(t.name, t_type, dims=t_value.shape, vals=t_value.flatten()))
		else:
			input_tensor.append(t)
			inputs_feed[t.name] = t_value

	lastn = new_tensor(rand_shape())
	pass_value(lastn)

	dq = []
	dq.append(lastn)

	no_succ = set()
	no_succ.add(lastn)

	def match(partial_shape):
		dq_l = len(dq)
		x_ord = np.arange(dq_l)
		np.random.shuffle(x_ord)
		for x_index in x_ord:
			x = dq[x_index]
			x_shape = tensor_shape(x)
			x_shape_l = len(x_shape)
			if x_shape_l == len(partial_shape):
				matched = True
				for i in range(x_shape_l):
					if (partial_shape[i] > 0) and (partial_shape[i] != x_shape[i]):
						matched = False
						break
				if matched:
					return x
		return None

	def matchWithF(f):
		dq_l = len(dq)
		x_ord = np.arange(dq_l)
		np.random.shuffle(x_ord)
		for x_index in x_ord:
			x = dq[x_index]
			x_shape = tensor_shape(x)
			x_shape_l = len(x_shape)
			if f(x_shape_l):
				return x
		return None


	for step in range(network_depth):
		v = np.random.randint(0, len(ops))
		new_node_type = ops[v]
		if ops_seq != None:
			new_node_type = ops_seq[step % len(ops_seq)]
		node_name = 'op' + str(step)
		kwargs = {}

		filter_f = lambda x: True
		if new_node_type == 'SpaceToDepth':
			filter_f = lambda x: x == 4
		if new_node_type in ['MaxPool', 'AveragePool', 'LpPool']:
			filter_f = lambda x: (x >= 3) and (x <= 5) # unsupprt on ONNXRuntime/TVM
		if new_node_type == 'Conv':
			filter_f = lambda x: x == 4
		if new_node_type in ['MatMul', 'Gemm']:
			filter_f = lambda x: x >= 2
		if new_node_type == 'Gemm':
			filter_f = lambda x: x == 2

		n1 = newOrReuse([], filter_f)

		n1_shape = tensor_shape(n1)
		n1_dim = len(n1_shape)				

		n3_shape = tensor_shape(n1)
		n4_shape = tensor_shape(n1)

		if new_node_type in ['Softmax', 'LpNormalization', 'Concat', 'Compress', 'Flatten']:
			kwargs['axis'] = np.random.randint(0, len(tensor_shape(n1)))
		if new_node_type in reduce_ops:
			kwargs['axes'] = [np.random.randint(0, len(tensor_shape(n1)))]

		if new_node_type == 'SpaceToDepth':
			def gcd(a, b):
				return (a if b == 0 else gcd(b, a % b)) 
			t = gcd(n1_shape[2], n1_shape[3])
			# TODO(generate factor of t)
			kwargs['blocksize'] = t
			n3_shape[1] = n1_shape[1] * t * t
			n3_shape[2] = n3_shape[2] // t
			n3_shape[3] = n3_shape[3] // t

		if new_node_type in ['MaxPool', 'AveragePool', 'LpPool']:
			kwargs['kernel_shape'] = [np.random.randint(1, n1_shape[i + 2] + 1) for i in range(n1_dim - 2)]
			kwargs['strides'] = [np.random.randint(1, n1_shape[i + 2] + 1) for i in range(n1_dim - 2)]
			for i in range(n1_dim - 2):
				n3_shape[i + 2] = (n3_shape[i + 2] - kwargs['kernel_shape'][i]) // kwargs['strides'][i] + 1
		if new_node_type == 'Conv':
			kwargs['kernel_shape'] = [np.random.randint(1, n1_shape[i + 2] + 1) for i in range(n1_dim - 2)]	

			kwargs['strides'] = [1 for i in range(n1_dim - 2)]
			

			kwargs['pads'] = [0 for i in range((n1_dim - 2) * 2)]

			kwargs['group'] = 1
			# kwargs['auto_pad'] = 'SAME_LOWER'
			out_dim = int(1e9)
			for i in range(n1_dim - 2):
				n3_shape[i + 2] = (n1_shape[i + 2] - kwargs['kernel_shape'][i] + kwargs['pads'][i * 2] + kwargs['pads'][i * 2 + 1]) // kwargs['strides'][i] + 1

			

		if new_node_type in ['LpPool', 'LpNormalization']:
			kwargs['p'] = np.random.randint(1, 3) * 2
		if new_node_type in ['LeakyRelu']:
			kwargs['alpha'] = np.random.randint(1, 3) * 0.01

		if new_node_type == 'Compress':
			l = n1_shape[kwargs['axis']]
			n2_dim = 1
			n2_shape = [l]
			while True:
				compress_tensor = [np.random.randint(0, 2) for i in range(l)]
				compress_sum = int(sum(compress_tensor))
				if compress_sum == 0:
					continue
				else:
					break
			n2 = new_tensor(n2_shape, TensorProto.BOOL, np.array(compress_tensor))
			pass_value(n2)
			n3_shape[kwargs['axis']] = compress_sum
			inputs = [n1, n2]
		elif new_node_type in extra_t_ops:
			n2_shape = tensor_shape(n1)
			n2_dim = len(n2_shape)

			if new_node_type == 'Conv':
				# TODO(change n2_shape[0, 1])
				n2_shape[0] = n1_shape[0]

				n2_shape[1] = n1_shape[1] // kwargs['group']
				for i in range(n1_dim - 2):
					n2_shape[2 + i] = kwargs['kernel_shape'][i]
				n3_shape[1] = n2_shape[0]


			if new_node_type in ['MatMul', 'Gemm']:
				n2_shape[n2_dim - 2] = n2_shape[n2_dim - 1]
				n2_shape[n2_dim - 1] = -1
				

			if new_node_type == 'Concat':
				n2_shape[kwargs['axis']] = np.random.randint(1, MAX_TENSOR_DIM_LEN + 1)
				n3_shape[kwargs['axis']] = n1_shape[kwargs['axis']] + n2_shape[kwargs['axis']]

			n2 = newOrReuse(n2_shape)
			inputs = [n1, n2]

			if new_node_type == 'Gemm':
				n2_2 = newOrReuse(n3_shape)
				if np.random.randint(0, 2) == 0:
					inputs.append(n2_2)
		else:
			if new_node_type in reduce_ops:
				n3_shape[kwargs['axes'][0]] = 1
			if new_node_type == 'Flatten':
				# kwargs['axis'] = 0
				d = kwargs['axis']
				p1 = 1
				p2 = 1
				n3_shape = []
				for i in range(n1_dim):
					if i < d:
						p1 = p1 * n1_shape[i]
					else:
						p2 = p2 * n1_shape[i]
				n3_shape = [p1, p2]
			inputs = [n1]

		if new_node_type == 'Split':
			kwargs['axis'] = np.random.randint(0, n1_dim)
			split_sum = n1_shape[kwargs['axis']]

			nums_output = 2
			if nums_output == 1:
				n2_shape = [1]
				split_t = [split_sum]
				n2 = new_tensor(n2_shape, TensorProto.INT64, np.array(split_t))
				pass_value(n2)
			else:
				split_mid = np.random.randint(0, split_sum) + 1
				n2_shape = [2]
				split_t = [split_mid, split_sum - split_mid]
				n2 = new_tensor(n2_shape, TensorProto.INT64, np.array(split_t))
				pass_value(n2)
				n3_shape[kwargs['axis']] = split_mid
				n4_shape[kwargs['axis']] = split_sum - split_mid
			kwargs['split'] = split_t
			inputs = [n1, n2]

		if new_node_type in multi_extra_t_ops:
			extra_num = np.random.randint(0, MAX_MULTI_INPUTS - 1)
			for t in range(extra_num):
				n_another = newOrReuse(tensor_shape(n1))
				inputs.append(n_another)
		if new_node_type == "Concat":
			extra_num = np.random.randint(0, MAX_MULTI_INPUTS - 1)
			for t in range(extra_num):
				n_another_shape = tensor_shape(n1)
				n_another_shape[kwargs['axis']] = np.random.randint(1, MAX_TENSOR_DIM_LEN + 1)
				n_another = newOrReuse(n_another_shape)
				n3_shape[kwargs['axis']] += n_another_shape[kwargs['axis']]
				inputs.append(n_another)


		if new_node_type in ['MatMul', 'Gemm']:
			n3_shape[n2_dim - 1] = tensor_shape(n2)[n2_dim - 1]

		n3 = new_tensor(n3_shape)
		outputs = [n3]

		if new_node_type in multi_out_ops:
			if n2_shape[0] == 2:
				n4 = new_tensor(n4_shape)
				outputs = [n3, n4]

		new_node = helper.make_node(new_node_type, inputs=inputs, outputs=outputs, name=node_name, **kwargs)
		node_list.append(new_node)

		for t in inputs:
			if t in no_succ:
				no_succ.remove(t)

		lastn = n3
		for x in outputs:
			dq.append(x)
			no_succ.add(x)

	def totElement(x):
		ans = 1
		for i in tensor_shape(x):
			ans = ans * i
		return ans

	output_ts = []
	tot_output_element = 0
	for x in no_succ:
		x2 = new_tensor([1, totElement(x)])
		output_ts.append(x2)
		n = helper.make_node('Flatten', inputs=[x], outputs=[x2], name='flatten_%s' % x, axis=0)
		node_list.append(n)
		tot_output_element += totElement(x)


	# print(output_ts)
	final_tensor = new_tensor([1, tot_output_element])	
	n = helper.make_node('Concat', inputs=output_ts, outputs=[final_tensor], name='concat_outputs', axis=-1)
	node_list.append(n)
	output_tensor.append(tensor_map[final_tensor])

	graph_def = helper.make_graph(node_list, "test-model", input_tensor, output_tensor, init_tensor)
	model = helper.make_model(graph_def, producer_name='onnx-example')

	return model, inputs_feed

def test(model_data):
	model, inputs_feed = model_data
	# print('The graph in model:\n{}'.format(model.graph))

	filename = "tmp.onnx"
	onnx.save(model, filename)

	global graph_num
	graph_num += 1
	# onnx.save(model, "tmp/g"+str(graph_num)+".onnx")
	

	np.save("inputs.npy", inputs_feed)
	sess = rt.InferenceSession(filename)
	output_name = sess.get_outputs()[0].name
	out = sess.run([output_name], inputs_feed)

	# onnx_model = model
	# inputs = inputs_feed
	#
	# shape_dict = {}
	# for k, v in inputs.items():
	# 	shape_dict[k] = v.shape
	#
	# mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
	#
	# # print(mod.astext(show_meta_data=False))
	#
	# opt_level = 3
	# # opt_level = np.random.randint(0, 4)
	# # opt_level = np.random.randint(2, 4)
	# # print("opt_level=%d" % opt_level)
	#
	# target = "llvm"
	#
	# if TEST_GPU:
	# 	target = "cuda"
	#
	# global error_config
	# error_config = [opt_level, target]
	#
	# with relay.build_config(opt_level=opt_level):
	# 	tvm_graph, tvm_lib, tvm_params = relay.build_module.build(mod, target, params=params)
	#
	# #print(tvm_graph)
	# #print(tvm_lib)
	# #print(tvm_params)
	#
	# ctx = tvm.cpu(0)
	# if TEST_GPU:
	# 	ctx = tvm.gpu()
	#
	# module = graph_runtime.create(tvm_graph, tvm_lib, ctx)
	# module.load_params(relay.save_param_dict(tvm_params))
	#
	# for k, v in inputs.items():
	# 	module.set_input(k, v)
	#
	# module.run()
	# out_deploy = module.get_output(0).asnumpy()
	#
	# fix_dec = 0
	# out = np.around(out, decimals=fix_dec)
	# out_deploy = np.around(out_deploy, decimals=fix_dec)
	#
	# #print(out)
	# #print(out_deploy)
	#
	# global totOp
	# global totSize
	# global ok
	# totOp+=len(set([x.op_type for x in model.graph.node]))
	# print([x.name for x in model.graph.node])
	# totSize+=len(list(model.graph.node))
	# ok += 1
	# print(ok)
	# print(totOp, totSize)
	# print(totOp/ok, totSize/ok)
	# if (ok == okUpper):
	# 	exit(0)
	#
	# if np.isnan(out).any() or np.isnan(out_deploy).any():
	# 	return
	#
	# # assert((out == out_deploy).all())
	# res = (out == out_deploy)
	# # assert(np.sum(res==False) < 10)
	# res = np.array(res)
	# # assert(np.sum(res==True) >= res.size * 0.9)




test_set = {}
g_metrics = []
f_name = './results/' + args.file[:-4] + "_" + str(args.minnode) + "_" + str(args.maxnode) + "_" + str(args.pickrate) + ".txt"
if os.path.exists(f_name):
	os.remove(f_name)

BATCHSIZE = 1000
BATCHNUM = ITER_NUM // BATCHSIZE
w_gI = [1, 1, 1, 1, 1, 1, 1]
start_time = time.perf_counter()
	


errs = []

err_num = 0
iter = 0
while True:
	model, _ = work()
	ret_op, ret_I, g_metrics, test_set = criterion.get_coverage([model], test_set, g_metrics)

	f = open(f_name, 'a')
	cur_time = time.perf_counter()
	f.write('at time ' + str(cur_time - start_time)+' \n')
	f.write('generate '+ str(iter+1) + ' valid models\n')
	f.write('{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n'.format('Object', 'OTC', 'IDC', 'ODC', 'SEC', 'SPC'))
	# for key in ret_op:
	# 	if ret_op[key][0] == 0:
	# 		f.write('{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n'.format(key, '-', '-', '-', '-', '-'))
	# 	else:
	# 		f.write('{:<10}{:<10.2%}{:<10.2%}{:<10.2%}{:<10.2%}{:<10.2%}\n'.format(key, ret_op[key][0], ret_op[key][1], ret_op[key][2], ret_op[key][3], ret_op[key][4]))

	f.write('{:<10}{:<10.2%}{:<10.2%}{:<10.4f}{:<10.2%}{:<10.4f}\n'.format('I', ret_I[0], ret_I[1], ret_I[2], ret_I[3], ret_I[4]))
	op_names, op_nums = '', ''
	for key in ret_op:
		op_names += (key+' ')
		op_nums += (str(ret_op[key][5]) + ' ')
	f.write(op_names + '\n')
	f.write(op_nums + '\n')
	f.write('{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}\n'.format('Object', 'NOO', 'NOT', 'NOP', 'NSP',' NTP', 'MLP', 'ALP'))
	f.write('{:<10}{:<10}{:<10.4f}{:<10.4f}{:<10}{:<10}{:<10}{:<10.4f}\n'.format('g', g_metrics[iter][0], g_metrics[iter][1], g_metrics[iter][2], g_metrics[iter][3], g_metrics[iter][4], g_metrics[iter][5], g_metrics[iter][6]))
	g_m = [sum(g[i] for g in g_metrics)/len(g_metrics) for i in range(7)]

	f.write('{:<10}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}\n'.format('I', g_m[0], g_m[1], g_m[2], g_m[3], g_m[4], g_m[5], g_m[6]))
	f.close()
	iter += 1

print('This is gen.py!\n')


"""
for iter in range(ITER_NUM):	
	try:
		work()
		if DEBUG and (iter % 100 == 0):
			print("iter=", iter)
			print("OK!")
	except Exception as err:
		err_m = str(err)

		if err_m in err_message_set:
			print('same error message!')
			continue
		err_message_set.add(err_m)

		error_num += 1
		print("Find Bugs! Number %d" % error_num)
		print("............\n............\n............\n............\n")
		print("error=", err_m)
		model = onnx.load("tmp.onnx")
		onnx.save(model, "output/bug%d.onnx" % error_num)
		inputs_feed = np.load("inputs.npy", allow_pickle=True)
		np.save("output/bug%d_inputs.npy" % error_num, inputs_feed)
		with open("output/bug%d_log.txt" % error_num, "w") as f:
			print("tvm_params =", error_config, file=f)
			print("error_log = \n", err, file=f)

		if DEBUG:
			break
"""
