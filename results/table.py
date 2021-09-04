import xlwt
from xlwt.Worksheet import Worksheet
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--interval', type=int,  help="time_interval", default=300)
parser.add_argument('--total', type=int,  help="total_time", default=1800)
parser.add_argument('--ispk', type=int, help='test pickrate or not', required=True)
parser.add_argument('--paths', type=str, help='path of file', required=True, nargs='*')
parser.add_argument('--types', type=str, help='types of files', required=True, nargs='*')
parser.add_argument('--case', type=str, help='setting of group', required=True, nargs='*')
parser.add_argument('--iter', type=int, help='iter of tests', default=5)
args = parser.parse_args()

if args.ispk:
    paths = [[args.paths[0]+str(i)+c+'.txt' for c in args.case] for i in range(1,args.iter+1)]
    type = [x.split('_')[-1] for x in args.case]
else:
    paths = [[p+str(i)+args.case[0]+'.txt' for p in args.paths] for i in range(1,args.iter+1)]
    type = args.types


time_interval = args.interval
total_time = args.total

I_name = ['OTC', 'IDC', 'ODC', 'SEC', 'SPC'] 
op_ms = [[[[] for j in range(len(type))]for i in range(5)] for k in range(1, args.iter+1)]
g_name =['NOO', 'NOT', 'NOP', 'NSP', 'NTP', 'MLP', 'ALP'] # , 'OTR1', 'OPR1', 'OPR2', 'NSP1', 'NSP2']
g_ms = [[[[] for j in range(len(type))] for i in range(7)] for k in range(1, args.iter+1)]
op_names = ['Identity', 'Abs', 'Neg', 'Reciprocal', 'Floor', 'Ceil', 'Round', 'Erf', 'Sign', 'Exp', 'Softsign', 'Softmax', \
     'Sigmoid', 'HardSigmoid', 'Relu', 'LeakyRelu', 'Selu', 'Sin', 'Cos', 'Sqrt', 'PRelu', 'Flatten', 'Add', 'Sub', 'Mul', \
         'Div', 'Sum', 'Max', 'Min', 'Mean', 'MaxPool', 'AveragePool', 'LpPool', 'Conv', 'MatMul', 'Gemm', 'Concat', 'SpaceToDepth', \
              'ReduceMax', 'ReduceMean', 'ReduceMin', 'ReduceProd', 'ReduceSumSquare', 'ReduceL1', 'ReduceL2', 'ReduceLogSumExp']
op_cnts = [[[] for i in range(len(type))] for k in range(1, args.iter+1)]
model_cnts = [0 for i in range(len(type))]

for iter, path in enumerate(paths):
    for t in range(len(path)):
        print(path[t])
        f = open(path[t], 'r')
        last_model = []
        times = []
        op_metrics = []
        graph_metrics = []

        lines = f.readlines()
        for i in range(len(lines)//10):
            tm = float(lines[i*10].split(' ')[2])
            times.append(tm)
            modelno = int(lines[i*10+1].split(' ')[1]) - 1
            I_metric = [x for x in lines[i*10+3].split(' ') if x != '']
            op_metrics.append([round(float(x[:-1])/100, 4) for x in I_metric[1:3]]+[float(I_metric[3])]+[round(float(I_metric[4][:-1])/100, 4)]+[float(I_metric[5])])
            g_metric = [x for x in lines[i*10+7].split(' ') if x != '']
            
            I_g_metric = [x for x in lines[i*10+8].split(' ') if x != '']
            graph_metrics.append([float(x) for x in I_g_metric[1:8]])

        last = len(lines)//10-1
        model_cnts[t] += (last+1)
        last_time = total_time
        while True:
            down_time = last_time - int(times[last])
            for j in range(down_time):
                last_model.append(last)
            last_time -= down_time
            while True:
                last -= 1
                if last < 0 or int(times[last]+1) <= last_time:
                    break
            if last == -1:
                for j in range(last_time+1):
                    last_model.append(-1)
                break

        last_model = last_model[::-1]
        last_model = last_model[::time_interval]
        
        for j in range(len(last_model)):   
            for m in range(len(op_ms[iter])):
                op_element = 0 if last_model[j] == -1 else op_metrics[last_model[j]][m]
                op_ms[iter][m][t].append(op_element)
            for m in range(len(g_ms[iter])):
                g_element = 0 if last_model[j] == -1 else graph_metrics[last_model[j]][m]
                g_ms[iter][m][t].append(g_element)

        op_nm = lines[last*10+4].split(' ')[:-1]
        op_c = lines[last*10+5].split(' ')[:-1]
        op_cnts[iter][t] = [int(op_c[op_nm.index(op_names[i])]) for i in range(len(op_names))]

op_ms = np.array(op_ms,dtype=np.float64)
g_ms = np.array(g_ms,dtype=np.float64)
op_cnts = np.array(op_cnts,dtype=np.float64)

op_ms = (op_ms[0] + op_ms[1] + op_ms[2] + op_ms[3] + op_ms[4])/5
g_ms = (g_ms[0] + g_ms[1] + g_ms[2] + g_ms[3] + g_ms[4])/5
op_cnts = (op_cnts[0] + op_cnts[1] + op_cnts[2] + op_cnts[3] + op_cnts[4])/5


workbook = xlwt.Workbook(encoding = 'ascii')
time_xs = [x for x in range(total_time+1)][::time_interval]
for i in range(len(I_name)):
    worksheet = workbook.add_sheet(I_name[i])
    worksheet.write(0,0, label = 'time')
    for t in range(len(type)):
        worksheet.write(0,1+t, label = type[t])
        for j, tm in enumerate(time_xs):
            worksheet.write(j+1,1+t, label = op_ms[i][t][j])
    for j, tm in enumerate(time_xs):
        worksheet.write(j+1,0, label = str(tm))    

for i in range(len(g_name)):
    worksheet = workbook.add_sheet(g_name[i])
    worksheet.write(0,0, label = 'time')
    for t in range(len(type)):
        worksheet.write(0,1+t, label = type[t])
        for j, tm in enumerate(time_xs):
            worksheet.write(j+1,1+t, label = g_ms[i][t][j])
    for j, tm in enumerate(time_xs):
        worksheet.write(j+1,0, label = str(tm))  

worksheet = workbook.add_sheet('opcnt')
for i in range(len(op_names)):
    worksheet.write(1+i, 0, label = op_names[i])
for t in range(len(type)):
    worksheet.write(0,1+t, label = type[t])
    worksheet.write(1+len(op_names),1+t,label=sum(op_cnts[t]))
    worksheet.write(1+len(op_names)+1,1+t,label=model_cnts[t]/args.iter)
    for i in range(len(op_names)):
        worksheet.write(1+i, 1+t, label = op_cnts[t][i] / sum(op_cnts[t]))
worksheet.write(1+len(op_names),0, label='total_op')
worksheet.write(1+len(op_names)+1,0,label='valid model nums')

if args.ispk:
    workbook.save('metric_pickrate.xls')
else:
    workbook.save('metric'+args.case[0]+'.xls')