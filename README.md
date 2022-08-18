# GenDice:  Efficient Test Generation for Deep Learning Compilers 

Source code and other supplemental materials  for "GenDice:  Efficient Test Generation for Deep Learning Compilers". 


## Requirements

- Python 3
- numpy==1.21.1
- onnx==1.8.0
- tvm
- onnxruntime==1.6.0
- netron==5.1.4
- xlwt==1.3.0

For the installation of **tvm**, please see [here](https://github.com/apache/tvm). 



## Usage

```
# for GenDice generation
timeout (time) python gen.py --minnode MINNODE --maxnode MAXNODE --pickrate pickExisRate --file (name of file that contains results) 
# eg: timeout 3600s python gen.py --minnode 1 --maxnode 30 --pickrate 0.95 --file result_dice_e{i}.txt"

# for baseline 1 purely random generation 
timeout (time) python gen.py --minnode MINNODE --maxnode MAXNODE --pickrate pickExisRate --file filename 

# for baseline 2 random generation with incremental generation
timeout (time) python gen.py --minnode MINNODE --maxnode MAXNODE --pickrate pickExisRate --file filename

# write the results into excels 
cd results
python table.py --interval time_interval --total total_time --ispk (1 for experiment with baselines; 0 for experiment of pickrate) --paths (names of files need to be processed) --types (different approaches) --case (different settings of parameters) --iter 5
#eg: 
# python table.py --interval 30 --total 900 --ispk 0 --paths result_dice_e result_rand_e result_winc_e --types dice purerand winc --case _10_10_0.95 --iter 5
# python table.py --interval 30 --total 3600 --ispk 1 --paths result_dice_e --types dice --case _1_30_0.5 _1_30_0.8 _1_30_0.9 _1_30_0.95 _1_30_0.98 _1_30_0.99 --iter 5

```



## Semantic Specification of Deep Learning Models

In deep learning, a computation graph is defined as a directed graph that comprises nodes and edges. Each node denotes a mathematical operation. Each edge describes a tensor that gets transferred between nodes. Computation graph is a suitable level of representation for deep learning model, considering that APIs are too high-level and hardware languages are too low-level.

Nodes on computation graph define mathematical operations on their related edges which are the input and output tensor(s) of the operation. Explicit forms of arithmetical expression of operations indicate the semantic specification of the operation (i.e., the constraints between the attributes of tensors).

In order to better model those constraints from operations, we categorize the operations into three types and describe their features respectively, as shown in the table below. Some of the operations may belong to more than one category. 

| Number | Operation Name  |               Type               |                          Parameters                          |
| :----: | :-------------: | :------------------------------: | :----------------------------------------------------------: |
|   1    |    Identity     |            *identity*            |                             None                             |
|   2    |       Abs       |            *identity*            |                             None                             |
|   3    |       Neg       |            *identity*            |                             None                             |
|   4    |   Reciprocal    |            *identity*            |                             None                             |
|   5    |      Floor      |            *identity*            |                             None                             |
|   6    |      Ceil       |            *identity*            |                             None                             |
|   7    |      Round      |            *identity*            |                             None                             |
|   8    |       Erf       |            *identity*            |                             None                             |
|   9    |      Sign       |            *identity*            |                             None                             |
|   10   |       Exp       |            *identity*            |                             None                             |
|   11   |    Softsign     |            *identity*            |                             None                             |
|   12   |     Softmax     |            *identity*            |                             axis                             |
|   13   |     Sigmoid     |            *identity*            |                             None                             |
|   14   |   HardSigmoid   |            *identity*            |                         alpha, beta                          |
|   15   |      Relu       |            *identity*            |                             None                             |
|   16   |    LeakyRelu    |            *identity*            |                            alpha                             |
|   17   |      Selu       |            *identity*            |                         alpha, gamma                         |
|   18   |       Sin       |            *identity*            |                             None                             |
|   19   |       Cos       |            *identity*            |                             None                             |
|   20   |      Sqrt       |            *identity*            |                             None                             |
|   21   |      PRelu      |          *multi-inputs*          |                             None                             |
|   22   |     Flatten     |          *multi-inputs*          |                             axis                             |
|   23   |       Add       |          *multi-inputs*          |                             None                             |
|   24   |       Sub       |          *multi-inputs*          |                             None                             |
|   25   |       Mul       |          *multi-inputs*          |                             None                             |
|   26   |       Div       |          *multi-inputs*          |                             None                             |
|   27   |       Sum       |          *multi-inputs*          |                             None                             |
|   28   |       Max       |          *multi-inputs*          |                             None                             |
|   29   |       Min       |          *multi-inputs*          |                             None                             |
|   30   |      Mean       |          *multi-inputs*          |                             None                             |
|   31   |     MaxPool     |          *multi-inputs*          | auto\_pad, ceil\_mode, dilations, kernel\_shape, pads, storage\_order, strides |
|   32   |   AveragePool   |          *multi-inputs*          | auto\_pad, ceil\_mode, count\_include\_pad, dilations, kernel\_shape, pads, strides |
|   33   |     LpPool      |          *multi-inputs*          |          auto\_pad, kernel\_shape, p, pads, strides          |
|   34   |      Conv       |          *multi-inputs*          |  auto\_pad, dilations, group, kernel\_shape, pads, strides   |
|   35   |     MatMul      |          *multi-inputs*          |                             None                             |
|   36   |      Gemm       |          *multi-inputs*          |                 alpha, beta, transA, transB                  |
|   37   |     Concat      | *multi-inputs*, *shape-changing* |                             axis                             |
|   38   |  SpaceToDepth   |         *shape-changing*         |                       blocksize, mode                        |
|   39   |    ReduceMax    |         *shape-changing*         |                        axes, keepdims                        |
|   40   |   ReduceMean    |         *shape-changing*         |                        axes, keepdims                        |
|   41   |    ReduceMin    |         *shape-changing*         |                        axes, keepdims                        |
|   42   |   ReduceProd    |         *shape-changing*         |                        axes, keepdims                        |
|   43   | ReduceSumSquare |         *shape-changing*         |                        axes, keepdims                        |
|   44   |    ReduceL1     |         *shape-changing*         |                        axes, keepdims                        |
|   45   |    ReduceL2     |         *shape-changing*         |                        axes, keepdims                        |
|   46   | ReduceLogSumExp |         *shape-changing*         |                        axes, keepdims                        |

The first type is called *identity* operation (e.g., `Absolute` operation, `Exponential` operation). It means that the shape of the input tensor is the same as the shape of the output tensor. For example, `Softmax` operation will take a single tensor and an integer parameter as the input. The integer parameter is named ![](http://latex.codecogs.com/svg.latex?axis) which defines the dimension that `Softmax` operation will be performed on. 

The second type is *multi-inputs* operation (e.g., `Add` operation, `Sub` operation). It means that the input of the operation requires multiple tensors. For example, `Add` operation takes two or more tensors as input and performs element-wise binary addition with them. We use ![](http://latex.codecogs.com/svg.latex?a_i) to denote the length of i-th dimension in tensor ![](http://latex.codecogs.com/svg.latex?A), and take two tensors as the input as the example:


![](http://latex.codecogs.com/svg.latex?C_{i_1,i_2,...i_n}=A_{i_1,i_2,...i_n}+B_{i_1,i_2,...i_n},\forall{i_1\in[1,a_1],...,i_n\in[1,a_n]})


The mathematic expression of addition naturally implies a semantic constraint between the inputs: the shape of tensor ![](http://latex.codecogs.com/svg.latex?A), ![](http://latex.codecogs.com/svg.latex?B), and ![](http://latex.codecogs.com/svg.latex?C) should be same (![](http://latex.codecogs.com/svg.latex?\dim(A)) denotes the dimension of tensor ![](http://latex.codecogs.com/svg.latex?A)):


![](http://latex.codecogs.com/svg.latex?dim(A)=dim(B)=dim(C)) and ![](http://latex.codecogs.com/svg.latex?{a_i=b_i=c_i},\forall{i\in[1,dim(A)]})


The third type is *shape-changing* operations (e.g., `ReduceMax` operation, `Pooling` operation, `Conv` operation. The shape of output tensors will change along with input tensors. Take the `Concat` operation as an example, it takes a list of tensors into a single tensor with a parameter axis that defines the dimension to concatenate on (also take two tensors as input as the example):


![](http://latex.codecogs.com/svg.latex?C_{i_1,i_2,i_{axis-1},k,i_{axis+1}...i_n}=A_{i_1,i_2,i_{axis-1},k,i_{axis+1}...i_n},\forall{k\in[1,a_{axis}]})



![](http://latex.codecogs.com/svg.latex?C_{i_1,i_2,i_{axis-1},k,i_{axis+1}...i_n}=B_{i_1,i_2,i_{axis-1},k-a_{axis},i_{axis+1}...i_n},otherwise)


The specification of `Concat` operation is as follows: all input tensors will have the same shape, except for the length of axis-th dimension; and for the axis-th dimension which input tensors concatenate on, the output tensor's length of that dimension will equal to the sum of input tensors':


![](http://latex.codecogs.com/svg.latex?dim(A)=dim(B)=dim(C))


![](http://latex.codecogs.com/svg.latex?c_i=a_i=b_i,\forall{i\in[1,dim(A)]},{i\neq{axis}})


![](http://latex.codecogs.com/svg.latex?c_i=a_i+b_i,i=axis)



## Metrics

To better present the efficiency of our generated models, we draw different metrics to measure our performance. We design 12 types of metrics in total. It contains two parts: graph-level metrics and operation-level metrics. We put forward graph-level metrics ourselves by analogy concepts in structural code coverage and combinatorial test. For operation-level metrics, we mainly follow the work by [Luo etc.](https://arxiv.org/pdf/2008.05933.pdf) They are listed below:

- **Graph-level Metrics**
  
  Let a **test set** be the set of models generated by an approach. Given a test set ![](http://latex.codecogs.com/svg.latex?I), which contains ![](http://latex.codecogs.com/svg.latex?n_I) graphs, the graph-level metrics are defined as follows. 
  
  - **Number of Operations (NOO)**, **Number of Operation Types (NOT)**, **Number of Operation Pairs (NOP)**, **Number of Shapes and Parameters (NSP)**. We put these metrics together because they all represent the diversity of a computation graph. The NOO of a graph ![](http://latex.codecogs.com/svg.latex?g) is defined as the total number of operations in ![](http://latex.codecogs.com/svg.latex?g). The NOT of a graph ![](http://latex.codecogs.com/svg.latex?g) is defined as the number of different types of operations in ![](http://latex.codecogs.com/svg.latex?g). The NOP of a graph ![](http://latex.codecogs.com/svg.latex?g) is defined as the number of different edges in ![](http://latex.codecogs.com/svg.latex?g) that connect two operations together. The NSP of a graph ![](http://latex.codecogs.com/svg.latex?g) is defined as the total number of different shapes of input tensors and parameters for each operation in graph ![](http://latex.codecogs.com/svg.latex?g). These metrics for test set  ![](http://latex.codecogs.com/svg.latex?I) are defined as the average of each of them of all the graphs in ![](http://latex.codecogs.com/svg.latex?I) : ![](http://latex.codecogs.com/svg.latex?NOX(I)=\frac{\Sigma_g{NOX_g(g)}}{n_I}), where ![](http://latex.codecogs.com/svg.latex?NOX\in\{NOO,NOT,NOP,NSP\}).
  - **Number of Tensor Paths (NTP)**. Given a graph ![](http://latex.codecogs.com/svg.latex?g) and an input tensor ![](http://latex.codecogs.com/svg.latex?t) of the model, we call a route that ![](http://latex.codecogs.com/svg.latex?t) goes along from the input to the output of the model a **Path** of tensor ![](http://latex.codecogs.com/svg.latex?t) in ![](http://latex.codecogs.com/svg.latex?g). Note that there may be more than one route from an input tensor ![](http://latex.codecogs.com/svg.latex?t) to the output of the model and we take them as different paths. The NTP of a graph ![](http://latex.codecogs.com/svg.latex?g) is defined as the number of different paths in ![](http://latex.codecogs.com/svg.latex?g). The NTP metric shows the number of different manipulations of an input tensor, which indicates the complexity of a model. The NTP of ![](http://latex.codecogs.com/svg.latex?I) is defined as the average of NTP of all the graphs in ![](http://latex.codecogs.com/svg.latex?I): ![](http://latex.codecogs.com/svg.latex?NTP(I)=\frac{\Sigma_g{NTP_g(g)}}{n_I}).
  - **Max Length of Paths (MLP)**. Given a path ![](http://latex.codecogs.com/svg.latex?p) in a graph ![](http://latex.codecogs.com/svg.latex?g), the length of ![](http://latex.codecogs.com/svg.latex?p) is defined as the number of operations in this path. The MLP metric and the ALP metric below present the depth of the model. The MLP of graph ![](http://latex.codecogs.com/svg.latex?g) is defined as the length of the longest path in ![](http://latex.codecogs.com/svg.latex?g). The MLP of ![](http://latex.codecogs.com/svg.latex?I) is defined as the average of MLP of all the graphs in ![](http://latex.codecogs.com/svg.latex?I): ![](http://latex.codecogs.com/svg.latex?MLP(I)=\frac{\Sigma_g{MLP_g(g)}}{n_I}).
  - **Average Length of Paths (ALP)** of graph ![](http://latex.codecogs.com/svg.latex?g) is defined as the average of the lengths of all the paths in ![](http://latex.codecogs.com/svg.latex?g): ![](http://latex.codecogs.com/svg.latex?ALP_g(g)=\frac{\Sigma_p{len(p)}}{n_p}), where ![](http://latex.codecogs.com/svg.latex?p) is a path in ![](http://latex.codecogs.com/svg.latex?g) and ![](http://latex.codecogs.com/svg.latex?n_p) is the number of different paths in ![](http://latex.codecogs.com/svg.latex?g). The ALP of ![](http://latex.codecogs.com/svg.latex?I) is defined as the average of ALP of all the graphs in ![](http://latex.codecogs.com/svg.latex?I): ![](http://latex.codecogs.com/svg.latex?ALP(I)=\frac{\Sigma_g{ALP_g(g)}}{n_I}).
  
- **Operation-level Metrics**
  
  An **operation corpus** is a set of operations with their attributes including the operation name and the possible number of different input degrees. Given an operation corpus ![](http://latex.codecogs.com/svg.latex?C) containing ![](http://latex.codecogs.com/svg.latex?n_C) different operations and a test set ![](http://latex.codecogs.com/svg.latex?I), 
  we first give the definition of each metric on an operator ![](http://latex.codecogs.com/svg.latex?o) in ![](http://latex.codecogs.com/svg.latex?I), denoted as ![](http://latex.codecogs.com/svg.latex?XXC_{op}(o)), and the operation-level metric of ![](http://latex.codecogs.com/svg.latex?I) is defined as the average of the metric on different operators, i.e., ![](http://latex.codecogs.com/svg.latex?XXC(I)=\frac{\Sigma_o{XXC_{op}(o)}}{n_C}), where ![](http://latex.codecogs.com/svg.latex?XXC\in\{OTC,IDC,ODC,SEC,SPC\}).
  
  - **Operation Type Coverage (OTC)** shows the diversity of operations in the test set ![](http://latex.codecogs.com/svg.latex?I). Let ![](http://latex.codecogs.com/svg.latex?{OTC}_{op}(o)) be 1 if operation ![](http://latex.codecogs.com/svg.latex?o) appears in ![](http://latex.codecogs.com/svg.latex?I), and 0 otherwise.
  - **Input Degree Coverage (IDC)** shows the diversity of the input degrees of the operations in the test set. For each operation ![](http://latex.codecogs.com/svg.latex?o) in ![](http://latex.codecogs.com/svg.latex?I), let ![](http://latex.codecogs.com/svg.latex?n_{id}(o,I)) and ![](http://latex.codecogs.com/svg.latex?n_{id}(o,C)) be the number of different input degrees of ![](http://latex.codecogs.com/svg.latex?o) in ![](http://latex.codecogs.com/svg.latex?I) and in ![](http://latex.codecogs.com/svg.latex?C) respectively, we have the IDC of the operation ![](http://latex.codecogs.com/svg.latex?o) as the ratio of ![](http://latex.codecogs.com/svg.latex?n_{id}(o,I)) and ![](http://latex.codecogs.com/svg.latex?n_{id}(o,C)): ![](http://latex.codecogs.com/svg.latex?{IDC}_{op}(o)=\frac{n_{id}(o,I)}{n_{id}(o,C)}). 
    For operations that have infinite numbers of input degrees  (e.g. `Max`), we set an upper bound for their ![](http://latex.codecogs.com/svg.latex?n_{id}(o,C)).
  - **Output Degree Coverage (ODC)** shows the diversity of output degrees of operations in ![](http://latex.codecogs.com/svg.latex?I). The ![](http://latex.codecogs.com/svg.latex?{ODC}_{op}(o)) is defined as the number of different output degrees of ![](http://latex.codecogs.com/svg.latex?o) in ![](http://latex.codecogs.com/svg.latex?I). 
  - **Single Edge Coverage (SEC)** shows the diversity of edges between the operations in ![](http://latex.codecogs.com/svg.latex?I). Let ![](http://latex.codecogs.com/svg.latex?n_e(o,I)) be the number of different edges directed from the operation ![](http://latex.codecogs.com/svg.latex?o) in ![](http://latex.codecogs.com/svg.latex?I). Supposing that each operation could be connected with all operations in ![](http://latex.codecogs.com/svg.latex?C),  we define the SEC of ![](http://latex.codecogs.com/svg.latex?o) as the ratio of ![](http://latex.codecogs.com/svg.latex?n_e(o,I)) and ![](http://latex.codecogs.com/svg.latex?n_C): ![](http://latex.codecogs.com/svg.latex?{SEC}_{op}(o)=\frac{n_e(o,I)}{n_C}). 
  - **Shapes and Parameters Coverage (SPC)** indicates the diversity of input and output tensor shapes of the operations in the test set. The ![](http://latex.codecogs.com/svg.latex?{SPC}_{op}(o)) is defined as the number of different input tensor shapes and parameters of ![](http://latex.codecogs.com/svg.latex?o) in ![](http://latex.codecogs.com/svg.latex?I). 

## Experiments and Results

We conducted 3 experiments to evaluate the efficiency of GenDice, comparing with two baselines. Also, we run 1 experiment to compare the results of different values of the parameter *pickTensorThreshold*. 

**Baselines**: 

The first baseline is purely random generation, called **'Whole Checking'**, whose relative results have a name of **'purerand'**. It relies on the compiler’s running to check whether it is satisfied with semantic specifications of deep learning model only after the construction of a whole model. 

The second baseline is also based on random generation but with incremental generation, called **'Incremental Checking'**, whose relative results have a name of **'withinc'/'winc'**. It checks whether the model satisfies the semantic specifications every time it adds a new node. 

The results of **GenDice** have a name of **'dice'**.

**Common Settings**: 

For experiment #1 and #2, we set a fixed time limitation as 1 hour and for experiment #3 and #4, we set a fixed time limitation as 15 minutes. We run GenDice and the two baselines (or other settings of *pickTensorThreshold*) separately for 5 times. We measure the results based on the metrics we raised. 

### #1. Generate models of multiple sizes in a fixed time duration, compared with other baselines

**settings**: In experiment #1, we choose the lowerbound of operation number as 1, upperbound of operation number as 30, upperbound of tensor's dimension as 5, upperbound of each dimension's length as 5, and upperbound of input tensor degrees for operations with infinite possible input degrees as 5. The *pickTensorThreshold* is 0.95. 

#### results:

The results of metrics of GenDice and other 2 baselines are presented in the table below:

|           |                 |      GenDice | Baseline1 |   Baseline2 |
| :-------: | :-------------: | -----------: | --------: | ----------: |
|           | \# valid models | **130537.2** |    4427.4 |       38106 |
|           |                 |              |           |             |
|           |       OTC       |     **100%** |  **100%** |     96.956% |
| operation |       IDC       |     **100%** |   99.782% |     96.438% |
|   level   |       ODC       | **12.30434** |   4.60434 |     4.03042 |
|  metrics  |       SEC       |   **99.29%** |   73.818% |     70.038% |
|           |       SPC       | **4717.678** |     122.4 |    186.0696 |
|           |                 |              |           |             |
|           |       NOO       |     15.50608 |   2.14308 |    15.50634 |
|           |       NOT       |  **12.6876** |    2.0754 |    11.40272 |
|   graph   |       NOP       | **13.30472** |    1.0187 |     2.54904 |
|   level   |       NSP       |      5.41996 |   1.21438 | **6.29048** |
|  metrics  |       NTP       | **25.74342** |   2.66026 |      14.401 |
|           |       MLP       |  **4.27634** |   1.70184 |     2.09338 |
|           |       ALP       |   **2.7077** |   1.50832 |     1.09786 |

We also present figures of the increase of operation-level metrics over time. 

![OTC_1_30_0.95](./imgs/OTC_1_30_0.95.png)

![IDC_1_30_0.95](./imgs/IDC_1_30_0.95.png)

![ODC_1_30_0.95](./imgs/ODC_1_30_0.95.png)

![SEC_1_30_0.95](./imgs/SEC_1_30_0.95.png)

![SPC_1_30_0.95](./imgs/SPC_1_30_0.95.png)

The distributions of different operators in GenDice and two baselines are shown in the following fig:

![distribution_1_30_0.95](./imgs/Distribution_1_30_0.95.png)

### #2. Generate models of multiple sizes in a fixed time duration, with different settings of *pickTensorThreshold*

**settings**: In experiment #2, we set the parameters except *pickTensorThreshold* the same as experiment #1, while the *pickTensorThreshold* varies among a range of [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]. Note that we only evaluate GenDice in this experiment.

#### results:

The results of metrics of different *pickTensorThreshold* values are presented in the table below:

|                 |     values      |          0.5 |      0.8 |      0.9 |         0.95 |     0.98 |         0.99 |
| :-------------: | :-------------: | -----------: | -------: | -------: | -----------: | -------: | -----------: |
|                 | \# valid models |     111297.6 | 125572.4 | 127330.2 | **130537.2** | 127104.8 |       126897 |
|                 |                 |              |          |          |              |          |              |
|                 |       OTC       |         100% |     100% |     100% |     **100%** |     100% |         100% |
| operation-level |       IDC       |         100% |     100% |     100% |     **100%** |     100% |         100% |
|     metrics     |       ODC       |      8.56522 | 11.19566 | 12.01738 | **12.30434** | 12.36524 | **12.44782** |
|                 |       SEC       |       99.29% |   99.29% |   99.29% |   **99.29%** |   99.29% |       99.29% |
|                 |       SPC       |     4204.926 | 4578.265 | 4655.713 | **4717.678** | 4682.752 |     4680.465 |
|                 |                 |              |          |          |              |          |              |
|                 |       NOO       |     15.51576 | 15.50026 | 15.50498 | **15.50608** |  15.5028 |     15.51682 |
|                 |       NOT       |     12.69428 | 12.68558 | 12.68922 |  **12.6876** | 12.68774 | **12.69548** |
|   graph-level   |       NOP       |      6.47186 |  10.8291 | 12.45178 | **13.30472** | 13.81482 | **14.00344** |
|     metrics     |       NSP       | **11.04458** |  7.37808 |  6.08246 |  **5.41996** |  5.02604 |      4.89982 |
|                 |       NTP       |     20.72918 | 22.71218 | 24.54368 | **25.74342** |  26.5563 | **26.90478** |
|                 |       MLP       |      2.92392 |   3.8029 |  4.11508 |  **4.27634** |  4.37192 |  **4.40768** |
|                 |       ALP       |      1.56998 |  2.25124 |  2.54606 |   **2.7077** |  2.80696 |   **2.8434** |



### #3.  Generate models of fixed small size in a fixed time duration, compared with other baselines

**settings**: In experiment #3, we set the parameters except the operation number of a model the same as experiment #1, while the operation number is fixed as 10. 

#### results:

The results of metrics of GenDice and other 2 baselines are presented in the table below:

|           |                 |       GenDice |  Baseline1 | Baseline2 |
| :-------: | :-------------: | ------------: | ---------: | --------: |
|           | \# valid models |   **42223.4** |       63.6 |    9263.8 |
|           |                 |               |            |           |
|           |       OTC       |      **100%** |     96.52% |    96.09% |
| operation |       IDC       |      **100%** |     95.26% |    95.31% |
|   level   |       ODC       |    **9.6957** |     2.8739 |    3.0217 |
|  metrics  |       SEC       |    **99.29%** |     20.99% |    58.64% |
|           |       SPC       | **2315.3652** |    14.6000 |   78.8391 |
|           |                 |               |            |           |
|           |       NOO       |            10 |         10 |        10 |
|           |       NOT       |    **9.0788** |     9.0094 |    8.5968 |
|   graph   |       NOP       |        7.3817 | **8.4471** |    1.5323 |
|   level   |       NSP       |    **4.0571** |     2.4145 |    5.2717 |
|  metrics  |       NTP       |   **14.2139** |    12.7480 |    9.6858 |
|           |       MLP       |        3.7878 | **4.2524** |    1.9519 |
|           |       ALP       |       2.49572 | **2.7993** |    1.1099 |

The figures of the increase of operation-level metrics over time are as follows:

![OTC_10_10_0.95](./imgs/OTC_10_10_0.95.png)

![IDC_10_10_0.95](./imgs/IDC_10_10_0.95.png)

![ODC_10_10_0.95](./imgs/ODC_10_10_0.95.png)

![SEC_10_10_0.95](./imgs/SEC_10_10_0.95.png)

![SPC_10_10_0.95](./imgs/SPC_10_10_0.95.png)

The distributions of different operators in GenDice and two baselines are shown in the following fig:

![Distribution_10_10_0.95](./imgs/Distribution_10_10_0.95.png)



### #4. Generate models of fixed large size in a fixed time duration, compared with other baselines

**settings**: In experiment #4, we set the parameters except the operation number the same as experiment #1, while the operation number is fixed as 100. 

#### results:

We found that Baseline 1 fails to generate any model of size 100 in the duration of 15 minutes. So results for  Baseline 1 are replaced by 'timeout'.

The results of metrics of GenDice and other 2 baselines are presented in the table below:

|                 |                 |       GenDice | Baseline1 | Baseline2 |
| :-------------: | :-------------: | ------------: | --------: | --------: |
|                 | \# valid models |   **11099.0** |         0 |     998.8 |
|                 |                 |               |           |           |
|                 |       OTC       |      **100%** |   timeout |    0.9565 |
| operation-level |       IDC       |      **100%** |   timeout |    0.9104 |
|     metrics     |       ODC       |   **13.7391** |   timeout |    3.2913 |
|                 |       SEC       |    **99.29%** |   timeout |   0.53846 |
|                 |       SPC       | **3577.5480** |   timeout |  77.28694 |
|                 |                 |               |           |           |
|                 |       NOO       |           100 |       100 |       100 |
|                 |       NOT       |   **40.8964** |   timeout |  28.54542 |
|   graph-level   |       NOP       |  **105.8384** |   timeout |  18.02338 |
|     metrics     |       NSP       |   **22.5433** |   timeout |  18.32588 |
|                 |       NTP       |  **269.4645** |   timeout |  86.66808 |
|                 |       MLP       |    **8.8227** |   timeout |   3.39168 |
|                 |       ALP       |    **4.7611** |   timeout |   1.22058 |

The figures of the increase of operation-level metrics over time are as follows:

![OTC_100_100_0.95](./imgs/OTC_100_100_0.95.png)

![IDC_100_100_0.95](./imgs/IDC_100_100_0.95.png)

![ODC_100_100_0.95](./imgs/ODC_100_100_0.95.png)

![SEC_100_100_0.95](./imgs/SEC_100_100_0.95.png)

![SPC_100_100_0.95](./imgs/SPC_100_100_0.95.png)

The distributions of different operators in GenDice and two baselines are shown in the following fig:

![Distribution_100_100_0.95](./imgs/Distribution_100_100_0.95.png)

## Bug Details

The table below contains bugs we have detected.

All of the bugs are listed and discussed in this section.

### Bugs on TVM/ONNX Runtime.
After finding bugs on TVM and ONNX Runtime, we reported them to the corresponding compiler developers. 

The table below contains bugs we have detected on TVM and ONNX Runtime.

![bugs](./imgs/bugs.png)

For the details of the bugs, please check *bugs.zip*.

**#1 (duplicate with #2)** It is an error bug on the backend. There is a pass named DynamicToStatic which should not be defined at level 3. It will lead the internal error when the deep learning model contains operations such as MatMul, Dropout. After our reporting, developers reordered dynamic to static and simplify inference, lower DynamicToStatic optimization level to fix it.

**#3 (duplicate with #4)** It is an error bug on the frontend. The developer from TVM has explained the cause of the bug and fixed it: “It’s due to a bad assumption made in PRelu conversion about the input layout......Our current PRelu converter assumes that incoming data is in NCHW format and that the slope will have C total elements. Neither of these are actual requirements for ONNX PReLu.”

**#5 and #6** These are error bugs on the frontend. It is because that the parameters of operations in ONNX allow default value while this is not supported in TVM.

**#7** It is an error bug on the backend. This bug is related to Gemm operator. It catches an edge case of the Relay’s type checker (Relay is the intermediate representation of TVM).

**#8** It is an error bug on the frontend. It is because that the matrix C in Gemm operation can be optional by ONNX specification but the TVM doesn’t allow that behavior.

**#9 and #10** These are error bugs on the frontend due to the inconsistent specifications between different ONNX versions on some of operations, causing the error on TVM’s frontend.

**#11** This is a wrong-result bug on the backend. The result is inconsistent between TVM and ONNX Runtime on a generated model containing Flatten and ReduceL1 operation. This bug has not been confirmed by developers yet.

**#12** This is a wrong-result bug on the backend. The way of LpPool operation’ calculation in ONNX Runtime is inconsistent with many other deep learning engines like TVM, PyTorch.

**#13** This is a wrong-result bug on the backend. When the input number of Sum operation is 5 and the input tensors are coincident, it will produce wrong result. This bug has been fixed in v0.8.0.

**#14** This is an error bug on the backend. When a BatchNormalization operation is appended by a Add operation, TVM compiler will crash. This bug has been fixed in v0.8.0.


### Bugs on the compiler named AnonymousX.

Besides two opensource deep learning compilers, we also conduct GenDice on a commercial deep learning compiler from the company AnonymousX.

It turns out 7 bugs, the details are listed as follows. We do not provide with the reproducible package for bugs on the compiler named AnonymousX due to the reason of anonymous reviewing.

**#1** It is an error bug on the frontend. Compiler does not support that the weight tensor of 𝐶𝑜𝑛𝑣 operation is an input tensor. Developer do not fix this bug due to the reason that they suppose users take this parameter as a constant because this parameter usually does not change after deployment.

**#2** It is an error bug on the backend. If the inputs of 𝑀𝑒𝑎𝑛 operation are the same tensor, then the calculation of mean operation will throw exception due to the naming error. It has already been fixed by developers.

**#3** It is an error bug on the backend. 𝑅𝑒𝑑𝑢𝑐𝑒𝑃𝑟𝑜𝑑 operation will register a buffer, but buffer size has not been assigned which leads to the incorrect parameters in calculation and further causes the final result wrong. It has already been fixed by developers.

**#4** It is an error bug on the frontend. Compiler frontend does not parse correctly for 𝑃𝑜𝑜𝑙𝑖𝑛𝑔 operation. It has already been fixed by developers.

**#5** It is an error bug on the backend. ONNX standard has changed after version 13. The compiler only implements old version and is not compatible with latest standard. It has already been fixed by developers.

**#6** It is an error bug on the backend. Compiler assumes the second input of 𝑃𝑅𝑒𝑙𝑢 holds a specific shape which is not consistent with ONNX standard.

**#7** It is a wrong-result bug on the backend. Compiler backend incorrectly covers the input data for the calculation on 𝐶𝑜𝑠 operation. It has already been fixed by developers.
