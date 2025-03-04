import copy, numpy as np
np.random.seed(0)
import time

# sigmod激活函数
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# 求激活函数导数
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# 生成训练数据集
int2binary = {}     #创建一个整数与其二进制的映射表
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]


# 输入设置
alpha = 0.4         # 学习率值越高损失函数越低
input_dim = 2
hidden_dim = 64      # 存储进微微的隐藏层的内存空间或者说隐藏层的神经元数量，设置的大时迭代次数会增大，也可能出现过拟合
output_dim = 1


# 初始化神经网络权重
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1  # 连接输入层和隐藏层的权重矩阵
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 # 连接隐藏层和输出层的权重矩阵
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # 权重矩阵
# 用以存储权重更新
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# 记录统计变量
total_train_start_time = time.time()  # 记录训练总时间开始
total_iterations = 0
total_recursions = 0
total_recursion_time = 0

# 训练
for j in range(10000):

    # 每次迭代的开始时间
    iteration_start_time = time.time()

    # 生成一个简单的加法问题（a + b = c）
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # 真实值
    c_int = a_int + b_int
    c = int2binary[c_int]

    # 预测值
    d = np.zeros_like(c)

    overallError = 0

    # 这两个列表将跟踪每个时间步骤的第 2 层导数和第 1 层的值。
    layer_2_deltas = list() # 这两个列表将跟踪每个时间步骤的第 2 层导数和第 1 层的值。
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim)) # 初始时刻没有上一层隐藏层，先设置为0

    # 记录递归过程的时间
    iteration_recursion_start_time = time.time()

    # 沿着二进制编码的位置移动
    for position in range(binary_dim):

        # 生成输入和输出
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # 隐藏层构建
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # 输出层构建
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # 第二层sigmoid激活函数的导数乘以第二层的误差，以便于后续更新权重
        layer_2_error = y - layer_2 # 计算第二层误差，比较网络的实际输出和期望输
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))    # 计算梯度并存储以实现反向传播
        overallError += np.abs(layer_2_error[0])    # 计算标量误差：每个为禁止位置的误差之和

        # 将预测值解码
        d[binary_dim - position - 1] = np.round(layer_2[0][0])  # 将输出四舍五入（为二进制值，因为它介于 0 和 1 之间）并将其存储在 d 的指定槽中。

        # 将 layer_1 的值复制到数组中，以便我们可以在下一个时间步骤中应用当前层的隐藏层。
        layer_1_values.append(copy.deepcopy(layer_1))
        
    iteration_recursion_end_time = time.time()  # 记录递归结束时间
    total_recursion_time += iteration_recursion_end_time - iteration_recursion_start_time  # 累积递归时间

    future_layer_1_delta = np.zeros(hidden_dim)     # 初始化预测的第一个隐藏层的误差梯度矩阵

    for position in range(binary_dim):  # 反向传播

        X = np.array([[a[position],b[position]]])   # 对输入数据进行索引
        layer_1 = layer_1_values[-position-1]      # 从列表中选择当前隐藏层
        prev_layer_1 = layer_1_values[-position-2]  # 从列表中选择前一个隐藏层

        # 输出层误差计算
        layer_2_delta = layer_2_deltas[-position-1]
        # 根据隐藏层的误差和当前输出层的误差，计算当前误差。
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # 保存权重更新
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    # 更新权重
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # 显示每1000次的训练结果查看训练效果
    if(j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")


'''

使用ReLU（Rectified Linear Unit）激活函数来代替Sigmoid激活函数来训练用于二进制加法的RNN（循环神经网络）。ReLU函数在许多任务中比Sigmoid函数表现得更好，因为它不会遇到Sigmoid函数在输入值非常大或非常小时梯度消失的问题。

在二进制加法任务中，RNN需要学习如何将两个二进制数逐位相加，并处理进位。使用ReLU激活函数，RNN的隐藏层可以更容易地学习到这种非线性关系。

以下是一些实现上的注意事项：

1. **输出层激活函数**：
   - 对于二进制加法，输出层通常需要一个能够输出0或1（或接近0和1的值）的激活函数。虽然ReLU本身不能输出0（除非输入为0），但您可以在输出层之后添加一个额外的步骤，将ReLU的输出转换为0或1。例如，您可以使用阈值函数，将小于某个阈值（如0.5）的输出视为0，大于或等于该阈值的输出视为1。
   - 另一种选择是使用Sigmoid函数作为输出层的激活函数，即使您在隐藏层使用ReLU。Sigmoid函数可以自然地输出0到1之间的值，并且可以通过阈值化来得到二进制输出。

2. **损失函数**：
   - 对于二进制分类任务（如这里的每一位的和是0还是1），通常使用交叉熵损失函数。即使您的输出层是ReLU加阈值化，您仍然可以使用交叉熵损失来比较预测输出（在阈值化之前）和真实标签。

3. **训练过程**：
   - 在训练过程中，您需要确保RNN能够正确地学习到二进制加法的规则，包括如何处理进位。这可能需要一些技巧，比如使用适当的学习率、正则化方法，以及足够的训练数据。

4. **数据预处理**：
   - 输入数据需要被正确地编码为二进制形式，并且可能需要被归一化或标准化以适应RNN的输入要求。

5. **RNN架构**：
   - 您可以选择使用简单的RNN、LSTM（长短期记忆网络）或GRU（门控循环单元）等不同类型的RNN架构。LSTM和GRU通常比简单的RNN更容易训练，因为它们能够更好地处理长期依赖关系。

总之，虽然ReLU激活函数在隐藏层中是一个很好的选择，但您仍然需要根据具体任务来选择输出层的激活函数和损失函数。在训练RNN进行二进制加法时，还需要注意数据预处理、RNN架构的选择以及训练过程的优化。


预测值全部为0的原因通常与以下几个因素相关，尤其是在使用 **ReLU 激活函数** 时。让我们逐一分析可能的原因，并提供解决方案。

### 1. **ReLU 激活函数的特点：**
   - ReLU 激活函数的输出为 `0`，当输入小于等于 0 时（\( \text{ReLU}(x) = \max(0, x) \)）。
   - 如果你的网络中的权重初始化不合理，或者输入值和权重之间的乘积导致输入小于零，那么 ReLU 输出的值将是 `0`。这可能导致神经元没有激活，从而影响模型的学习。

### 2. **权重初始化问题：**
   - 如果权重初始化的值过小或者权重更新得过慢，可能导致网络输出非常小的值，这使得 ReLU 激活函数输出接近 0。 
   - 你可以尝试使用 **He 初始化**（适用于 ReLU 激活函数），它通过加大初始化的方差，避免了很多神经元输出 `0` 的问题。通常，初始化权重的标准方式是：
     \[
     W \sim \mathcal{N}(0, \sqrt{\frac{2}{\text{input\_dim}}})
     \]
   - 在这个代码中，权重是通过 `2 * np.random.random(...) - 1` 来初始化的，这使得权重范围在 `-1` 到 `1` 之间。可以改为 `np.random.randn(...) * np.sqrt(2 / input_dim)` 进行初始化。

### 3. **学习率过高或过低：**
   - 如果学习率过高，网络可能在训练过程中跳跃过最优解，导致无法学习到有意义的特征。反之，如果学习率过低，可能导致梯度更新太小，网络无法有效学习。
   - 你可以尝试调低或调高 `alpha`（学习率）以观察效果。

### 4. **梯度消失问题：**
   - ReLU 是一种非线性激活函数，但它也可能遭遇梯度消失问题。虽然 ReLU 输出在正区间没有问题，但对于负值输入，ReLU 输出为 0，这会导致梯度为 0，从而使得权重更新停滞。
   - 你可以通过使用 **Leaky ReLU**（为负值部分提供一个小斜率）来缓解梯度消失的问题。`Leaky ReLU` 的公式是：
     \[
     \text{LeakyReLU}(x) = \max(\alpha x, x)
     \]
     其中，\(\alpha\) 是一个小的常数（通常是 `0.01`），它允许负值输入也有小的输出。

### 5. **激活函数的导数问题：**
   - 如果在计算导数时，ReLU 输出的梯度为 `0`，这意味着模型没有学习任何东西。需要确保 **ReLU 导数的正确计算**，即：
     - 对于 ReLU，输出大于 0 时，梯度是 `1`。
     - 对于小于或等于 0 的输出，梯度是 `0`。
   - 你可以在计算 `relu_derivative` 时调试并确保其返回的梯度值是否正确。

### 6. **梯度更新问题：**
   - `synapse_0_update`、`synapse_1_update` 和 `synapse_h_update` 在每次迭代后都需要加权更新。你可以检查这些更新的计算过程，确保梯度更新是有效的，并且学习率 `alpha` 是否足够大，能够让权重发生显著变化。

### 解决方案：

1. **权重初始化：**
   使用 He 初始化权重：
   ```python
   synapse_0 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
   synapse_1 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
   synapse_h = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)
   ```

2. **学习率调整：**
   可以适当调整 `alpha`（学习率），比如将它调低至 0.01 或 0.001。

3. **改用 Leaky ReLU：**
   试试 Leaky ReLU：
   ```python
   def leaky_relu(x, alpha=0.01):
       return np.where(x > 0, x, alpha * x)
   
   def leaky_relu_derivative(x, alpha=0.01):
       return np.where(x > 0, 1, alpha)
   ```

4. **检查梯度更新：**
   确保权重更新有效。你可以打印出每次更新的权重值，观察是否发生了变化。

### 修改后的部分代码：

```python
# He 初始化权重
synapse_0 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
synapse_1 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
synapse_h = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)

# Leaky ReLU 激活函数
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
```

### 总结：
当你看到预测值始终为 `0` 时，通常是因为 **ReLU 激活函数的负值部分导致了梯度为 0**，或者权重初始化问题导致神经元没有激活。你可以通过使用 `He 初始化`、调整学习率、或者改用 `Leaky ReLU` 来解决这些问题。
'''


import copy, numpy as np
import time

np.random.seed(0)

# sigmod 激活函数
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# 求激活函数导数
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# 生成训练数据集
int2binary = {}  # 创建一个整数与其二进制表示的映射表
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# 输入设置
alpha = 0.4  # 学习率值越高损失函数越低
input_dim = 2
hidden_dim = 64  # 存储隐藏层的内存空间，设置的大时迭代次数会增大，也可能出现过拟合
output_dim = 1

# He 初始化权重
synapse_0 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
synapse_1 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
synapse_h = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# 记录统计变量
total_train_start_time = time.time()  # 记录训练总时间开始
total_iterations = 0
total_recursions = 0
total_recursion_time = 0

# 训练
for j in range(10000):

    iteration_start_time = time.time()  # 每次迭代的开始时间

    # 生成一个简单的加法问题（a + b = c）
    a_int = np.random.randint(largest_number / 2)  # int version
    a = int2binary[a_int]  # binary encoding

    b_int = np.random.randint(largest_number / 2)  # int version
    b = int2binary[b_int]  # binary encoding

    # 真实值
    c_int = a_int + b_int
    c = int2binary[c_int]

    # 预测值
    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    iteration_recursion_start_time = time.time()  # 记录递归过程的时间

    # 沿着二进制编码的位置移动
    for position in range(binary_dim):
        total_recursions += 1  # 递归次数 +1

        # 生成输入和输出
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # 隐藏层构建
        layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error) * sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])

        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))

    iteration_recursion_end_time = time.time()  # 记录递归结束时间
    total_recursion_time += iteration_recursion_end_time - iteration_recursion_start_time  # 累积递归时间

    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):

        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]

        # error at output layer
        layer_2_delta = layer_2_deltas[-position - 1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # 每次迭代时间结束
    iteration_end_time = time.time()
    total_iterations += 1  # 迭代次数 +1

    # print out progress
    if j % 1000 == 0:
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")

# 训练总时间
total_train_end_time = time.time()
total_train_time = total_train_end_time - total_train_start_time

# 输出总训练统计信息
print(f"\nTotal training time: {total_train_time:.4f} seconds.")
print(f"Total iterations: {total_iterations}")
print(f"Total recursions: {total_recursions}")
print(f"Total recursion time: {total_recursion_time:.4f} seconds.")





#  RNN_sigmoid
'''
import copy, numpy as np
np.random.seed(0)

# sigmod激活函数
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# 求激活函数导数
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# 生成训练数据集
int2binary = {}     #创建一个证书与其二进制的映射表
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]


# 输入设置
alpha = 0.4         # 学习率值越高损失函数越低
input_dim = 2
hidden_dim = 64      # 存储进微微的隐藏层的内存空间或者说隐藏层的神经元数量，设置的大时迭代次数会增大，也可能出现过拟合
output_dim = 1


# 初始化神经网络权重
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1  # 连接输入层和隐藏层的权重矩阵
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1 # 连接隐藏层和输出层的权重矩阵
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # 权重矩阵
# 用以存储权重更新
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# 训练
for j in range(10000):

    # 生成一个简单的加法问题（a + b = c）
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # 真实值
    c_int = a_int + b_int
    c = int2binary[c_int]

    # 预测值
    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = list() # 这两个列表将跟踪每个时间步骤的第 2 层导数和第 1 层的值。
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim)) # 初始时刻没有上一层隐藏层，先设置为0

    # 沿着二进制编码的位置移动
    for position in range(binary_dim):

        # 生成输入和输出
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # 隐藏层构建
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # 输出层构建
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # 第二层sigmoid激活函数的导数乘以第二层的误差，以便于后续更新权重
        layer_2_error = y - layer_2 # 计算第二层误差，比较网络的实际输出和期望输
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))    # 计算梯度并存储以实现反向传播
        overallError += np.abs(layer_2_error[0])    # 计算标量误差：每个为禁止位置的误差之和

        # 将预测值解码
        d[binary_dim - position - 1] = np.round(layer_2[0][0])  # 将输出四舍五入（为二进制值，因为它介于 0 和 1 之间）并将其存储在 d 的指定槽中。

        # 将 layer_1 的值复制到数组中，以便我们可以在下一个时间步骤中应用当前层的隐藏层。
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)     # 初始化预测的第一个隐藏层的误差梯度矩阵

    for position in range(binary_dim):  # 反向传播

        X = np.array([[a[position],b[position]]])   # 对输入数据进行索引
        layer_1 = layer_1_values[-position-1]      # 从列表中选择当前隐藏层
        prev_layer_1 = layer_1_values[-position-2]  # 从列表中选择前一个隐藏层

        # 输出层误差计算
        layer_2_delta = layer_2_deltas[-position-1]
        # 根据未来隐藏层的误差和当前输出层的误差，计算当前隐藏层的误差。
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    # 更新权重
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # 显示每1000次的训练结果查看训练效果
    if(j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")
'''

#  RNN_tanh
'''
import copy
import numpy as np
np.random.seed(0)

# tanh 激活函数
def tanh(x):
    output = np.tanh(x)
    return output

# tanh 的导数
def tanh_output_to_derivative(output):
    return 1 - output * output

# 生成训练数据集
int2binary = {}  # 创建一个整数与二进制表示的映射表
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# 输入设置
alpha = 0.02
input_dim = 2
hidden_dim = 16
output_dim = 1

# Xavier初始化神经网络权重（适合tanh激活函数）
synapse_0 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2. / (input_dim + hidden_dim))
synapse_1 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2. / (hidden_dim + output_dim))
synapse_h = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2. / (hidden_dim + hidden_dim))

synapse_0 = np.random.randn(input_dim, hidden_dim) * 0.1
synapse_1 = np.random.randn(hidden_dim, output_dim) * 0.1
synapse_h = np.random.randn(hidden_dim, hidden_dim) * 0.1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# 训练
for j in range(10000):

    # 生成一个简单的加法问题（a + b = c）
    a_int = np.random.randint(largest_number / 2)  # int version
    a = int2binary[a_int]  # binary encoding

    b_int = np.random.randint(largest_number / 2)  # int version
    b = int2binary[b_int]  # binary encoding

    # 真实值
    c_int = a_int + b_int
    c = int2binary[c_int]

    # 预测值
    d = np.zeros_like(c)

    overallError = 0
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))  # 初始化隐藏层

    # 沿着二进制编码的位置移动
    for position in range(binary_dim):

        # 生成输入和输出
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # 隐藏层构建
        layer_1 = tanh(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        # 输出层（新的二进制表示）
        layer_2 = tanh(np.dot(layer_1, synapse_1))

        # 计算误差
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error) * tanh_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])

        # 解码估计，以便我们可以将其打印出来
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # 存储隐藏层，以便我们可以在下一个时间步使用它
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):

        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]

        # 输出层的误差
        layer_2_delta = layer_2_deltas[-position - 1]
        # 隐藏层的误差
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * tanh_output_to_derivative(layer_1)

        # 更新所有权重，以便我们可以再试一次
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    # 更新权重
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    # 清除权重更新
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # 打印进度
    if j % 1000 == 0:
        print("Error: " + str(overallError))
        print("Pred: " + str(d))
        print("True: " + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")
'''

# RNN_ReLU
'''
import copy, numpy as np
import time

np.random.seed(0)

# ReLU激活函数
def relu(x):
    return np.maximum(0, x)

# ReLU的导数
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 生成训练数据集
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

# 输入设置
alpha = 0.01
input_dim = 2
hidden_dim = 16
output_dim = 1

# 使用He初始化神经网络权重
synapse_0 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
synapse_1 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
synapse_h = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2 / hidden_dim)

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# 训练
for j in range(10000):

    # 生成一个简单的加法问题（a + b = c）
    a_int = np.random.randint(largest_number / 2)
    a = int2binary[a_int]

    b_int = np.random.randint(largest_number / 2)
    b = int2binary[b_int]

    # 真实值
    c_int = a_int + b_int
    c = int2binary[c_int]

    # 预测值
    d = np.zeros_like(c)

    overallError = 0

    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))

    # 沿着二进制编码的位置移动
    for position in range(binary_dim):

        # 生成输入和输出
        X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # 隐藏层构建
        layer_1 = relu(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))

        # 输出层
        layer_2 = relu(np.dot(layer_1, synapse_1))

        # 计算误差
        layer_2_error = y - layer_2
        layer_2_deltas.append(layer_2_error * relu_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])

        # 解码预测值
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # 存储隐藏层值
        layer_1_values.append(copy.deepcopy(layer_1))

    future_layer_1_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):

        X = np.array([[a[position], b[position]]])
        layer_1 = layer_1_values[-position - 1]
        prev_layer_1 = layer_1_values[-position - 2]

        # 输出层误差
        layer_2_delta = layer_2_deltas[-position - 1]
        # 隐藏层误差
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * relu_derivative(layer_1)

        # 更新权重
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)

        future_layer_1_delta = layer_1_delta

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0

    # 输出进度
    if j % 1000 == 0:
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")



你提供的原始代码是一个基于神经网络的二进制加法实现，而最终版本我提供的是基于神经网络的二进制减法实现。下面我将详细说明原始代码和最终版本之间的区别，以及改动的原因：

### 1. **运算类型变化：加法 → 减法**
   - **原始代码：** 进行的是加法运算，代码中的 `c_int = a_int + b_int`。
   - **最终版本：** 更改为了减法运算，代码中的 `c_int = a_int - b_int`。

   **改动的原因：** 
   - 这是最基本的变化。原始代码的目标是训练神经网络来模拟二进制加法，而你要求修改成二进制减法。

### 2. **二进制表示的扩展：负数补码表示**
   - **原始代码：** 只考虑了正整数，`int2binary` 字典仅用于映射正整数到二进制表示，范围为 `[0, 2^binary_dim - 1]`。
   - **最终版本：** 为了处理负数，我们修改了 `int2binary` 映射，增加了对负数的补码表示支持。具体而言，`int2binary` 被扩展为支持负数，并通过 `int_to_binary` 函数将负数转换为补码。

   **改动的原因：**
   - 在减法运算中，可能会出现负数（例如，`a_int - b_int` 可能会小于零）。为了正确处理负数的二进制表示，我们需要用补码来表示负数。补码表示法能够统一处理正数和负数，使得二进制算术运算更为直观。

### 3. **二进制编码的处理：**
   - **原始代码：** 使用 `np.unpackbits` 将整数转换为二进制数组，将整数 `a_int` 和 `b_int` 转换为二进制数组后，输入神经网络进行加法运算。
   - **最终版本：** 保留了 `np.unpackbits` 来进行二进制转换，但我们将其应用于处理扩展后的补码表示（对于负数）。通过 `int_to_binary` 函数支持正数和负数的转换。

   **改动的原因：**
   - 为了使代码支持负数，我们必须确保二进制表示能够反映补码，因此修改了 `int_to_binary` 函数，并调整了 `int2binary` 字典的生成方式。

### 4. **错误处理：**
   - **原始代码：** 当执行 `c = int2binary[c_int]` 时，`c_int` 必须始终在 `0` 到 `largest_number` 范围内（即正整数），否则会引发 `KeyError`。
   - **最终版本：** 扩展了 `int2binary` 字典以支持负数，因此 `c_int` 在计算过程中可以是负数，我们避免了 `KeyError` 错误。

   **改动的原因：**
   - 由于现在的操作包括了负数，我们必须确保 `c_int` 的值总是能在 `int2binary` 字典中找到对应的二进制表示。通过扩展字典的范围，使得负数的情况也能够被处理。

### 5. **训练数据生成：**
   - **原始代码：** 每次迭代时，随机生成 `a_int` 和 `b_int` 并将其转换为二进制。然后通过加法计算得到 `c_int`，最终作为目标输出 `c` 。
   - **最终版本：** 同样生成 `a_int` 和 `b_int`，并且通过减法计算得到 `c_int`。然后将 `c_int` 转换为二进制并作为目标输出 `c`。

   **改动的原因：**
   - 由于我们将运算改为减法，目标值 `c_int` 计算方式发生了变化。每次迭代时，我们仍然通过随机生成 `a_int` 和 `b_int` 来生成训练数据，但这次的目标输出 `c_int` 是通过减法而非加法得出的。

### 6. **反向传播与权重更新：**
   - **原始代码：** 在反向传播部分，网络的权重更新是根据加法任务的误差进行的。加法运算通过计算输出层的误差并将其反向传播来更新权重。
   - **最终版本：** 保持了相同的反向传播结构，唯一的区别是网络的目标（即 `y`）变为了减法的结果。

   **改动的原因：**
   - 由于网络的目标输出 `c` 发生了变化（由加法结果变为减法结果），网络会根据减法误差进行权重更新。除此之外，反向传播的结构本身没有变化。

### 总结：主要的改动和原因
1. **运算类型的改变：** 将加法改为减法，代码中的 `c_int = a_int - b_int`。
2. **补码表示支持：** 通过扩展 `int2binary` 字典支持负数的补码表示，确保减法操作中的负数能够正确表示。
3. **错误处理和边界处理：** 防止负数时的 `KeyError`，通过补码转换避免了这一问题。
4. **训练目标的变化：** 目标输出 `c` 由加法的结果变为减法的结果，但训练过程的其余部分基本保持不变。

这些改动确保了原始加法网络能够扩展为支持二进制减法的神经网络，并且能正确处理负数的二进制表示。
'''