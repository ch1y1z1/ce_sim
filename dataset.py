import numpy as np


def encode_input(n_bits, num):
    """将输入数字编码为二进制表示的数组

    该函数将输入数字转换为二进制表示，并将每个二进制位表示为两个数字：0表示为[1,0]，1表示为[0,1]。

    参数：
        n_bits：二进制位数
        num：要编码的数字

    返回：
        编码后的数组，长度为2*n_bits
    """
    r = np.zeros(2 * n_bits)  # 初始化全0数组
    for idx in range(n_bits):
        if num & (1 << idx):  # 如果第idx位为1
            r[2 * idx + 1] = 1  # 设置对应位置为[0,1]
        else:  # 如果第idx位为0
            r[2 * idx] = 1  # 设置对应位置为[1,0]
    return r[::-1]  # 返回反转后的数组


def decode_input(r):
    """将编码后的输入数组解码回原始数字

    该函数将编码后的输入数组转换回原始数字。

    参数：
        r：编码后的数组

    返回：
        解码后的原始数字
    """
    length = r.size  # 获取数组长度
    arr = np.zeros(length, dtype=int)  # 初始化权重数组
    indices = np.arange(1, length, 2)  # 获取奇数索引
    arr[indices] = 2 ** (indices // 2)  # 设置对应位置的2的幂次方
    arr = arr[::-1]  # 反转数组
    return int(np.sum(arr * r))  # 计算加权和得到原始数字


def encode_output(n_bits, num):
    """将输出数字编码为二进制数组

    该函数将输出数字转换为二进制表示，每个二进制位直接用0和1表示。

    参数：
        n_bits：二进制位数
        num：要编码的数字

    返回：
        编码后的数组，长度为n_bits
    """
    r = np.zeros(n_bits)  # 初始化全0数组
    for idx in range(n_bits):
        if num & (1 << idx):  # 如果第idx位为1
            r[idx] = 1  # 设置为1
        else:  # 如果第idx位为0
            r[idx] = 0  # 设置为0
    return r[::-1]  # 返回反转后的数组


def decode_output(r):
    """将编码后的输出数组解码回原始数字

    该函数将编码后的输出数组转换回原始数字。

    参数：
        r：编码后的数组

    返回：
        解码后的原始数字
    """
    length = len(r)  # 获取数组长度
    indices = np.arange(0, length, 1)  # 生成索引数组
    arr = 2 ** (indices)  # 计算每个位置对应的2的幂次方
    arr = arr[::-1]  # 反转数组
    return np.sum(arr * r).astype(int)  # 计算加权和得到原始数字


def prepare_io_dataset(dataset_config):
    """准备输入输出数据集

    该函数根据数据集配置字典准备输入输出数据集。

    参数：
        dataset_config：数据集配置字典，包含以下键：
            - input：输入数据，可以是数组或文件路径
            - output：输出数据，可以是数组或文件路径
            - n_bits：二进制位数

    返回：
        元组：(n_bits, masks, expected_output, inputs, outputs)
            - n_bits：二进制位数
            - masks：编码后的输入数组
            - expected_output：编码后的输出数组
            - inputs：原始输入数据
            - outputs：原始输出数据
    """
    # 加载输入数据
    inputs = dataset_config["input"]
    if type(inputs) is str:  # 如果是文件路径，则从文件加载
        inputs = np.load(inputs)
    else:  # 否则直接转换为数组
        inputs = np.array(inputs)

    # 加载输出数据
    outputs = dataset_config["output"]
    if type(outputs) is str:  # 如果是文件路径，则从文件加载
        outputs = np.load(outputs)
    else:  # 否则直接转换为数组
        outputs = np.array(outputs)

    n_bits = dataset_config["n_bits"]  # 获取二进制位数

    # 编码输入和输出数据
    masks = np.array(list(map(lambda x: encode_input(n_bits, x), inputs)))
    expected_output = np.array(list(map(lambda x: encode_output(n_bits, x), outputs)))

    return n_bits, masks, expected_output, inputs, outputs


if __name__ == "__main__":
    # 测试编码和解码函数

    # 测试2位二进制数的编码和解码
    n_bits = 2
    for num in range(2**n_bits):
        print(
            f"{num}: {encode_input(n_bits, num)}: {decode_input(encode_input(n_bits, num))}"
        )
    for num in range(2**n_bits):
        print(
            f"{num}: {encode_output(n_bits, num)}: {decode_output(encode_output(n_bits, num))}"
        )

    # 测试3位二进制数的编码和解码
    n_bits = 3
    for num in range(2**n_bits):
        print(
            f"{num}: {encode_input(n_bits, num)}: {decode_input(encode_input(n_bits, num))}"
        )
    for num in range(2**n_bits):
        print(
            f"{num}: {encode_output(n_bits, num)}: {decode_output(encode_output(n_bits, num))}"
        )
