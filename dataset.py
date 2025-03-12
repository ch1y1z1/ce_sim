import numpy as np


def encode_input(n_bits, num):
    r = np.zeros(2 * n_bits)
    for idx in range(n_bits):
        if num & (1 << idx):
            r[2 * idx + 1] = 1
        else:
            r[2 * idx] = 1
    return r[::-1]


def decode_input(r):
    length = r.size
    arr = np.zeros(length, dtype=int)
    indices = np.arange(1, length, 2)
    arr[indices] = 2 ** (indices // 2)
    arr = arr[::-1]
    return int(np.sum(arr * r))


def encode_output(n_bits, num):
    r = np.zeros(n_bits)
    for idx in range(n_bits):
        if num & (1 << idx):
            r[idx] = 1
        else:
            r[idx] = 0
    return r[::-1]


def decode_output(r):
    length = len(r)
    indices = np.arange(0, length, 1)
    arr = 2 ** (indices)
    arr = arr[::-1]
    return np.sum(arr * r).astype(int)


def prepare_io_dataset(dataset_config):
    inputs = dataset_config["input"]
    if type(inputs) is str:
        inputs = np.load(inputs).tolist()
    else:
        inputs = np.array(inputs)
    ouputs = dataset_config["output"]
    if type(ouputs) is str:
        ouputs = np.load(ouputs).tolist()
    else:
        ouputs = np.array(ouputs)
    n_bits = dataset_config["n_bits"]
    masks = np.array(list(map(lambda x: encode_input(n_bits, x), inputs)))
    num_masks = len(masks)
    expected_output = np.array(list(map(lambda x: encode_output(n_bits, x), ouputs)))

    return n_bits, masks, expected_output, inputs, ouputs


if __name__ == "__main__":
    # test
    n_bits = 2
    for num in range(2 ** n_bits):
        print(
            f"{num}: {encode_input(n_bits, num)}: {decode_input(encode_input(n_bits, num))}"
        )
    for num in range(2 ** n_bits):
        print(
            f"{num}: {encode_output(n_bits, num)}: {decode_output(encode_output(n_bits, num))}"
        )

    n_bits = 3
    for num in range(2 ** n_bits):
        print(
            f"{num}: {encode_input(n_bits, num)}: {decode_input(encode_input(n_bits, num))}"
        )
    for num in range(2 ** n_bits):
        print(
            f"{num}: {encode_output(n_bits, num)}: {decode_output(encode_output(n_bits, num))}"
        )
