import autograd.numpy as npa


def encode_input(n_bits, num):
    r = npa.zeros(2 * n_bits)
    for idx in range(n_bits):
        if num & (1 << idx):
            r[2 * idx + 1] = 1
        else:
            r[2 * idx] = 1
    return r[::-1]


def decode_input(r):
    length = r.size
    arr = npa.zeros(length, dtype=int)
    indices = npa.arange(1, length, 2)
    arr[indices] = 2 ** (indices // 2)
    arr = arr[::-1]
    return int(npa.sum(arr * r))


def encode_output(n_bits, num):
    r = npa.zeros(n_bits)
    for idx in range(n_bits):
        if num & (1 << idx):
            r[idx] = 1
        else:
            r[idx] = 0
    return r[::-1]


def decode_output(r):
    length = len(r)
    indices = npa.arange(0, length, 1)
    arr = 2 ** (indices)
    arr = arr[::-1]
    return int(npa.sum(arr * r))


if __name__ == "__main__":
    # test
    n_bits = 2
    for num in range(2**n_bits):
        print(
            f"{num}: {encode_input(n_bits, num)}: {decode_input(encode_input(n_bits, num))}"
        )
    for num in range(2**n_bits):
        print(
            f"{num}: {encode_output(n_bits, num)}: {decode_output(encode_output(n_bits, num))}"
        )

    n_bits = 3
    for num in range(2**n_bits):
        print(
            f"{num}: {encode_input(n_bits, num)}: {decode_input(encode_input(n_bits, num))}"
        )
    for num in range(2**n_bits):
        print(
            f"{num}: {encode_output(n_bits, num)}: {decode_output(encode_output(n_bits, num))}"
        )
