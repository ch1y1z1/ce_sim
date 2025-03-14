from matplotlib import pyplot as plt
import toml
import autograd.numpy as npa
from dataset_chi import decode_output, encode_input, encode_output
from layer import Layer
import time
import os

if __name__ == "__main__":
    t = time.asctime(time.localtime(time.time()))
    config = "./Configuration/5bit.toml"
    config = toml.load(config)
    save_load_config = config["save_load"]

    layers_config = config["layers"]
    layers = []
    num_layers = len(layers_config)
    for layer_config in layers_config:
        layer = Layer(
            config["grid"],
            layer_config["input"],
            layer_config["output"],
            config["basic"],
        )
        layers.append(layer)

    if "load_path" in save_load_config:
        rhos = npa.load(save_load_config["load_path"])
        rhos = rhos.reshape((num_layers, -1))
        for i, layer in enumerate(layers):
            layer.use_rho(rhos[i])

    # prepare dataset:
    dataset_config = config["dataset"]
    inputs = dataset_config["input"]
    if type(inputs) is str:
        inputs = npa.load(inputs).tolist()
    ouputs = dataset_config["output"]
    if type(ouputs) is str:
        ouputs = npa.load(ouputs).tolist()
    n_bits = dataset_config["n_bits"]
    masks = list(map(lambda x: encode_input(n_bits, x), inputs))
    expected_output = list(map(lambda x: encode_output(n_bits, x), ouputs))

    def obj(rhos):
        rhos = rhos.reshape((num_layers, -1))
        tmp = masks
        for layer, rho in zip(layers, rhos):
            tmp = layer.objective(rho, tmp)
        mse = npa.mean((npa.array(tmp) - npa.array(expected_output)) ** 2, axis=1)

        return npa.sum(mse)

    rhos = npa.array([la.rho for la in layers])

    def cb_test(idx, _, rhos):
        if idx % save_load_config["test_freq"] == 0:
            rhos = rhos.reshape((len(layers), -1))
            predict = masks
            for idj in range(num_layers):
                predict = layers[idj].objective(rhos[idj], predict)
            print(predict)

            predict = npa.array(predict).round().astype(int)
            pred_label = list(map(decode_output, predict.tolist()))

            print(f"real: {ouputs}")
            print(f"pred: {pred_label}")

            accu = npa.sum(npa.array(pred_label) == npa.array(ouputs)) / len(ouputs)
            print(f"accuracy: {accu}")

            if "save_path" in save_load_config:
                npa.save(
                    f"{save_load_config['save_path']}/{t}/{idx}.npy",
                    rhos,
                )

    rhos = rhos.reshape((len(layers), -1))
    for idx in range(num_layers):
        layers[idx].use_rho(rhos[idx])

    # cb_test(0, (), rhos)

    # masks = masks[:2]
    num_masks = len(masks)

    # plot
    os.makedirs(f"{save_load_config['save_path']}/{t}/epochs_{0}", exist_ok=True)
    for ndx in range(num_masks):
        fig, ax = plt.subplots(1, num_layers, figsize=(4 * num_layers, 4))
        # ax = ax.flatten(1)

        funcs = []
        vmax_all = 0

        tmp = masks[ndx]
        for idx in range(num_layers):
            print(tmp)
            tmp, vmax, func = layers[idx].viz_abs(tmp, ax=ax[idx])
            funcs.append(func)
            vmax_all = max(vmax_all, vmax)
        print(tmp)

        for func in funcs:
            func(vmax_all)

        # save
        plt.savefig(f"{save_load_config['save_path']}/{t}/epochs_{0}/{inputs[ndx]}_{ouputs[ndx]}.png")
        plt.close(fig)
