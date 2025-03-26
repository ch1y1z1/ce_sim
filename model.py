import toml
from flax import nnx

from layer import Layer
import jax.numpy as jnp

import matplotlib.pyplot as plt


class CEmodel(nnx.Module):
    """复合电磁场模拟模型
    
    由多个Layer组成的级联结构
    
    属性：
        layers：Layer实例列表
    """
    def __init__(self, config):
        """根据配置文件初始化多层结构
        
        参数：
            config：配置文件字典
        """
        layers_config = config["layers"]
        self.layers = []
        for layer_config in layers_config:
            layer = Layer(
                config["grid"].copy(),
                layer_config["input"],
                layer_config["output"],
                config["basic"],
            )
            self.layers.append(layer)

    def __call__(self, masks: jnp.ndarray):
        """前向传播
        
        参数：
            masks：输入掩码向量
        
        返回：
            输出向量
        """
        tmp = masks
        for layer in self.layers:
            tmp = layer(tmp)
        return tmp

    def viz_abs(self, mask, verbose=False):
        """可视化各层电磁场分布
        
        参数：
            mask：输入掩码向量
            verbose：是否打印调试信息
        
        返回：
            fig：matplotlib图像对象
            ax：matplotlib轴对象
        """
        num_layers = len(self.layers)
        fig, ax = plt.subplots(1, num_layers, figsize=(4 * num_layers, 6))

        funcs = []
        vmax_all = 0
        tmp = mask
        for idx, layer in enumerate(self.layers):
            if verbose:
                print(tmp)
            tmp, vmax, func = layer.viz_abs(tmp, ax=ax[idx])
            funcs.append(func)
            vmax_all = max(vmax_all, vmax)
        if verbose:
            print(tmp)

        for func in funcs:
            func(vmax_all)

        return fig, ax


if __name__ == "__main__":
    config_file = "./Configuration/2bit.toml"
    config = toml.load(config_file)
    model = CEmodel(config)

    print(
        model(
            jnp.array(
                [
                    [1, 0, 0, 1],
                    [0, 1, 0, 1],
                ]
            )
        )
    )

    model.viz_abs(jnp.array([1, 0, 0, 1]))
    plt.show()
