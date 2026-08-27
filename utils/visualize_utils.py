import matplotlib.pyplot as plt
import os
import numpy as np

def visualize_feature_maps(outputs, save_dir="feature_maps", min_channels=3):
    os.makedirs(save_dir, exist_ok=True)

    for layer_name, feature in outputs.items():
        if feature.ndim != 4:
            continue  # 只处理 [B,C,H,W] 格式

        b, c, h, w = feature.shape
        grid = max(c, min_channels)
        fig, axs = plt.subplots(1, grid, figsize=(grid * 3, 3))

        for i in range(grid):
            axs[i].imshow(feature[0, i], cmap='viridis')
            axs[i].axis('off')
            axs[i].set_title(f'{layer_name}\nChannel {i}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{layer_name.replace('.', '_')}.png"))
        plt.close()
