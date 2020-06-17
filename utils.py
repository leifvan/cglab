from matplotlib import pyplot as plt
import random
import string
import imageio
import os

def plot_diff(warped, target, i):
    _, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(warped, cmap='coolwarm', vmin=-1, vmax=1)
    axs[1].imshow(warped - target, cmap='coolwarm')
    axs[2].imshow(-target, cmap='coolwarm', vmin=-1, vmax=1)

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()


class GifExporter:
    def __init__(self):
        self.image_paths = []
        os.makedirs("data/temp/", exist_ok=True)

    def add_current_fig(self):
        random_filename = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        path = f"data/temp/{random_filename}.png"
        plt.savefig(path)
        self.image_paths.append(path)

    def save_gif(self, path, duration=None):
        with imageio.get_writer(path, mode='I', duration=duration) as writer:
            for path in self.image_paths:
                image = imageio.imread(path)
                writer.append_data(image)

        for path in self.image_paths:
            os.remove(path)