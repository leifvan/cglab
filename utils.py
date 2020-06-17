from matplotlib import pyplot as plt


def plot_diff(warped, target, i):
    _, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(warped, cmap='coolwarm', vmin=-1, vmax=1)
    axs[1].imshow(warped - target, cmap='coolwarm')
    axs[2].imshow(-target, cmap='coolwarm', vmin=-1, vmax=1)

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"data/plot/plot{i}.png")
    plt.show()