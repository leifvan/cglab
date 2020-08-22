from pathlib import Path
from subprocess import run
import numpy as np

BCPD_DIR = Path("bcpd_win")

_param_aliases = dict(omega='k', lamb='l', kappa='k', gamma='g')


def run_bayesian_coherent_point_drift(target_set, source_set, output="y", kernel=None, **kwargs):
    np.savetxt(BCPD_DIR / "X.txt", target_set, delimiter=',')
    np.savetxt(BCPD_DIR / "Y.txt", source_set, delimiter=',')

    # kwargs to string
    kwargs = {_param_aliases.get(k, k): v for k, v in kwargs.items()}

    params_list = [f"-s{output}"]

    if kernel is not None:
        params_list.append(f"-G{kernel}")

    for k, v in kwargs.items():
        if type(v) is bool:
            params_list.append(f"-{k}")
        else:
            params_list.append(f"-{k} {v}")

    params = ' '.join(params_list)
    run(f"{BCPD_DIR / 'bcpd.exe'} {params}", cwd=BCPD_DIR)

    return_values = []
    for out_char in output:
        return_values.append(np.loadtxt(BCPD_DIR / f"output_{out_char}.txt"))

    if len(return_values) == 1:
        return return_values[0]

    return tuple(return_values)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    target_set = np.random.random(size=(10, 2))
    source_set = np.random.random(size=(20, 2)) + 10

    plt.scatter(target_set[:, 0], target_set[:, 1])
    plt.scatter(source_set[:, 0], source_set[:, 1])
    plt.show()

    result = run_bayesian_coherent_point_drift(target_set, source_set, params="")

    plt.scatter(target_set[:, 0], target_set[:, 1])
    plt.scatter(result[:, 0], result[:, 1])
    plt.show()
