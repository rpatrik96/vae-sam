from os.path import isfile
import numpy as np
import pandas as pd

BLUE = "#1A85FF"
RED = "#D0021B"
METRIC_EPS = 1e-6

from matplotlib import rc


def plot_typography(
    usetex: bool = False, small: int = 16, medium: int = 20, big: int = 22
):
    """
    Initializes font settings and visualization backend (LaTeX or standard matplotlib).
    :param usetex: flag to indicate the usage of LaTeX (needs LaTeX indstalled)
    :param small: small font size in pt (for legends and axes' ticks)
    :param medium: medium font size in pt (for axes' labels)
    :param big: big font size in pt (for titles)
    :return:
    """

    # font family
    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})

    # backend
    rc("text", usetex=usetex)
    rc("font", family="serif")

    # font sizes
    rc("font", size=small)  # controls default text sizes
    rc("axes", titlesize=big)  # fontsize of the axes title
    rc("axes", labelsize=medium)  # fontsize of the x and y labels
    rc("xtick", labelsize=small)  # fontsize of the tick labels
    rc("ytick", labelsize=small)  # fontsize of the tick labels
    rc("legend", fontsize=small)  # legend fontsize
    rc("figure", titlesize=big)  # fontsize of the figure title


def sweep2df(
    sweep_runs,
    filename,
    save=False,
    load=False,
):
    csv_name = f"{filename}.csv"
    npy_name = f"{filename}.npz"

    if load is True and isfile(csv_name) is True and isfile(npy_name) is True:
        print(f"\t Loading {filename}...")
        return pd.read_csv(csv_name)

    if load is True and isfile(csv_name) is True and isfile(npy_name) is True:
        print(f"\t Loading {filename}...")
        npy_data = np.load(npy_name)
        val_loss_histories = npy_data["val_loss_histories"]
        val_scale_inv_histories = npy_data["val_scale_inv_histories"]

        return pd.read_csv(csv_name), val_loss_histories, val_scale_inv_histories

    data = []
    val_scale_inv_histories = []
    val_loss_histories = []
    for run in sweep_runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict

        if run.state == "finished":
            try:
                # if True:
                # print(run.name)
                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config = {k: v for k, v in run.config.items() if not k.startswith("_")}

                sam_update = config["model.sam_update"]

                try:
                    rae_update = config["rae_update"]

                except:
                    rae_update = False

                try:
                    tie_grad_coeff_sam = config["tie_grad_coeff_sam"]
                except:
                    tie_grad_coeff_sam = False

                try:
                    enc_var = config["model.enc_var"]
                except:
                    enc_var = 0

                seed_everything = config["seed_everything"]

                val_scale_inv = 1.0 / summary["val_scale"]

                val_loss = summary["val_loss"]
                val_kl = summary["val_kl"]
                val_recon_loss_no_sam = summary["val_recon_loss_no_sam"]
                val_recon_loss = summary["val_recon_loss"]
                val_recon_loss_sam = summary["val_recon_loss_sam"]

                val_loss_history = run.history(keys=[f"val_loss"])
                min_val_loss_step, min_val_loss = (
                    val_loss_history.idxmin()[1],
                    val_loss_history.min()[1],
                )
                val_loss_histories.append(val_loss_history["val_loss"])

                val_scale_history = run.history(keys=[f"val_scale"])
                max_val_scale_step, max_val_scale = (
                    val_scale_history.idxmax()[1],
                    val_scale_history.max()[1],
                )

                val_scale_inv_histories.append(1.0 / val_scale_history["val_scale"])

                min_val_scale_inv = 1.0 / max_val_scale

                val_scale_inv4min_val_loss = (
                    1.0 / val_scale_history.iloc[int(min_val_loss_step)]["val_scale"]
                )

                data.append(
                    [
                        run.name,
                        sam_update,
                        rae_update,
                        tie_grad_coeff_sam,
                        seed_everything,
                        enc_var,
                        val_scale_inv,
                        min_val_scale_inv,
                        val_scale_inv4min_val_loss,
                        val_loss,
                        min_val_loss,
                        val_kl,
                        val_recon_loss,
                        val_recon_loss_sam,
                        val_recon_loss_no_sam,
                    ]
                )

            except:
                print(f"Encountered a faulty run with ID {run.name}")

    runs_df = pd.DataFrame(
        data,
        columns=[
            "name",
            "sam_update",
            "rae_update",
            "tie_grad_coeff_sam",
            "seed_everything",
            "enc_var",
            "val_scale_inv",
            "min_val_scale_inv",
            "val_scale_inv4min_val_loss",
            "val_loss",
            "min_val_loss",
            "val_kl",
            "val_recon_loss",
            "val_recon_loss_sam",
            "val_recon_loss_no_sam",
        ],
    ).fillna(0)

    # Prune histories to the minimum length
    min_len = np.array([len(v) for v in val_loss_histories]).min()

    val_loss_histories = np.array([v[:min_len] for v in val_loss_histories])
    val_scale_inv_histories = np.array([v[:min_len] for v in val_scale_inv_histories])

    if save is True:
        runs_df.to_csv(csv_name)
        np.savez_compressed(
            npy_name,
            val_loss_history=val_loss_histories,
            val_scale_inv_history=val_scale_inv_histories,
        )

    return runs_df, val_loss_histories, val_scale_inv_histories
