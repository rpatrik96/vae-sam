from os.path import isfile

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

    data = []
    for run in sweep_runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary = run.summary._json_dict

        if run.state == "finished":
            try:
                # if True:
                # .config contains the hyperparameters.
                #  We remove special values that start with _.
                config = {k: v for k, v in run.config.items() if not k.startswith("_")}

                sam_update = config["sam_update"]
                try:
                    rae_update = config["rae_update"]
                    tie_grad_coeff_sam = config["tie_grad_coeff_sam"]
                except:
                    rae_update = tie_grad_coeff_sam = False
                seed_everything = config["seed_everything"]
                enc_var = config["enc_var"]

                val_scale_inv = 1.0 / summary["val_scale"]

                val_loss = summary["val_loss"]
                val_kl = summary["val_kl"]
                val_recon_loss_no_sam = summary["val_recon_loss_no_sam"]
                val_recon_loss = summary["val_recon_loss"]
                val_recon_loss_sam = summary["val_recon_loss_sam"]

                data.append(
                    [
                        run.name,
                        sam_update,
                        rae_update,
                        tie_grad_coeff_sam,
                        seed_everything,
                        enc_var,
                        val_scale_inv,
                        val_loss,
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
            "val_loss",
            "val_kl",
            "val_recon_loss",
            "val_recon_loss_sam",
            "val_recon_loss_no_sam",
        ],
    ).fillna(0)

    if save is True:
        runs_df.to_csv(csv_name)

    return runs_df
