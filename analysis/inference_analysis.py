import ast
import glob
import hydra

import numpy as np
import pandas as pd
import scipy.stats as st

from omegaconf import DictConfig
from typing import Iterable


def calculate_mean_and_conf_int(data: Iterable, alpha: float = 0.95) -> tuple[float, tuple[float, float]]:
    """
    Calculate and report the mean and confidence interval of the data.
    
    :param data: Iterable data to calculate the mean and confidence interval.
    :param alpha: Confidence level (default: 0.95).
    :return: Tuple of the mean and confidence interval.
    """
    conf_int = st.t.interval(
        alpha=alpha, df=len(data) - 1,
        loc=np.mean(data),
        scale=st.sem(data),
    )
    return np.mean(data), conf_int


@hydra.main(
    version_base="1.3",
    config_path="../configs/analysis",
    config_name="inference_analysis.yaml",
)
def main(cfg: DictConfig):
    """Analyze the inference results of a trained model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    # load the baseline molecule generation metrics and Vina energies
    metric_results_filepaths = glob.glob(cfg.input_molecule_metrics_csv_filepath.replace(".csv", "*.csv"))
    assert metric_results_filepaths, f"Baseline molecule generation metrics file not found via: {cfg.input_molecule_metrics_csv_filepath.replace('.csv', '*.csv')}"
    metric_results = pd.read_csv(metric_results_filepaths[0])
    vina_energies = pd.read_csv(cfg.input_molecule_vina_energy_csv_filepath)
    vina_scores = [score for score_list in vina_energies["scores"].apply(ast.literal_eval).values for score in score_list]

    # calculate the corresponding means and confidence intervals for each metric
    vina_energy_mean, vina_energy_conf_int = calculate_mean_and_conf_int(vina_scores)
    qed_mean, qed_conf_int = calculate_mean_and_conf_int(metric_results["QED"].values)
    sa_mean, sa_conf_int = calculate_mean_and_conf_int(metric_results["SA"].values)
    lipinski_mean, lipinski_conf_int = calculate_mean_and_conf_int(metric_results["Lipinski"].values)
    diversity_mean, diversity_conf_int = calculate_mean_and_conf_int(metric_results["Diversity"].values)

    # report the results
    print(f"Mean Vina energy: {vina_energy_mean} with confidence interval: ±{vina_energy_conf_int[1] - vina_energy_mean}")
    print(f"Mean QED: {qed_mean} with confidence interval: ±{qed_conf_int[1] - qed_mean}")
    print(f"Mean SA: {sa_mean} with confidence interval: ±{sa_conf_int[1] - sa_mean}")
    print(f"Mean Lipinski: {lipinski_mean} with confidence interval: ±{lipinski_conf_int[1] - lipinski_mean}")
    print(f"Mean Diversity: {diversity_mean} with confidence interval: ±{diversity_conf_int[1] - diversity_mean}")

    # evaluate and report PoseBusters results
    pb_results = pd.read_csv(cfg.bust_results_filepath)
    pb_results["valid"] = (
        pb_results["mol_pred_loaded"].astype(bool)
        & pb_results["mol_cond_loaded"].astype(bool)
        & pb_results["sanitization"].astype(bool)
        & pb_results["all_atoms_connected"].astype(bool)
        & pb_results["bond_lengths"].astype(bool)
        & pb_results["bond_angles"].astype(bool)
        & pb_results["internal_steric_clash"].astype(bool)
        & pb_results["aromatic_ring_flatness"].astype(bool)
        & pb_results["double_bond_flatness"].astype(bool)
        & pb_results["internal_energy"].astype(bool)
        & pb_results["passes_valence_checks"].astype(bool)
        & pb_results["passes_kekulization"].astype(bool)
    )
    num_pb_valid_molecules_mean, num_pb_valid_molecules_conf_int = calculate_mean_and_conf_int(pb_results["valid"])
    print(f"Mean percentage of PoseBusters-valid molecules: {num_pb_valid_molecules_mean * 100} % with confidence interval: ±{(num_pb_valid_molecules_conf_int[1] - num_pb_valid_molecules_mean) * 100}")

    num_pb_valid_molecules_without_clashes_mean, num_pb_valid_molecules_without_clashes_conf_int = calculate_mean_and_conf_int(pb_results["valid"] & (pb_results["num_pairwise_clashes_protein"] == 0))
    print(f"Mean percentage of PoseBusters-valid molecules without protein-ligand steric clashes: {num_pb_valid_molecules_without_clashes_mean * 100} % with confidence interval: ±{(num_pb_valid_molecules_without_clashes_conf_int[1] - num_pb_valid_molecules_without_clashes_mean) * 100}")


if __name__ == "__main__":
    main()
