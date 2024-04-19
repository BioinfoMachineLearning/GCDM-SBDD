import glob
import hydra
import os

import pandas as pd

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from posebusters import PoseBusters


def resolve_input_protein_dir(data_dir: str, dataset: str) -> str:
    """Resolve the protein directory for a given method.

    :param data_dir: The data directory.
    :param dataset: The dataset name.
    :return: The protein directory for the given method.
    """
    assert os.path.exists(data_dir), f"Data directory {data_dir} does not exist."
    if dataset == "bindingmoad":
        return os.path.join(
            data_dir,
            "processed_noH_full"
        )
    elif dataset == "crossdocked":
        return os.path.join(
            data_dir,
            "processed_crossdock_noH_full_temp"
        )
    else:
        raise ValueError(f"Invalid dataset: {dataset}")


def register_custom_omegaconf_resolvers():
    """Register custom OmegaConf resolvers."""
    OmegaConf.register_new_resolver(
        "resolve_input_protein_dir",
        lambda data_dir, dataset: resolve_input_protein_dir(data_dir, dataset),
    )


def create_molecule_table(input_molecule_dir: str, input_protein_dir: str, dataset: str) -> pd.DataFrame:
    """Create a molecule table from the inference results of a trained model checkpoint.

    :param input_molecule_dir: Directory containing the generated molecules of a trained model checkpoint.
    :param input_protein_dir: Directory containing the target protein structures for the generated molecules.
    :param dataset: The dataset name.
    :return: Molecule table as a Pandas DataFrame.
    """
    inference_sdf_results = [item for item in glob.glob(os.path.join(input_molecule_dir, "*")) if item.endswith(".sdf")]
    assert inference_sdf_results, f"No SDF files found in {input_molecule_dir}."
    dataset_protein_files = [item for item in glob.glob(os.path.join(input_protein_dir, "test", "*")) if item.endswith(".pdb")]
    assert dataset_protein_files, f"No PDB files found in {input_protein_dir}."
    new_dataset_protein_files = []
    for item in inference_sdf_results:
        if dataset == "bindingmoad":
            protein_filepath = glob.glob(
                os.path.join(input_protein_dir, "test", Path(item).stem.split("_")[0] + "*" + ".pdb")
            )[0]
        elif dataset == "crossdocked":
            protein_filepath = glob.glob(
                os.path.join(input_protein_dir, "test", "-".join(Path(item).stem.split("-")[:2]) + "*" + ".pdb")
            )[0]
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
        new_dataset_protein_files.append(protein_filepath)
    dataset_protein_files = new_dataset_protein_files
    mol_table = pd.DataFrame(
        {
            "mol_pred": [item for item in inference_sdf_results if item is not None],
            "mol_true": None,
            "mol_cond": [item for item in dataset_protein_files if item is not None],
        }
    )
    return mol_table


@hydra.main(
    version_base="1.3",
    config_path="../configs/analysis",
    config_name="molecule_analysis.yaml",
)
def main(cfg: DictConfig):
    """Analyze the generated molecules from a trained model checkpoint.

    :param cfg: Configuration dictionary from the hydra YAML file.
    """
    os.makedirs(Path(cfg.bust_results_filepath).parent, exist_ok=True)
    print(f"Processing input molecule directory {cfg.input_molecule_dir}...")
    mol_table = create_molecule_table(cfg.input_molecule_dir, cfg.input_protein_dir, cfg.dataset)
    buster = PoseBusters(config="dock", top_n=None)
    bust_results = buster.bust_table(mol_table, full_report=cfg.full_report)
    bust_results.to_csv(cfg.bust_results_filepath, index=False)
    print(f"PoseBusters results for input molecule directory {cfg.input_molecule_dir} saved to {cfg.bust_results_filepath}.")


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
