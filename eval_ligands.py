import os
import argparse
import csv
import uuid
from analysis.metrics import MoleculeProperties
from typing import List
from rdkit import Chem


def main(pocket_mols: List[List[Chem.Mol]], output_dir: str):
    mol_metrics = MoleculeProperties()
    all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity = mol_metrics.evaluate(pocket_mols)

    # Generate a unique random log filename
    log_filename = f"metrics_{uuid.uuid4().hex}.csv"
    log_filepath = os.path.join(output_dir, log_filename)

    # Write the metrics to the log file in CSV format
    with open(log_filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Pocket Index", "QED", "SA", "LogP", "Lipinski", "Diversity"])
        for i, (qed_list, sa_list, logp_list, lipinski_list, diversity) in enumerate(zip(all_qed, all_sa, all_logp, all_lipinski, per_pocket_diversity)):
            # Write each molecule's metrics for the current pocket
            for j in range(len(qed_list)):
                writer.writerow([i, qed_list[j], sa_list[j], logp_list[j], lipinski_list[j], diversity])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate molecule properties for each pocket.")
    parser.add_argument("input_directory", type=str, help="Path to the directory containing SDF files.")
    parser.add_argument("output_directory", type=str, help="Path to the directory for saving log files.")
    args = parser.parse_args()

    pocket_mols = []
    for root, _, files in os.walk(args.input_directory):
        for file in files:
            if file.endswith(".sdf"):
                pocket_path = os.path.join(root, file)
                suppl = Chem.SDMolSupplier(pocket_path)
                pocket_mols.append([mol for mol in suppl if mol is not None])

    main(pocket_mols, args.output_directory)
