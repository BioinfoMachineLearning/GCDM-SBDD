import os
import sys
import glob


def get_conda_bin_dir():
    if os.path.basename(sys.executable) in ['python', 'python2', 'python3']:
        # note: the case where the current script is run
        # with the full path to the Conda Python executable
        return os.path.dirname(sys.executable)
    else:
        return ''


def pdbs_to_pdbqts(pdb_dir, pdbqt_dir, dataset, conda_bin_dir=""):
    for file in glob.glob(os.path.join(pdb_dir, '*.pdb')):
        name = os.path.splitext(os.path.basename(file))[0]
        outfile = os.path.join(pdbqt_dir, name + '.pdbqt')
        pdb_to_pdbqt(file, outfile, dataset, conda_bin_dir=conda_bin_dir)
        print 'Wrote converted file to {}'.format(outfile)


def pdb_to_pdbqt(pdb_file, pdbqt_file, dataset, conda_bin_dir=""):
    python2_exec_path = os.path.join(conda_bin_dir, 'python2')
    prepare_receptor_script_path = os.path.join(conda_bin_dir, "prepare_receptor4.py")
    if dataset == 'crossdocked':
        os.system('{} {} -r {} -o {}'.format(python2_exec_path, prepare_receptor_script_path, pdb_file, pdbqt_file))
    elif dataset == 'bindingmoad':
        os.system('{} {} -r {} -o {} -A checkhydrogens -e'.format(python2_exec_path, prepare_receptor_script_path, pdb_file, pdbqt_file))
    else:
        raise NotImplementedError
    return pdbqt_file


if __name__ == '__main__':
    conda_bin_dir = get_conda_bin_dir()
    pdbs_to_pdbqts(sys.argv[1], sys.argv[2], sys.argv[3], conda_bin_dir=conda_bin_dir)
