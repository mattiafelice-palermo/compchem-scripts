# Dihedral Representation Script

## Description
This Python script processes Protein Data Bank (PDB) files to generate 2D representations with highlighted dihedral atoms. It is capable of processing either individual PDB files or all PDB files within a specified directory. The script supports specifying dihedral indices either directly through the command line or via an external file.

## Usage

To use this script, run it with Python and provide the necessary arguments. Here are some example usage scenarios:

1. **Process all PDB files in a directory**:
python script_name.py 

This command processes all PDB files in the directory from where it's launched. It expects a 'dihedrals.txt' file in folders named with the same name of the PDB file, in the parent directory of where it's executed:

root_folder/
│
├── draw_dihedral/ # Execute inside here script
│   ├── dihedral1.pdb
│   ├── dihedral2.pdb
│   ├── ...
│   ├── README.md
│
├── dihedral1/
│   ├── dihedrals.txt
│   ├── ...
|
├── dihedral2/
│   ├── dihedrals.txt
│   ├── ...
|
...


2. **Process a specific PDB file**:
python script_name.py /path/to/file.pdb

This command processes a specific PDB file. It will still look for a dihedrals.txt file as specified above.

3. **Process all PDB files in a directory with specified dihedral indices**:
python script_name.py /path/to/directory -idx 1 2 3 4

This command processes all PDB files in the specified directory using the dihedral indices 1, 2, 3, and 4.

4. **Process a specific PDB file with dihedral indices read from a file**:
python script_name.py /path/to/file.pdb -f indices.txt

This command processes a specific PDB file using dihedral indices specified in the `indices.txt` file.

## Arguments

- `file`: Path to the PDB file or directory containing PDB files. Defaults to the current directory if not provided.
- `-idx` or `--indices`: Four integers representing the indices of the atoms making up the dihedral. Example: `-idx 1 2 3 4`.
- `-f` or `--file_indices`: Path to a file containing the dihedral indices. The file should contain a single line with four integers.

## Help
For more information on how to use the script, you can invoke the help function with the `-h` flag:

python script_name.py -h
This command will display a detailed help message including the script's description and information about all the arguments that the script accepts.