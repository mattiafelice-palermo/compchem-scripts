import re
import subprocess
import os
import shutil
import argparse
import textwrap
import sys
from functools import partial
import math
import itertools

import numpy as np
import jobdispatcher as jd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate

def scan_workflow(args):
    # Read parameters and store in a dictionary
    parameters, hydrogen_sn = mdp_to_dictionary(args.topology, include_hydrogens=args.include_hydrogens)



    if not args.skip_orca_calc:
        # Create a ORCA file for each parameter
        write_orca_input(parameters, args.input_file, threads_per_calc=args.threads_per_calc, cpus=args.total_cores, scan_steps=args.scan_steps, charge=args.charge, multiplicity=args.multiplicity)   

        # Create jobs, each cding in the directory and calling orca input and launch through jobdispatcher
        launch_orca_jobs(parameters, args.threads_per_calc, args.total_cores)

    else:
        # Skipping ORCA calcs
        print("Skipping ORCA calculations as requested. Assuming all necessary files are already present.")
    
    if args.abf and not args.skip_gromacs_calc:
        run_gromacs_abf(parameters, args.topology, args.threads_per_calc, args.total_cores, args.scan_steps, args.input_file, args.range)
    elif args.constrained_opt and not args.skip_gromacs_calc:
        # Convert xyz files to g96 to keep fine geometry details - all geometries in a single g96 file
        # TODO disable lincs constraint by default
        xyz_to_g96(parameters)
        # Option 1: recalculate the energies with MM force field
        run_gromacs(parameters, args.topology, args.threads_per_calc, args.total_cores)
    # Option 2: for each geometry, constrain bonded interaction and minimize
    elif not args.skip_gromacs_calc:
        xyz_to_g96(parameters, split=True)
        run_gromacs_relaxed(parameters, args.topology, args.threads_per_calc, args.total_cores, args.scan_steps, args.include_hydrogens, hydrogen_sn)
    else:
        # Skip all calculations
        print("Skipping GROMACS calculations as requested. Assuming all necessary files are already present.")
        
    # Minimize difference to obtain optimized parameters
    
    # Process data
    results = {}
    for parameter in parameters:
        result = process_data_and_plot(parameter, parameters[parameter][0], os.getcwd(), args.fixed_fit_eq_value)
        results[parameter] = result

    # Prepare data for tabulation
    headers = ['Parameter', 'MM K (kcal/mol/Å²)', 'QM K (kcal/mol/Å²)', 'Cumulative Error (kcal/mol)', 'MAE (kcal/mol)', 'RMSE (kcal/mol)', 'R²']
    table = []

    for parameter, metrics in results.items():
        # Create a row for each parameter, extracting each metric
        row = [parameter.replace(" ", "_"), metrics['mm_k'], metrics['qm_k'], metrics['cumulative'], metrics['mae'], metrics['rmse'], metrics['r_squared']]
        table.append(row)

    # Sort the table by the MAE column (index 3)
    table = sorted(table, key=lambda x: x[4])

    # Print the table
    print(tabulate(table, headers=headers))#, tablefmt='grid'))


def mdp_to_dictionary(file_path, include_hydrogens=False) -> dict:
    """
    Converts MDP file contents to a pandas DataFrame representing the simulation parameters.

    Args:
        file_path (str): Path to the MDP file.

    Returns:
        dict: Dictionary containing the parameters extracted from the MDP file.
    """

    bond_lines = []
    angle_lines = []
    read_bonds = False
    read_angles = False
    read_atomtypes = False
    read_atoms = False

    # to exclude bonds containing hydrogens from the scan
    hydrogen_at = [] # atomtypes corresponding to an atom number of 1
    hydrogen_sn = [] # serial number of hydrogen atoms in the structure

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith(';') or not line.strip(): # skip empty lines or comments
                continue
            if "[ atomtypes ]" in line:
                read_atomtypes = True
                continue
            if read_atomtypes and line.startswith('['):
                read_atomtypes = False
            if read_atomtypes:
                atom_number = int(line.split()[1])
                if atom_number == 1:
                    atom_type = line.split()[0]
                    hydrogen_at.append(atom_type)
            if "[ atoms ]" in line:
                read_atoms = True
                continue
            if read_atoms and line.startswith('['):
                read_atoms = False
            if read_atoms:
                atom_type = line.split()[1]
                if atom_type in hydrogen_at:
                    hydrogen_sn.append(int(line.split()[0]))
            if "[ bonds ]" in line:
                read_bonds = True
                continue
            if read_bonds:
                bond_lines.append(line)
            if read_bonds and line.startswith('['):
                read_bonds = False
            if "[ angles ]" in line:
                read_angles = True
                continue
            if read_angles:
                angle_lines.append(line)
            if read_angles and line.startswith('['):
                read_angles = False

    # Regex pattern to match numbers and floating point numbers, ignoring optional comments
    pattern_bond = r'\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)(\s*;.*)?'
    pattern_angle = r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)(\s*;.*)?'

    params = {}

    # Iterate over each line in the data list to extract bond parameters
    for line in bond_lines:
        match = re.match(pattern_bond, line)
        if match:
            # Extract all groups, but filter out the comment
            values = match.groups()[:-1]  # Ignore the last group which is the comment      
            if (not include_hydrogens) and (int(values[0]) in hydrogen_sn or int(values[1]) in hydrogen_sn):
                print(f"Skipping bond {values[0]}-{values[1]} as it contains a hydrogen atom")
                continue
            key = f"b {values[0]} {values[1]}"
            params.setdefault(key, []).append([values[3], values[4]])

    # Iterate over each line in the angle data list to extract bond parameters
    # TODO: why are hydrogens now excluded for angles?
    for line in angle_lines:
        match = re.match(pattern_angle, line)
        if match:
            # Extract all groups, but filter out the comment
            values = match.groups()[:-1]  # Ignore the last group which is the comment
            key = f"a {values[0]} {values[1]} {values[2]}"
            params.setdefault(key, []).append([values[4], values[5]])
    return params, hydrogen_sn

def write_orca_input(parameters, geometry, method=None, basis_set=None, charge=None, multiplicity=None, threads_per_calc= None, cpus=None, memory=None, scan_steps = None):
    """
    Writes an ORCA input file for each parameter set.
    """
    if method is None:
        method = 'b97-3c'
    if basis_set is None:
        basis_set = ''
    if charge is None:
        charge = 0
    if multiplicity is None:
        multiplicity = 1
    if threads_per_calc is None:    
        threads_per_calc = 1
    if cpus is None:
        cpus = 1
    if memory is None:
        memory = 4000

    cwd = os.getcwd()

    for parameter in parameters:
        # Create a folder for each parameter
        folder_name = parameter.replace(' ', '_')
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        splitted = [str(int(atom_id)-1) for atom_id in parameter.split()[1:]]
        scan_type = parameter.split()[0].upper()
        atom_ids = ' '.join(splitted)
        eq_value = float(parameters[parameter][0][0])

        if scan_type == 'B':
            eq_value *= 10
        start = eq_value - eq_value * 0.1
        end = eq_value + eq_value * 0.1
        steps = scan_steps

        input_path = os.path.join(cwd, folder_name, f'{folder_name}.inp')
        geometry_path = os.path.join(cwd, geometry)
        orca_input =  f"! {method} {basis_set} Opt\n%pal nprocs {threads_per_calc} end\n%maxcore {memory}\n"
        orca_input += f"%geom Scan\n{scan_type} {atom_ids} = {start}, {end}, {steps}\nend\nend\n"
        orca_input += f"* xyzfile {charge} {multiplicity} {geometry_path}\n"
        with open(input_path, 'w') as f:
            f.write(orca_input)

def launch_orca_jobs(parameters, threads, total_cores):

    cwd = os.getcwd()
    def run_orca(parameter_id):
        folder_path = os.path.join(cwd, parameter_id)
        # Launch the job
        try:
            subprocess.run(f"module load orca && $(which orca) {parameter_id}.inp > {parameter_id}.out 2> {parameter_id}.err", shell=True, check=True, cwd=folder_path, executable='/bin/bash')
        except Exception as e:
            print(e)
        
    jobs = []
    for parameter in parameters:
        parameter_id = parameter.replace(' ', '_')
        job = jd.Job(name=f"{parameter_id}", function=lambda parameter_id=parameter_id : run_orca(f"{parameter_id}"), cores=threads)
        jobs.append(job)

    dispatcher = jd.JobDispatcher(jobs, maxcores=total_cores, engine="multiprocessing")
    results = dispatcher.run()

    return results


def xyz_to_g96(parameters, split=False, box_dimensions=[20.0, 20.0, 20.0]):
    for parameter in parameters:
        parameter_id = parameter.replace(' ', '_') 
        xyz_filename = f'{parameter_id}/{parameter_id}.allxyz'
        xyz_path = os.path.join(os.getcwd(), xyz_filename)
        g96_filename = f'{parameter_id}/{parameter_id}'
        positions = read_xyz_positions(xyz_path)

        # Remove existing file to avoid appending to old content, if present
        g96_filename = f'{g96_filename}.g96'
        split_filename = f'{g96_filename[:-4]}'
        if os.path.exists(f'{g96_filename}'):
            print(f'Removing existing file {g96_filename}')
            os.remove(f'{g96_filename}')
        for i, position in enumerate(positions):
            if split:
                g96_filename =f'{split_filename}_{i}.g96' # -4 to remove extension
                if os.path.exists(f'{g96_filename}'):
                    os.remove(f'{g96_filename}')
            
            write_g96(position[2:], g96_filename) # Start from the third line to skip headers
    return g96_filename

def read_xyz_positions(xyz_filename):
    with open(xyz_filename, 'r') as file:
        xyz_positions = [[]]
        counter = 0
        for line in file:
            # Prepare a new list for the next set of positions upon encountering '>'
            if line.startswith('>'):
                counter +=1
                xyz_positions.append([])
                continue
            xyz_positions[counter].append(line)
    return xyz_positions

def write_g96(positions, output_filename, box_size=(20.0, 20.0, 20.0), mode='a'):
    with open(output_filename, mode) as file:
        file.write("TITLE\n")
        file.write("Converted from XYZ format\n")
        file.write("END\n")
        file.write("POSITIONRED\n")
        for i, position in enumerate(positions):
            pos = position.split()[1:]
            # Format the coordinates with sufficient space and precision
            formatted_pos = "{:15.9f}{:15.9f}{:15.9f}".format(float(pos[0])/10., float(pos[1])/10., float(pos[2])/10.)
            file.write(formatted_pos + "\n")
        file.write("END\n")
        file.write("BOX\n")
        file.write("{:15.9f}{:15.9f}{:15.9f}\n".format(box_size[0], box_size[1], box_size[2]))
        file.write("END\n")

def run_gromacs(parameters, topology, threads, total_cores):
    cwd = os.getcwd()

    def run_job(parameter_id, cwd, topology, threads):
        path = os.path.join(cwd, parameter_id)
        grompp(parameter_id, f'{cwd}/{topology}', path)
        mdrun(parameter_id, path, threads)
        extract_energy(parameter_id, path)      


    write_gromacs_mdp(parameters, cwd)

    jobs = []

    for parameter in parameters:
        parameter_id = parameter.replace(' ', '_')
        job = jd.Job(name=f"{parameter_id}", function=run_job, arguments = [parameter_id, cwd, topology, threads], cores=threads)
        jobs.append(job)

    dispatcher = jd.JobDispatcher(jobs, maxcores=total_cores, engine="multiprocessing")
    results = dispatcher.run()

def write_gromacs_mdp(parameters, cwd):
    mdp_file_content = textwrap.dedent(
        f""";--- RUN CONTROL
            integrator              = md-vv
            nsteps                  = 10000000 ; IGNORED
            dt                      = 0.0001
            ;--- OUTPUT CONTROL
            nstvout                 = 10000
            nstenergy               = 10000
            nstcalcenergy           = 1
            nstlog                  = 10000
            nstcomm                 = 1
            ;NEIGHBOR SEARCHING
            cutoff-scheme           = verlet
            pbc                     = xyz
            ;--- ELECTROSTATICS
            coulombtype             = PME
            rcoulomb                = 1.1 ; uncomment if you're not going to use mdrun -tunepme
            ;--- VAN DER WAALS
            vdwtype                 = cut-off
            rvdw                    = 1.1
            vdw-modifier            = potential-switch
            rvdw-switch             = 1.05
            ;--- BONDS
            constraints             = h-bonds
            constraint-algorithm    = lincs
        """)

    with open(f'{cwd}/resample.mdp', 'w') as f:
        f.write(mdp_file_content)

    for parameter in parameters:
        parameter_id = parameter.replace(' ', '_')  # Convert parameter into a valid directory/filename
        link_path = os.path.join(cwd, parameter_id, 'resample.mdp')  # Define full path for the new symbolic link

        # Check if the link already exists, if so, remove it first
        if os.path.islink(link_path):
            os.unlink(link_path)
        
        # Create a symbolic link pointing to 'target' with the name 'link_name' in the directory 'parameter_id'
        os.symlink(f'{cwd}/resample.mdp', link_path)

def grompp(parameter_id, topology, path, scan_step=None):
    if scan_step is None:
        scan_step = ""
    else:
        scan_step = f'_{scan_step}'
        topology_base =  os.path.basename(topology)[:-4]
        topology = f'{topology_base}{scan_step}.top'
    # Run GROMPP
    try:
        grompp = subprocess.run(f'module load gromacs && gmx grompp -f resample.mdp'
                                f' -c {parameter_id}{scan_step}.g96 -p {topology} -o resample{scan_step}.tpr -maxwarn 2 > grompp{scan_step}.out 2> grompp{scan_step}.err',
                                shell=True, cwd=path, check=True, executable='/bin/bash')
    except:
        raise RuntimeError(f"Failed to run GROMACS grompp command in {parameter_id}{scan_step}")

def mdrun(parameter_id, path, threads, scan_step=None, rerun=True):
    if scan_step is None:
        scan_step = ""
        rerun_flag = f'-rerun {parameter_id}.g96'
    else:
        scan_step = f'_{scan_step}'
        rerun_flag = ""

    # Crazy madness for backward compatibility, but ok!
    if rerun:
        rerun_flag = f'-rerun {parameter_id}.g96'
    else:
        rerun_flag = ""
    # Run MDRUN
    try:
        mdrun = subprocess.run(f'module load gromacs && gmx mdrun -nt {threads}'
                               f' -s resample{scan_step}.tpr -deffnm {parameter_id}{scan_step} {rerun_flag} > mdrun{scan_step}.out 2> mdrun{scan_step}.err', shell=True, cwd=path, check=True, executable='/bin/bash')

    except:
        raise RuntimeError(f"Failed to run GROMACS mdrun command in {parameter_id}{scan_step}")

def extract_energy(parameter_id, path, scan_step=None, abf=False):
    if scan_step is None:
        scan_step = ""
    else:
        scan_step = f'_{scan_step}'
    try:
        result = subprocess.run(f'module load gromacs && gmx energy -f {parameter_id}{scan_step}.edr -o {parameter_id}{scan_step}.xvg', 
                                input='Potential\n\n', check=True, text=True, capture_output=True, 
                                cwd=path, executable='/bin/bash', shell=True)
    except Exception as e:
        print(e)
        raise RuntimeError(f"Failed to run GROMACS energy command in {parameter_id}{scan_step}")
    
    print(f"Energy extracted from {parameter_id}{scan_step}.edr")

def run_gromacs_relaxed(parameters, topology, threads, total_cores, scans, include_hydrogens, hydrogen_sn):
    cwd = os.getcwd()

    def run_job(parameter_id, cwd, topology, threads, scan_step=None):
        path = os.path.join(cwd, parameter_id)
        grompp(parameter_id, f'{cwd}/{topology}', path, scan_step)
        mdrun(parameter_id, path, threads, scan_step)
        extract_energy(parameter_id, path, scan_step)    


    write_gromacs_mdp_minimization(cwd, parameters)
    all_parameters, hydrogen_sn = mdp_to_dictionary(topology, include_hydrogens=include_hydrogens)

    jobs = []

    for parameter in parameters:
        for scan_step in range(scans):
            parameter_id = parameter.replace(' ', '_') 
            write_constrained_topology(parameter, scan_step, topology, cwd, include_hydrogens, hydrogen_sn, all_parameters)
            job = jd.Job(name=f"{parameter_id}_{scan_step}", function=run_job, arguments = [parameter_id, cwd, topology, threads], keyword_arguments = {'scan_step': scan_step}, cores=threads)
            jobs.append(job)

    dispatcher = jd.JobDispatcher(jobs, maxcores=total_cores, engine="multiprocessing")
    results = dispatcher.run()
    for parameter in parameters:
        append_energies(parameter, scans, cwd)

def append_energies(parameter, scans, cwd):
    path = os.path.join(cwd, parameter)
    parameter_id = parameter.replace(' ', '_') 

    xvg_lines = []

    for scan_step in range(scans):
        with open(f'{cwd}/{parameter_id}/{parameter_id}_{scan_step}.xvg', 'rb') as f:
            f.seek(-2, 2)  # Jump to the second last byte.
            while f.read(1) != b'\n':  # Keep reading backwards until you find the new line.
                f.seek(-2, 1)
            xvg_lines.append(f.readline().decode()) # the last line of the file

    with open(f'{cwd}/{parameter_id}/energy.xvg', 'w') as f:
        for line in xvg_lines:
            f.write(line)

def write_constrained_topology(parameter_id, scan_step, topology, cwd, include_hydrogens, hydrogen_sn, parameters):
    # 1 calculate bond(s) distances
    if parameter_id.startswith('b'):
        atoms = [int(parameter_id.split()[1]), int(parameter_id.split()[2])]
    else:
        atoms = [int(parameter_id.split()[1]), int(parameter_id.split()[2]), int(parameter_id.split()[3])] 

    bonded_atoms = [ [int(key.split()[1]), int(key.split()[2])] for key in list(parameters.keys()) if key.startswith('b') ]

    # Check if any of the atoms in the bonded interaction are hydrogens 
    constrained_hydrogens = []
    for atom in atoms:
        if atom in hydrogen_sn:
            constrained_hydrogens.append(atom)

    if len(constrained_hydrogens) > 0 and not include_hydrogens:
        atoms = []

    # if len(constrained_hydrogens) == 1 and include_hydrogens and parameter_id.startswith('a'):
    #     # find the other bonded atom
    #     hydrogen = constrained_hydrogens[0]
    #     atoms.remove(hydrogen)
    #     for bond in bonded_atoms:
    #         print(bond)
    #         if hydrogen in bond and atoms[0] in bond:
    #             atoms.remove(atoms[0])
    #             break
    #         if hydrogen in bond and atoms[1] in bond:
    #             atoms.remove(atoms[0])
    #             break
    #     atoms = (hydrogen, atoms[0])
    
    # if len(constrained_hydrogens) == 2 and include_hydrogens and parameter_id.startswith('a'):
    #     atoms = (constrained_hydrogens[0], constrained_hydrogens[1])

            

    parameter_id = parameter_id.replace(' ', '_')

    coordinates = get_atom_coordinates(f'{cwd}/{parameter_id}/{parameter_id}_{scan_step}.g96', atoms)

    distance_dict = {}
    # Generate all combinations of atom pairs from the list
    for atom1, atom2 in itertools.combinations(atoms, 2):
        # Retrieve coordinates for both atoms
        coord1 = coordinates[atom1]
        coord2 = coordinates[atom2]
        # Calculate distance and store it in the dictionary
        distance_dict[(atom1, atom2)] = calculate_distance(coord1, coord2)

    topology_root = topology[:-4]
    
    # 2 copy the topology file from cwd to the parameter folder
    os.system(f'cp {topology} {cwd}/{parameter_id}/{topology_root}_{scan_step}.top')

    # 3 add them at the end of the file, it works!
    with open(f'{cwd}/{parameter_id}/{topology_root}_{scan_step}.top', 'a') as f:
        if atoms:
            f.write('\n\n[ constraints ]\n')
        for atom1, atom2 in distance_dict:
            f.write(f'{atom1} {atom2} 1 {distance_dict[(atom1, atom2)]}\n')
    

def get_atom_coordinates(filename, atom_numbers):
    """
    Read a file and return the coordinates of specified atoms.

    :param filename: Path to the file containing atom coordinates.
    :param atom_numbers: A list of atom numbers (1-based index) for which coordinates are requested.
    :return: A dictionary with atom numbers as keys and their coordinates as values (tuples).
    """
    coordinates = {}
    start_reading = False
    atom_index = 1  # Start counting from 1 since atom_numbers are 1-based index
    
    with open(filename, 'r') as file:
        for line in file:
            if 'POSITIONRED' in line:
                start_reading = True
                continue
            if 'END' in line and start_reading:
                break
            if start_reading:
                if atom_index in atom_numbers:
                    # Extract coordinates from the line, convert them to float, and store them
                    parts = line.strip().split()
                    coordinates[atom_index] = tuple(float(part) for part in parts)
                atom_index += 1

    return coordinates


def write_gromacs_mdp_minimization(cwd, parameters):
    mdp_file_content = textwrap.dedent(
        f""";--- RUN CONTROL
            integrator               = steep    ; Algorithm (steep = steepest descent minimization)
            emtol                    = 1000.0   ; Stop minimization when the maximum force < 1000.0 kJ/mol/nm
            emstep                   = 0.01     ; Minimization step size
            nsteps                   = 50000    ; Maximum number of (minimization) steps to perform
            ;--- OUTPUT CONTROL
            ;nstvout                 = 10000
            ;nstenergy               = 10000
            ;nstcalcenergy           = 1
            ;nstlog                  = 10000
            ;nstcomm                 = 1
            ;NEIGHBOR SEARCHING
            cutoff-scheme           = verlet
            pbc                     = xyz
            ;--- ELECTROSTATICS
            coulombtype             = PME
            rcoulomb                = 1.1 ; uncomment if you're not going to use mdrun -tunepme
            ;--- VAN DER WAALS
            vdwtype                 = cut-off
            rvdw                    = 1.1
            vdw-modifier            = potential-switch
            rvdw-switch             = 1.05
            ;--- BONDS
            ;constraints             = h-bonds
            ;constraint-algorithm    = lincs
        """)
    
    with open(f'{cwd}/minimize.mdp', 'w') as f:
        f.write(mdp_file_content)

    for parameter in parameters:
        parameter_id = parameter.replace(' ', '_')  # Convert parameter into a valid directory/filename
        link_path = os.path.join(cwd, parameter_id, 'resample.mdp')  # Define full path for the new symbolic link

        # Check if the link already exists, if so, remove it first
        if os.path.islink(link_path):
            os.unlink(link_path)
        
        # Create a symbolic link pointing to 'target' with the name 'link_name' in the directory 'parameter_id'
        os.symlink(f'{cwd}/minimize.mdp', link_path)

#################
#--- ABF RUN ---#
#################
def run_gromacs_abf(parameters, topology, threads, total_cores, scan_steps, input_file, range):
    cwd = os.getcwd()

    positions = read_xyz_positions(input_file)
    write_g96(positions[0][2:], os.path.join(cwd, 'positions.g96'), mode='w')

    def run_job(parameter_id, cwd, topology, threads):
        path = os.path.join(cwd, parameter_id)
        grompp(parameter_id, f'{cwd}/{topology}', path) # param_id, topology file path, simulation path
        mdrun(parameter_id, path, threads, rerun=False)
        os.system(f'mv {cwd}/{parameter_id}/{parameter_id}.czar.pmf {cwd}/{parameter_id}/energy.xvg')
        

    jobs = []

    for i, parameter in enumerate(parameters):
        parameter_id = parameter.replace(' ', '_')
        write_colvars_file(parameters, parameter_id, scan_steps, cwd, range)
        write_gromacs_abf_mdp(parameter_id, cwd)
        # Copy the positions.g96 file to the parameter directory
        os.system(f'cp {cwd}/positions.g96 {cwd}/{parameter_id}/{parameter_id}.g96')
        # Copy the topology file to the parameter directory
        os.system(f'cp {topology} {cwd}/{parameter_id}/')
        #extensions = ','.join(['tpr', 'err', 'out', 'zcount', 'trr', 'pmf', 'grad', 'edr', 'czar.pmf', 'czar.grad', 'count', 'state', 'log', 'traj', 'xvg'])
        #print(extensions)
        #os.system(f"rm {os.path.join(cwd, parameter_id)}/*.{{{extensions}}}")
        job = jd.Job(name=f"{parameter_id}", function=run_job, arguments = [parameter_id, cwd, topology, threads], cores=threads)
        jobs.append(job)
        
        # FOR TESTING PURPOSES ONLY
        # if i > 2:
        #     break   


    dispatcher = jd.JobDispatcher(jobs, maxcores=total_cores, engine="multiprocessing")
    results = dispatcher.run()
    


def write_colvars_file(parameters, parameter_id, scan_steps, cwd, range):
    parameter_split = parameter_id.split("_")
    if len(parameter_split) == 3:
        parameter_type = 'distance'
        a, b = parameter_split[1:]
        string_a = f"group1 {{ atomNumbers {a} }}"
        string_b = f"group2 {{ atomNumbers {b} }}"
        string_c = ""
    else:
        parameter_type = 'angle'
        a, b, c = parameter_split[1:]
        string_a = f"group1 {{ atomNumbers {a} }}"
        string_b = f"group2 {{ atomNumbers {b} }}"
        string_c = f"group3 {{ atomNumbers {c} }}"

    eq_value = float(parameters[parameter_id.replace("_", " ")][0][0]) # what a mess

    if parameter_type == 'angle':
        unit = 'º'
    else:
        unit = 'nm'
    
    if range == 'ff':
        start = eq_value - eq_value * 0.1
        end = eq_value + eq_value * 0.1
        steps = scan_steps
        width = (end - start) / (steps - 1)
        end -= width

        print(f'Scanning parameter {parameter_id} from {start} {unit} to {end} {unit} (FF range)')
        
    if range == 'qm':
        positions = []
        try:
            print(f'Attempting to extract range from file {parameter_id}.relaxscanact.dat')
            with open (f'{cwd}/{parameter_id}/{parameter_id}.relaxscanact.dat') as file:
                for line in file:
                    positions.append(float(line.split()[0]) / 10 if parameter_type == "distance" else float(line.split()[0]))
        except Exception:
            #TODO: FIX BARE EXCEPTION ASAP
            print("fail")

        start = positions[0]
        end = positions[-1]

        print(f'Scanning parameter {parameter_id} from {start} {unit} to {end} {unit} (QM range)')

        width = (end - start) / (scan_steps -1)
        #start -= width/2
        end -= width   
        
    
    colvars_file_content = textwrap.dedent(
        f"""colvar {{
  name phi
  width {width}
  lowerBoundary  {start}
  upperBoundary  {end+width}
  # {parameter_type} definition
  {parameter_type} {{
    {string_a}
    {string_b}
    {string_c}
  }}
  extendedLagrangian on
  extendedFluctuation 3.0
}}
abf {{
 colvars phi
 fullSamples 200
}}""")
    # write colvars file to disk
    with open(f'{cwd}/{parameter_id}/colvars.dat', 'w') as f:
        f.write(colvars_file_content)

def write_gromacs_abf_mdp(parameter_id, cwd):
    mdp_file_content = textwrap.dedent(
        f"""; MDP file

; Run parameters
integrator              = md-vv                ; leap-frog integrator
nsteps                  = 4000000          ; 4'000'000 = 2 ns
dt                      = 0.002            ; in ps (0.001 = 1 fs)
nstcomm                 = 100                 ; freq. for cm-motion removal
ld_seed                 = -1

; Bond constraints
continuation            = no                ; continue from npt equilibration
constraints             = none              ; constrain hydrogen bond lengths
constraint_algorithm    = lincs             ; default
lincs_order             = 4                 ; default

; X/V/F/E outputs
nstxout                 = 50000            ; pos out   ---  1000  ps
nstvout                 = 50000            ; vel out   ---  1000  ps
nstfout                 = 0                 ; force out ---  no
nstlog                  = 50000             ; energies to log (20 ps)
nstenergy               = 50000               ; energies to energy file
nstxout-compressed      = 500               ; xtc, 1 ps
compressed-x-precision  = 100000

; Neighbour list
cutoff-scheme           = Verlet            ;
ns_type                 = grid              ; neighlist type
nstlist                 = 20                ; Freq. to update neighbour list
rlist                   = 1.2               ; nm (cutoff for short-range NL)

; Coulomb interactions
coulombtype             = Cut-off
rcoulomb                = 1.1                ; nm (direct space sum cut-off)
;optimize_fft            = yes               ; optimal FFT plan for the grid

; van der Waals interactions
vdwtype                 = Cut-off           ; Van der Waals interactions
rvdw                    = 1.1               ; nm (LJ cut-off)
vdw-modifier            = potential-switch
rvdw-switch             = 1.05
DispCorr                = No                ; use dispersion correction

; Energy monitoring
energygrps              = System

; Temperature coupling is on
tcoupl                  = andersen-massive         ; modified Berendsen thermostat
tc-grps                 = system        ; two coupling groups - more accurate
tau_t                   = 1.0        ; time constant, in ps
ref_t                   = 200        ; reference temperature, one for each group, in K

; Periodic boundary conditions
pbc                     = xyz               ; 3-D PBC
; Velocity generation
gen_vel                 = yes
; COLVARS
colvars-active = true
colvars-configfile = colvars.dat""")
    
    # write mdp file to disk
    with open(f'{cwd}/{parameter_id}/resample.mdp', 'w') as f:
        f.write(mdp_file_content)
def process_data_and_plot(parameter_id, parameter_value, cwd, fix_x0=False):
    """Processes and plots energy data from two files."""
    parameter_id = parameter_id.replace(' ', '_')
    energy_path = f"{cwd}/{parameter_id}/energy.xvg"
    qm_path = f"{cwd}/{parameter_id}/{parameter_id}.relaxscanact.dat"

    # Read and convert data
    mm_data = read_energy_data(energy_path)
    qm_data = read_energy_data(qm_path)
    mm_data[:, 1] = convert_energy(mm_data[:, 1], 'kJ/mol', 'kcal/mol')
    qm_data[:, 1] = convert_energy(qm_data[:, 1], 'Hartree', 'kcal/mol')

    # Normalize data
    mm_data = normalize_data(mm_data)
    qm_data = normalize_data(qm_data)

    # For bonds, convert to angstroms
    if parameter_id.startswith('b'):
        mm_data[:, 0] *= 10  # Convert from nm to Angstrom

    def harmonic_spring(x, k, x0):
        """Harmonic spring model function."""
        return 0.5 * k * (x - x0)**2
    
    print(f'Generating plot for {parameter_id}')
    def try_curve_fit(func, xdata, ydata, p0, initial_maxfev=600, retries=3, bounds=True):
        if bounds:
            # Extract the minimum and maximum values from mm_data[:, 0] and qm_data[:, 0]
            x_min, x_max = xdata.min(), xdata.max()
            if p0[1] > x_max or p0[1] < x_min:
                p0[1] = (x_min + x_max) / 2  # Set x0 to the average of the min and max values

        for i in range(retries):
            try:
                if bounds:
                    return curve_fit(func, xdata, ydata, p0=p0, maxfev=initial_maxfev, bounds=([0, x_min], [np.inf, x_max]))
                else:
                    return curve_fit(func, xdata, ydata, p0=p0, maxfev=initial_maxfev)
            except RuntimeError as e:
                if i < retries - 1:
                    initial_maxfev *= 2  # Double maxfev for the next attempt
                else:
                    raise RuntimeError(f"Curve fit failed after {retries} attempts: {e}")
    
    eq_value = float(parameter_value[0])
    
    xlabel = "Bond length (Å)" if parameter_id.startswith('b') else "Angle (°)"
    if parameter_id.startswith('b'): eq_value *= 10  # Convert from nm to Angstrom
    
    harmonic_spring_fixed = partial(harmonic_spring, x0=eq_value)



    if fix_x0:
        harmonic_spring_fixed = partial(harmonic_spring, x0=eq_value)
        mm_opt, mm_cov = try_curve_fit(lambda x, k: harmonic_spring_fixed(x, k), mm_data[:, 0], mm_data[:, 1], p0=[1.0])
        qm_opt, qm_cov = try_curve_fit(lambda x, k: harmonic_spring_fixed(x, k), qm_data[:, 0], qm_data[:, 1], p0=[1.0])
        mm_k = mm_opt[0]
        qm_k = qm_opt[0]
        mm_x0 = eq_value
        qm_x0 = eq_value
    else:
        def harmonic_spring_with_x0(x, k, x0): return harmonic_spring(x, k, x0)
        mm_opt, mm_cov = try_curve_fit(harmonic_spring_with_x0, mm_data[:, 0], mm_data[:, 1], p0=[1.0, eq_value], bounds=True)
        qm_opt, qm_cov = try_curve_fit(harmonic_spring_with_x0, qm_data[:, 0], qm_data[:, 1], p0=[1.0, eq_value], bounds=True)
        mm_k = mm_opt[0]
        qm_k = qm_opt[0]
        mm_x0 = mm_opt[1]
        qm_x0 = qm_opt[1]

    # Generate fitted data points for plotting
    x_fitted = np.linspace(min(qm_data[:, 0]), max(qm_data[:, 0]), 300)
    
    if fix_x0:
        mm_fitted = harmonic_spring(x_fitted, mm_k, eq_value)
        qm_fitted = harmonic_spring(x_fitted, qm_k, eq_value)
    else:
        mm_fitted = harmonic_spring(x_fitted, mm_k, mm_x0)
        qm_fitted = harmonic_spring(x_fitted, qm_k, qm_x0)

    np.savetxt(f"{cwd}/{parameter_id}_mm_data.csv", np.column_stack((mm_data[:, 0], mm_data[:, 1])), delimiter=',', header='Bond Length,Energy', comments=f'{mm_k}, {mm_x0}')
    np.savetxt(f"{cwd}/{parameter_id}_qm_data.csv", qm_data, delimiter=',', header='Bond Length,Energy', comments=f'{qm_k}, {qm_x0}')

    if parameter_id.startswith('b'):
        mm_x0_string = f', x0={mm_x0:.2f} Å'
        qm_x0_string = f', x0={qm_x0:.2f} Å'
    else:
        mm_x0_string = f', x0={mm_x0:.2f}°'
        qm_x0_string = f', x0={qm_x0:.2f}°'
    
    hide_eq_value = True
    if hide_eq_value:
        mm_x0_string = ""
        qm_x0_string = ""

    # Plot data
    plt.figure(figsize=(10, 6))
    plt.plot(mm_data[:, 0], mm_data[:, 1], 'o-', label=f'MM (k={mm_k:.2f} kcal/mol/Å²){mm_x0_string}', markersize=10, linewidth=3)
    plt.plot(qm_data[:, 0], qm_data[:, 1], 's-', label=f'QM (k={qm_k:.2f} kcal/mol/Å²){qm_x0_string}', markersize=10, linewidth=3)

    plot_fit = False
    if plot_fit:
        plt.plot(x_fitted, mm_fitted, 'b--', label='Fitted MM')
        plt.plot(x_fitted, qm_fitted, 'r--', label='Fitted QM')
    
    # Increase font size for specific labels
    plt.title(f'{parameter_id}', fontsize=16)  # Title font size
    plt.xlabel(xlabel, fontsize=20)  # X-axis label font size
    plt.ylabel('Energy (kcal/mol)', fontsize=20)  # Y-axis label font size
    plt.tick_params(axis='both', which='major', labelsize=18)  # Font size for major ticks
    plt.grid(False)
    plt.legend(fontsize=14)  # Legend font size
    plt.tight_layout()  # Automatically adjusts layout to reduce padding
    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.12, right=0.88)  # Adjust layout for better spacing

    plt.savefig(f"{cwd}/{parameter_id}_energy_plot.png")

    plt.close()

    # Extracting energy values from the data arrays
    mm_energies = mm_data[:, 1]
    qm_energies = qm_data[:, 1]

    # If the x values in mm_data and qm_data are different, then only return mm_k, qm_k, and skip the various errors evaluations
    if not np.allclose(mm_data[:, 0], qm_data[:, 0]):
        return {'mm_k': mm_k, 'qm_k': qm_k}

    cumulative_error = (mm_energies - qm_energies).sum()

    # Calculate MAE
    mae = np.mean(np.abs(mm_energies - qm_energies))

    # Calculate RMSE
    rmse = np.sqrt(np.mean((mm_energies - qm_energies)**2))

    # Calculate R^2
    ss_res = np.sum((mm_energies - qm_energies)**2)
    ss_tot = np.sum((qm_energies - np.mean(qm_energies))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return {'mm_k': mm_k, 'qm_k': qm_k, 'mae': mae, 'rmse': rmse, 'r_squared': r_squared, 'cumulative': cumulative_error} 


def read_energy_data(file_path):
    """Reads energy data from a file, skipping headers."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith(('#', '@')):
                parts = line.split()
                data.append([float(parts[0]), float(parts[1])])
    return np.array(data)

def convert_energy(values, from_unit, to_unit):
    """Converts energy from one unit to another."""
    if from_unit == 'kJ/mol' and to_unit == 'kcal/mol':
        return values * 0.239005736  # conversion factor
    elif from_unit == 'Hartree' and to_unit == 'kcal/mol':
        return values * 627.509  # conversion factor
    return values

def normalize_data(data):
    """Normalizes data by subtracting the minimum value."""
    min_value = np.min(data[:, 1])
    data[:, 1] -= min_value
    return data



def calculate_distance(p1, p2):
    """Calculate the Euclidean distance between two points in 3D."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scans a set of parameters and creates a folder for each parameter.')
    parser.add_argument('-i', '--input-file', type=str, help='Structure from which to start the calculations from. XYZ format.')
    parser.add_argument('-t', '--topology', type=str, help='Path to the input file containing the parameters to scan. Should be a self-standing TOP file with no includes.')
    parser.add_argument('-tpc', '--threads-per-calc', type=int, help='Number of threads to use for each calculation.')
    parser.add_argument('--total-cores', type=int, help='Number of cores to use for the calculations.')
    parser.add_argument('--scan-steps', type=int, default=8, help='Number of steps to scan (default: 8).')
    parser.add_argument('--abf', action='store_true', help='Use ABF simulations')
    parser.add_argument('--range', type=str, default='qm', help='Use "ff" to scan around the minimum value in the force field, "qm" to use the "qm" range.')
    parser.add_argument('--fixed-fit-eq-value', action='store_true', help='Use the fixed fit equilibrium value.')
    parser.add_argument('--skip-orca-calc', action='store_true', help='Skip the ORCA calculation.')
    parser.add_argument('--skip-gromacs-calc', action='store_true', help='Skip the GROMACS calculation.')
    parser.add_argument('--include-hydrogens', action='store_true', help='Include hydrogens in the calculation.')
    parser.add_argument('--constrained-opt', action='store_true', help='Perform a constrained optimization.')
    parser.add_argument('--charge', type=int, help='Charge of the simulated specie.')
    parser.add_argument('--multiplicity', type=int, help='Multiplicity of the simulate specie.')
    args = parser.parse_args()
    scan_workflow(args)
