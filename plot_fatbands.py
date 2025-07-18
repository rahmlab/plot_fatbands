import numpy as np
import math
import pymatgen
import sys
import argparse
import logging
import os.path

from pymatgen.io.vasp.outputs import Procar, Vasprun
from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.core import Structure
from pymatgen.electronic_structure.core import Spin, Orbital

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

class SaneFormatter(argparse.RawTextHelpFormatter, 
                    argparse.ArgumentDefaultsHelpFormatter):
    pass

parser = argparse.ArgumentParser(prog='plot_fatbands.py',description=("Plot projected band structure (fatbands) from a VASP calculation.\n" +
									'Author: Marco Cappelletti. Inspired by Kevin Waters (kwaters4.github.io) and sumo-bandplot.\n' +
                                    'By default it assumes that:\n'  +
									'\tthe current directory contains KPOINTS and vasprun.xml from the band structure calculation\n' +
                                    '\tthe directory ../dos contains vasprun.xml from the DOS calculation\n' +
                                    '\tthe parent directory (../) contains the POSCAR file\n' +
                                    '\tthe grandparent directory (../../) contains the POTCAR file\n')
									,formatter_class=SaneFormatter)
parser.add_argument('-B','--vasprun-file-bands', type=str, help='Path of the vasprun.xml file of the band calculation', default='vasprun.xml')
parser.add_argument('-K','--KPOINTS-file', type=str, help='Path of the KPOINTS file with the band path', default='KPOINTS')
parser.add_argument('-C','--POSCAR-file', type=str, help='Path of the POSCAR file', default='../POSCAR')
parser.add_argument('-O','--PROCAR-file', type=str, help='Path of the PROCAR file from the band calculation', default='PROCAR')
parser.add_argument('-P','--POTCAR-file', type=str, help='Path of the POTCAR file', default='../../POTCAR')
parser.add_argument('-D','--vasprun-file-dos', type=str, help='Path of the vasprun.xml file of the dos calculation', default='../dos/vasprun.xml')
parser.add_argument('-p','--project', help='Band projection to rgb. Either 2 (red and green) or 3 arguments (red, green, blue). Nomenclature:\n'
													'\t- E/n: all orbitals of element with symbol E (H, C, N, ...) or atom index n (1, 2, ...)\n'
                                                    '\t- E/n.o: o-orbital of element with symbol E or atom index n (s, px, py, pz, dxy, ...)\n'
                                                    '\t- E/n.s.pz: s+pz orbitals of element with symbol E or atom index n\n'
                                                    '\t- X.s: s orbitals of all elements (literally X, see example)\n'
                                                    '\t- O.s.pz+N.pz: sum of O(s,pz) and N(pz)\n'
                                                    'You are free to mix anything you want. There is no restriction.\n'
                                                    'Examples (a number represents an atom index, starting from 1, as in PROCAR):\n'
                                                    '\t-p N O.s.pz+H.s C+X.pz =>'
                                                    ' red: N(all orb.), green: O(s+pz)+H(s), blue: C(all orb.)+(all atoms pz)\n'
                                                    '\t-p 1+2+3 4.px.py.pz+5 H.s  =>'
                                                    ' red: 1(all orb.)+2(all orb.)+3(all orb.), green: 4(px+py+pz)+5+(px), blue: H(s)\n'
                                                    , nargs='+', default=['X.px', 'X.py', 'X.s.pz'])
parser.add_argument('-t','--projection-type', type=str, help=f'Type of projection.\n'
                                                    '\t- blend: plot colored circles based on the (r,g,b) projection.\n'
                                                    '\t- stack: plot stacked red, green, and blue circles, with size depending on projection.\n', 
                                                    choices=['blend','stack'], default='stack')
parser.add_argument('--split', help='Split projections in different plots.', action='store_true')
parser.add_argument('-n','--normalization', type=str, help=f'Normalization of the projection.\n'
													'\t- all: with respect to all contributions.\n'
													'\t- selection: with respect to selection only.\n', choices=['all','selection'], default='selection')
parser.add_argument('-l','--max-l',type=int,default=1,choices=[1,2,3],help='Maximum value of l (angular momemtum) for the projection. Increases computational costs, so increase it only if necessary.')
parser.add_argument('-N','--no-projection',help='Do not perform any projection',action='store_true')
parser.add_argument('-m','--emin', type=float, help='Minimum of energy in the plot. If none, it chooses the lower limit', default=None)
parser.add_argument('-M','--emax', type=float, help='Maximum of energy in the plot. If none, it chooses the upper limit', default=None)
parser.add_argument('-s','--scale', type=float, help='DOS scale factor', default=1.0)
parser.add_argument('-H','--height', type=float, help='Height of the plot in inches', default=3.5)
parser.add_argument('-W','--width', type=float, help='Width of the plot in inches', default=3.3)
parser.add_argument('-r','--ratio', type=float, help='Bandplot - dosplot width ratio', default=3.0)
parser.add_argument('--blw','--band-lw', type=float, help='Linewidth of bands, when projection is off.', default=2.0)
parser.add_argument('--bs','--band-size', type=float, help='Circle size of bands, when projection is on.', default=6.0)
parser.add_argument('--ba','--band-alpha', type=float, help='alpha value (transparency) of bands', default=0.9)
parser.add_argument('--dlw','--dos-lw', type=float, help='Linewidth of DOS', default=1.0)
parser.add_argument('--vlw','--lines-lw', type=float, help='Linewidth of vertical lines. Set it to 0 to remove them', default=1.0)
parser.add_argument('--flw','--Fermi-lw', type=float, help='Linewidth of Fermi level. Set it to 0 to remove it', default=1.0)
parser.add_argument('--glw','--grid-lw', type=float, help='Linewidth of grid. Set it to 0 to remove them', default=1.0)
parser.add_argument('-f','--font-size', type=float, help='Fontsize', default=7)
parser.add_argument('-o','--output-file', type=str, help='Path and name of the output file, excluding the format', default='fatbands')
parser.add_argument('--format', type=str, help='Output file format', choices=['pdf','png'], default='pdf')
parser.add_argument('--redo','--readlog',  help='Rerun the last command, if plot_fatbands.log file is present. This overrides every other argument!', action='store_true')

args = parser.parse_args()
redo = args.redo

if redo is True:
    if os.path.isfile('plot_fatbands.log') is False:
        raise ValueError('I cannot rerun: plot_fatbands.log does not exist')

    with open('plot_fatbands.log') as f:
        argstr = f.readline()

    args = parser.parse_args(argstr.split()[1:])   


logging.basicConfig(
    filename="plot_fatbands.log",
    level=logging.INFO,
    filemode="w",
    format="%(message)s",
)

if redo is True:
    logging.info(argstr)
else:
    logging.info(" ".join(sys.argv[:]))

scale=args.scale
no_proj=args.no_projection



def CheckInput():
    # accepts only 1-3 entries for the projection
    if len(args.project) < 1 and len(args.project) > 3:
        raise ValueError('Only 1, 2 or 3 components allowed for the projection')
    elif len(args.project) == 1 and args.normalization == 'selection':
        print('\tWARNING: you selected 1 projection and \'selection\' normalization. This does not make sense. Normalization is changed to \'all\'')
        logging.info('WARNING: normalization is changed to \'all\'')
        args.normalization='all'


    if args.split is True and no_proj is True:
        raise ValueError('Cannot split non-projected bands!')

def rgbcircles(ax, KPOINTS, energies, rgb):

    if args.projection_type == 'blend':
        ax.scatter(KPOINTS,energies,s=args.bs,color=list(zip(rgb[0],rgb[1],rgb[2])),alpha=args.ba)

    elif args.projection_type == 'stack':
        ax.plot(KPOINTS,energies,lw=1.0,color='k',alpha=0.5)
        sizes=np.array(list(zip(rgb[0],rgb[1],rgb[2])),dtype=float)
        ax.scatter(KPOINTS,energies,s=args.bs*(sizes[:,0]+sizes[:,1]+sizes[:,2]),color=(0,0,1),edgecolor='none',alpha=args.ba)
        ax.scatter(KPOINTS,energies,s=args.bs*(sizes[:,0]+sizes[:,1]),color=(0,1,0),edgecolor='none',alpha=args.ba)
        ax.scatter(KPOINTS,energies,s=args.bs*sizes[:,0],color=(1,0,0),edgecolor='none',alpha=args.ba)



def CalculateProjectionsAndPlot():

    # calculates contributions for bands and DOS projections
    el_orbs = []
    el_orbs_labels = []

    # read the projections
    # 1-3 components, e.g. 'N.s.pz', 'N.s+O.s.pz', 'N+O.s', 'N+O', 'X', 'X.s', 'X.s+N.px.py.pz', ...
    for component in args.project:
        element_components = component.split("+")   # split elements, e.g. 'N.s+O.s.pz' becomes ['N.s', 'O.s.pz']
        color_component = []			# list of components for one color
        label_text = []                 # more readable, user-friendly text for the legend
        for element_component in element_components:  # e.g. for element in the list ['N.s', 'O.s.pz'] (example above)
            splits = element_component.split(".")   # e.g. 'N.s.pz' converted to ['N', 's', 'pz']
            element = splits[0]                     # first element is the atom symbol, e.g. 'N'
            if len(splits) == 1:                    # if splits has one element, then no orbitals are specified (e.g. 'N') => plot all orbitals (= 'all')
                orbitals = 'all'
            else:
                orbitals = splits[1:]               # e.g. ['s', 'pz'] 
            color_component.append([element,orbitals])	

            if element == 'X':                      # 'X' represents all atoms
                if len(splits) == 1:                # if input is only 'X'
                    label_text.append('s+p_x+p_y+p_z')      # this makes no sense to choose, it's the projection of all atoms and orbitals, but who am I to judge
                else:
                    label_text.append("+".join(orbitals).replace('p','p_'))   # more readable, e.g. 'X.s.pz' becomes 's+p_z'
            else:
                if len(splits) == 1:                        # all orbitals, only the element: 'N'; or the index: '2'
                    if element.isnumeric() is True:
                        label_text.append(atom_labels[int(element)-1]+element)
                    else:
                        label_text.append(element)
                else:
                    if element.isnumeric() is True:
                        label_text.append(f"{atom_labels[int(element)-1]+element}({','.join(orbitals).replace('p','p_')})")
                    else:
                        label_text.append(f"{element}({','.join(orbitals).replace('p','p_')})")   # e.g. 'O.s.pz' becomes 'O(s+p_z)'
        el_orbs.append(color_component)
        el_orbs_labels.append("+".join(label_text))         # e.g. 'O.s.pz+H.s' becomes 'O(s+p_z)+H(s)'

    
    color_values = { 0: 'red', 1: 'green', 2: 'blue' }
    print('\tProjections:')
    for color_idx, color_contrib in enumerate(el_orbs_labels):
        print(f'\t    - {color_values[color_idx]}:\t{color_contrib}')

    print(f'\tNormalization mode is: \'{args.normalization}\'')
    print(f'\tProjection type is: \'{args.projection_type}\'')


    logging.info(f'Calculating contributions for orbitals with angular momentum up to l = {args.max_l}. Ignoring the rest if l < 3')
    if args.normalization == 'selection':
        logging.info(f'Normalization mode is {args.normalization}: selected contributions sum to 1. Other contributions (if any) will not be visible in the plot!')
    else:
        logging.info(f'Normalization mode is {args.normalization}: selected contributions are normalized with respect to all contributions.')


    logging.info(f'\nContribution table (only plotted bands) ([red,green,blue] format)')

    if len(el_orbs_labels) == 1:
        logging.info(f'Color is red = {el_orbs_labels[0]}')
    elif len(el_orbs_labels) == 2:
        logging.info(f'Colors are: red = {el_orbs_labels[0]}, green = {el_orbs_labels[1]}')
    elif len(el_orbs_labels) == 3:
        logging.info(f'Colors are: red = {el_orbs_labels[0]}, green = {el_orbs_labels[1]}, blue = {el_orbs_labels[2]}')

    logging.info(f'band' + '\t\t\t\t' + '\t\t\t\t\t'.join(labels))


    # as in VASP
    orbital_values = { 's': 0,
                       'py': 1, 'pz': 2, 'px': 3,
                       'dxy': 4, 'dyz': 5, 'dz2': 6, 'dxz': 7, 'dx2_y2': 8,
                       'f_3' : 9, 'f_2' : 10, 'f_1' : 11, 'f0' : 12, 'f1' : 13, 'f2' : 14, 'f3' : 15 }

    max_l_index = (args.max_l + 1)**2   # 1 -> 4, 2 -> 9, 3 -> 16. So to have range(0,max_l_index) = [0,1,...,max_l_index-1]

    # contributions to the band per each band, k-point, and color: contrib_bands[band][k-point][color]
    contrib_bands = np.zeros((bands.nb_bands, len(bands.kpoints), 3))

    # contributions to the DOS per each color: contrib_dos[color][energy]
    contrib_dos = np.zeros((len(el_orbs), len(dosrun.pdos[0][Orbital.s][Spin.up])))


    # obtain contributions
    # sum over all bands
    for b in range(min_band_to_plot,max_band_to_plot+1):
        # sum over all k-points
        for k in range(len(bands.kpoints)):
            for color_idx, color_contrib in enumerate(el_orbs): # color_idx: 1-3. color_contrib: list of [elements, orbitals] for each color
                for element_contrib in color_contrib:
                    element = element_contrib[0]        # e.g. 'X', 'N'
                    orbitals = element_contrib[1]       # e.g. 'all', 's', ['s', 'pz']
                    if element == 'X':                  # if all atoms, get all indexes
                        element_indexes = range(len(atom_labels))
                    elif element.isnumeric() is True:
                        element_indexes = [int(element)-1]
                    else:                               # else, get the indexes with label = atom symbol (e.g. 'C')
                        element_indexes = [i for i, x in enumerate(atom_labels) if x == element]

                    if orbitals == 'all':
                        orbital_indexes = range(0,max_l_index)  # sum all orbitals if 'all'
                    else:
                        orbital_indexes = [orbital_values[o] for o in orbitals]

                    for i in element_indexes:
                        for j in orbital_indexes:
                            contrib_bands[b,k,color_idx] += data[Spin.up][k][b][i][j]**2
                            if k == 0 and b == min_band_to_plot:	# needs to be done just once 
                                contrib_dos[color_idx] += np.array(dosrun.pdos[i][Orbital(j)][Spin.up])

            # normalization
            if args.normalization == 'selection':
                if np.sum(contrib_bands[b,k,:]) != 0:
                    contrib_bands[b,k,:] = contrib_bands[b,k,:]/np.sum(contrib_bands[b,k,:])
                elif k in labels_kpt_num+1:     # I don't know why, VASP prints all contribution to zero at those points
                    contrib_bands[b,k,:] = contrib_bands[b,k-1,:]

                # print contributions at high-symmetry point to log file
            elif args.normalization == 'all':
                tot = 0.0
                for i in range(len(atom_labels)):
                    for j in range(0,max_l_index):
                        tot += data[Spin.up][k][b][i][j]**2
                if tot != 0:
                    contrib_bands[b,k,:] = contrib_bands[b,k,:]/tot
                #else:
                    #contrib_bands[b,k,:] = contrib_bands[b,k-1,:]
        if b < 9:
            b_str = '  ' + str(b+1)
        elif b < 99:
            b_str = ' ' + str(b+1)
        else:
            b_str = str(b+1)
        logging.info(f'{b_str}: ' + '\t'
                     + '\t'.join('[' + ', '.join(f'{x:.2f}' for x in contrib_bands[b, lkn, :])
                     + ']' for lkn in labels_kpt_num))


    # Plotting -----------------------------------------------------------------

    if args.split is False:
        for b in range(min_band_to_plot,max_band_to_plot+1):
            rgbcircles(ax_bands,
                KPOINTS,
                [e - bands.efermi for e in bands.bands[Spin.up][b]],
                [contrib_bands[b,:,0], contrib_bands[b,:,1],contrib_bands[b,:,2]])  #third element is np.zeros if only two colors are defined
    else:
        zero_projections = np.zeros(len(bands.kpoints))
        for proj_n in range(len(args.project)):
            ax = ax_bands[proj_n]
            for b in range(min_band_to_plot,max_band_to_plot+1):
                if proj_n == 0:
                    rgb = [ contrib_bands[b,:,0], zero_projections, zero_projections ]
                elif proj_n == 1:
                    rgb = [ zero_projections, contrib_bands[b,:,1], zero_projections ]
                elif proj_n == 2:
                    rgb = [ zero_projections, zero_projections, contrib_bands[b,:,2] ]

                rgbcircles(ax,
                    KPOINTS,
                    [e - bands.efermi for e in bands.bands[Spin.up][b]],
                    rgb)

    # plot DOS
    ax_DOS.plot(contrib_dos[0],dosrun.tdos.energies - dosrun.efermi, \
            c=(1.0,0.0,0.0), label = f'${el_orbs_labels[0]}$', linewidth = args.dlw)
    if len(el_orbs) >= 2:
        ax_DOS.plot(contrib_dos[1],dosrun.tdos.energies - dosrun.efermi, \
            c=(0.0,1.0,0.0), label = f'${el_orbs_labels[1]}$', linewidth = args.dlw)
    if len(el_orbs) == 3:
        ax_DOS.plot(contrib_dos[2],dosrun.tdos.energies - dosrun.efermi, \
            c=(0.0,0.0,1.0), label = f'${el_orbs_labels[2]}$', linewidth = 1)

    # --------------------------------------------------------------------------










#----- Program starts -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print('--- plot_fatbands.py --------------------------------------')
    print(f'\tPlotting fatbands into {args.output_file}.{args.format}')
    if redo is True:
        print(f'\tRedoing previous run: {argstr}',end='')
    print(f'\tAdditional data is printed into plot_fatbands.log')

    CheckInput()


    # Read files ---------------------------------------------------------------

    # Load Structure
    structure = Structure.from_file(args.POSCAR_file)
    atom_labels = structure.labels
    # Load Band Structure Calculations
    bands = Vasprun(args.vasprun_file_bands,parse_potcar_file=args.POTCAR_file).get_band_structure(args.KPOINTS_file, line_mode = True)
    # Read KPOINTS file with path
    kpts = Kpoints.from_file(args.KPOINTS_file)  

    if no_proj is False:
        # projected bands
        data = Procar(args.PROCAR_file).data

    # density of states
    dosrun = Vasprun(args.vasprun_file_dos,parse_potcar_file=args.POTCAR_file)

    # --------------------------------------------------------------------------


    # Interpret k-points -------------------------------------------------------

    # k-point labels
    n_labels = len(kpts.labels)
    labels = []
    labels.append(kpts.labels[0])
    for label_idx in range(1,n_labels,2):
        labels.append(kpts.labels[label_idx])

    # get kpoint number for each high symmetry point
    labels_kpt_num = np.zeros(len(labels), dtype=int)
    lab_idx = 0
    for idx, bkpt in enumerate(bands.kpoints):
        if bkpt.label is not None and idx-1 not in labels_kpt_num: # usually they are duplicated, I just take the first
            labels_kpt_num[lab_idx] = idx
            lab_idx += 1

    # Number of points between kpoints, found in the KPOINTS file
    step = kpts.num_kpts

    # --------------------------------------------------------------------------


    # Initialize plot ----------------------------------------------------------

    # general options for plot
    font = {'family': 'serif', 'size': args.font_size}
    plt.rc('font', **font)

    # if no split is requested: one band plot and one DOS plot
    if args.split is False:
        # set up 2 graph with aspec ratio args.ratio/1
        # plot 1: bands diagram
        # plot 2: DOS
        gs = GridSpec(1, 2, width_ratios=[args.ratio,1], wspace=0.1)
        fig = plt.figure(figsize=(args.width, args.height))
        ax_bands = plt.subplot(gs[0])
        ax_DOS = plt.subplot(gs[1]) #, sharey=ax1)

    # if split is requested: N band plots (N=number of projections, 1-3) and one DOS plot
    else:
        print('\tWARNING: split is requested, remember to change plot width!')
        width_ratios = [ args.ratio/len(args.project) for proj_n in range(len(args.project)) ]
        width_ratios.append(1)
        gs = GridSpec(1, len(args.project)+1, width_ratios=width_ratios, wspace=0.1)
        fig = plt.figure(figsize=(args.width, args.height))
        ax_bands = [ plt.subplot(gs[proj_n]) for proj_n in range(len(args.project)) ]
        ax_DOS = plt.subplot(gs[len(args.project)]) #, sharey=ax1)
        
    # --------------------------------------------------------------------------

    # Set both fermi levels equal to the band fermi level
    bands.efermi =  dosrun.efermi #= 0

    # set y limits for the plot
    emin = args.emin
    emax = args.emax

    # if either is not defined, get the min and/or the max of energy from bands.bands.keys()
    if emin is None and emax is None:
        emin=1000
        emax=-1000
        for spin in bands.bands.keys():
            for b in range(bands.nb_bands):
                emin = min(emin, min(bands.bands[spin][b]))
                emax = max(emax, max(bands.bands[spin][b]))
        emin = emin - bands.efermi
        emax = emax - bands.efermi
    elif emin is None:
        emin=1000
        for spin in bands.bands.keys():
            for b in range(bands.nb_bands):
                emin = min(emin, min(bands.bands[spin][b]))
        emin = emin - bands.efermi
    elif emax is None:
        emax=-1000
        for spin in bands.bands.keys():
            for b in range(bands.nb_bands):
                emax = max(emax, max(bands.bands[spin][b]))
        emax = emax - bands.efermi

    # set first and last bands to plot, so that it calculates the projections only on those
    min_band_to_plot = 1000
    max_band_to_plot = -1

    for spin in bands.bands.keys():
        min_band_to_plot = min(min_band_to_plot, np.min(np.where(bands.bands[spin] >= emin + bands.efermi)[0]))
        max_band_to_plot = max(max_band_to_plot, np.max(np.where(bands.bands[spin] <= emax + bands.efermi)[0]))

    print(f'\tPlotting bands from {emin} to {emax} eV, band numbers from {min_band_to_plot+1} to {max_band_to_plot+1}')

    reciprocal = bands.lattice_rec.matrix/(2*math.pi)

    # Empty lists used for caculating the distances between K-Points
    # unchanged from Kevin Waters's script
    KPOINTS = [0.0]
    DIST = 0.0
    # Create list with distances between Kpoints (Individual), corrects the spacing
    for k in range(len(bands.kpoints)-1 ):
        Dist = np.subtract(bands.kpoints[k+1].frac_coords,bands.kpoints[k].frac_coords)
        DIST += np.linalg.norm(np.dot(reciprocal,Dist))
        KPOINTS.append(DIST)


    TICKS = [0.0]
    for i in range(step,len(KPOINTS)+step,step):
        TICKS.append(KPOINTS[i-1])
    # set y-axis limit
    if args.split is False:
        ax_bands.set_ylabel(r"$E - E_f$ (eV)",labelpad=-2)   #labelpad might work bad
        ax_bands.set_ylim(emin, emax)
        ax_bands.grid(lw=args.glw,alpha=0.5)
        ax_bands.hlines(y=0, xmin=0, xmax=len(bands.kpoints), color="k", lw=args.flw)
        for i in range(step,len(KPOINTS)+step,step):
            ax_bands.vlines(KPOINTS[i-1], emin, emax, "k",lw=args.vlw)
        ax_bands.set_xticks(TICKS)
        ax_bands.set_xticklabels(labels)
        ax_bands.tick_params(axis='x', which='both', length=0, pad=5)
        ax_bands.set_xlim(0, KPOINTS[-1])
    else:
        ax_bands[0].set_ylabel(r"$E - E_f$ (eV)",labelpad=-2)   #labelpad might work bad
        for axis in ax_bands:
            axis.set_ylim(emin,emax)
            axis.grid(lw=args.glw,alpha=0.5)
            axis.hlines(y=0, xmin=0, xmax=len(bands.kpoints), color="k", lw=args.flw)
            for i in range(step,len(KPOINTS)+step,step):
                axis.vlines(KPOINTS[i-1], emin, emax, "k",lw=args.vlw)
            axis.set_xticks(TICKS)
            axis.set_xticklabels(labels)
            axis.tick_params(axis='x', which='both', length=0, pad=5)
            axis.set_xlim(0, KPOINTS[-1])
            if axis != ax_bands[0]:
                axis.set_yticks([])
                axis.set_yticklabels([])

    ax_DOS.set_ylim(emin, emax)


    if no_proj is False:
        CalculateProjectionsAndPlot()
        DOSlabel='total'
    else:
        print('\tNo projection requested. Plotting normal bands.')
        for b in range(min_band_to_plot,max_band_to_plot+1): 
            ax_bands.plot(KPOINTS,[e - bands.efermi for e in bands.bands[Spin.up][b]], lw=args.blw, color='k')
        DOSlabel=None

    ax_DOS.fill_betweenx(dosrun.tdos.energies - dosrun.efermi,
        0,dosrun.tdos.densities[Spin.up],
        color = (0.7, 0.7, 0.7),
        facecolor = (0.7, 0.7, 0.7))
    ax_DOS.plot(dosrun.tdos.densities[Spin.up],
        dosrun.tdos.energies - dosrun.efermi,
        color = (0.6, 0.6, 0.6),
        label = DOSlabel, lw=args.dlw)

    if no_proj is False:
        ax_DOS.legend(fancybox=False, shadow=False, prop={'size': args.font_size-1},labelspacing=0.15,borderpad=0.20,handlelength=1.2,framealpha=0.6)

    # scaling factor for the x axis limit, if the peaks are too high
    maxdos = max(dosrun.tdos.densities[Spin.up])/scale

    
    ax_DOS.set_yticklabels([])
    ax_DOS.grid(lw=args.glw,alpha=0.5)
    ax_DOS.set_xticks([])
    ax_DOS.set_xlim(0,maxdos)
    ax_DOS.hlines(y=0, xmin=0, xmax=maxdos, color="k", lw=args.flw)
    ax_DOS.set_xlabel("DOS")

    # Plotting 
    # -----------------
    plt.savefig(f"{args.output_file}.{args.format}", format=args.format, bbox_inches='tight', dpi=400)


    print('\tFile saved.')
    print('-----------------------------------------------------------')
