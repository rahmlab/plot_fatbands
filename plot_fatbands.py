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
									'Author: Marco Cappelletti. Heavily inspired by Kevin Waters (kwaters4.github.io) and sumo-bandplot.\n' +
                                    'By default is assumes that:\n'  +
									'\tthe current directory contains KPOINTS, vasprun.xml from the band calculations\n' +
                                    '\tthe directory ../dos contains vasprun.xml from the dos calculation\n' +
                                    '\tthe parent directory (../) contains the POSCAR file\n' +
                                    '\tthe parent parent directory (../../) contains the POTCAR file\n')
									,formatter_class=SaneFormatter)
parser.add_argument('-B','--vasprun-file-bands', type=str, help='Path of the vasprun.xml file of the band calculation', default='vasprun.xml')
parser.add_argument('-K','--KPOINTS-file', type=str, help='Path of the KPOINTS file with the band path', default='KPOINTS')
parser.add_argument('-C','--POSCAR-file', type=str, help='Path of the POSCAR file', default='../POSCAR')
parser.add_argument('-O','--PROCAR-file', type=str, help='Path of the PROCAR file from the band calculation', default='PROCAR')
parser.add_argument('-P','--POTCAR-file', type=str, help='Path of the POTCAR file', default='../../POTCAR')
parser.add_argument('-D','--vasprun-file-dos', type=str, help='Path of the vasprun.xml file of the dos calculation', default='../dos/vasprun.xml')
parser.add_argument('-p','--project', help='Band projection to rgb. Either 2 (red and green) or 3 arguments (red, green, blue). Nomenclature:\n'
													'\t- E: all orbitals of element E (H, C, N, O, ...)\n'
                                                    '\t- E.o: o-orbital of element E (s, px, py, pz, dxy, ...)\n'
                                                    '\t- E.s.pz: s+pz orbitals of element E\n'
                                                    '\t- X.s: s orbitals of all elements\n'
                                                    '\t- O.s.pz+N.pz: sum of O(s,pz) and N(pz)\n', nargs='+', default=['X.px', 'X.py', 'X.s.pz'])
parser.add_argument('-n','--normalization', type=str, help=f'Normalization of the projection.\n'
													'\t-\'all\': with respect to all contributions.\n'
													'\t-\'selection\': with respect to selection only\n', choices=['all','selection'], default='selection')
parser.add_argument('-l','--max-l',type=int,default=1,choices=[1,2,3],help='Maximum value of l (angular momemtum) for the projection. Increases computational costs, so increase it only if necessary.')
parser.add_argument('-N','--no-projection',help='Do not perform any projection',action='store_true')
parser.add_argument('-m','--emin', type=float, help='Minimum of energy in the plot. If none, it chooses the lower limit', default=None)
parser.add_argument('-M','--emax', type=float, help='Maximum of energy in the plot. If none, it chooses the upper limit', default=None)
parser.add_argument('-s','--scale', type=float, help='DOS scale factor', default=1.0)
parser.add_argument('-H','--height', type=float, help='Height of the plot in inches', default=3.5)
parser.add_argument('-W','--width', type=float, help='Width of the plot in inches', default=3.3)
parser.add_argument('-r','--ratio', type=float, help='Bandplot - dosplot width ratio', default=3.0)
parser.add_argument('--plw','--plot-lw', type=float, help='Linewidth of bands and total DOS', default=2.0)
parser.add_argument('--vlw','--lines-lw', type=float, help='Linewidth of vertical lines. Set it to 0 to remove them', default=1.0)
parser.add_argument('--flw','--Fermi-lw', type=float, help='Linewidth of Fermi level. Set it to 0 to remove it', default=1.0)
parser.add_argument('--glw','--grid-lw', type=float, help='Linewidth of grid. Set it to 0 to remove them', default=1.0)
parser.add_argument('-f','--font-size', type=float, help='Fontsize', default=7)
parser.add_argument('-o','--output-file', type=str, help='Path and name of the output file, excluding the format', default='fatbands')
parser.add_argument('--format', type=str, help='Output file format', choices=['pdf','png'], default='pdf')
parser.add_argument('--redo','--readlog',  help='Rerun the last command, if plot_fatbands.log file is present. This overrides every other argument!', action='store_true')

args = parser.parse_args()

if args.redo is True:
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

if args.redo is True:
    logging.info(argstr)
else:
    logging.info(" ".join(sys.argv[:]))

scale=args.scale
no_proj=args.no_projection



# plot colored line. Function written by Kevin Waters
def rgbline(ax, KPOINTS, e, red, green, blue, alpha=1.):
    #creation of segments based on
    #http://nbviewer.ipython.org/urls/raw.github.com/dpsanders/matplotlib-examples/master/colorline.ipynb
    pts = np.array([KPOINTS, e]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    nseg = len(KPOINTS) -1
    a = np.ones(nseg, np.float64)*alpha
    lc = LineCollection(seg, colors=list(zip(red,green,blue,a)), linewidth = args.plw)
    ax.add_collection(lc)


def CalculateProjections():
    # accepts only 2 or 3 entries for the projection
    if no_proj is False and ( len(args.project) != 2 and len(args.project) != 3 ):
        raise ValueError('Either 2 or 3 components for the projection')

    # calculates contributions for bands and DOS projections
    el_orbs = []
    el_orbs_labels = []
    for component in args.project:      # either 2 or 3 components, e.g. 'N.s.pz', 'N.s+O.s.pz', 'N+O.s', 'N+O', 'X', 'X.s', 'X.s+N.px.py.pz', ...
        element_components = component.split("+")   # split elements, e.g. 'N.s+O.s.pz' becomes ['N.s', 'O.s.pz']
        color_component = []
        label_text = []
        for element_component in element_components:
            splits = element_component.split(".")   # e.g. 'N.s.pz' converted to ['N', 's', 'pz']
            element = splits[0]                     # first element is the atom symbol
            if len(splits) == 1:                    # if no orbital is specified (e.g. 'N'), then plot all orbitals (= 'all')
                orbitals = 'all'
            else:
                orbitals = splits[1:]               # e.g. ['s', 'pz'] 
            color_component.append([element,orbitals])

            # write text for legend
            if element == 'X':                      # 'X' represents all atoms
                if len(splits) == 1:
                    label_text.append('s+p_x+p_y+p_z')      # this makes no sense to choose, it's the projection of all atoms and orbitals, but who am I to judge
                else:
                    label_text.append("+".join(orbitals).replace('p','p_'))   # e.g. 'X.s.pz' becomes 's+p_z'
            else:
                if len(splits) == 1:                        # all orbitals, only the element: 'N'
                    label_text.append(element)
                else:
                    label_text.append(f"{element}({','.join(orbitals).replace('p','p_')})")   # e.g. 'O.s.pz' becomes 'O(s+p_z)'
        el_orbs.append(color_component)
        el_orbs_labels.append("+".join(label_text))         # e.g. 'O.s.pz+H.s' becomes 'O(s+p_z)+H(s)'

    
    color_values = { 0: 'red', 1: 'green', 2: 'blue' }
    print('\tProjections:')
    for color_idx, color_contrib in enumerate(el_orbs_labels):
        print(f'\t\t{color_values[color_idx]}: {color_contrib}')

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

    logging.info('Contribution table ([red, green, blue] format)')
    logging.info(f'Colors are: red = {el_orbs_labels[0]}, green = {el_orbs_labels[1]}, blue = {el_orbs_labels[2]}')
    logging.info(f'band' + '\t\t\t' + '\t\t\t\t\t'.join(labels))

    # obtain contributions
    # sum over all bands
    for b in range(bands.nb_bands):
        # sum over all k-points
        for k in range(len(bands.kpoints)):
            for color_idx, color_contrib in enumerate(el_orbs): # color_idx: up to either 2 or 3. color_contrib: list of [elements, orbitals] for each color
                for element_contrib in color_contrib:
                    element = element_contrib[0]        # e.g. 'X', 'N'
                    orbitals = element_contrib[1]       # e.g. 'all', 's', ['s', 'pz']
                    if element == 'X':                  # if all atoms, get all indexes
                        element_indexes = range(len(atom_labels))
                    else:                               # else, get the indexes with label = atom symbol (e.g. 'C')
                        element_indexes = [i for i, x in enumerate(atom_labels) if x == element]

                    if orbitals == 'all':
                        orbital_indexes = range(0,max_l_index)  # sum all orbitals if 'all'
                    else:
                        orbital_indexes = [orbital_values[o] for o in orbitals]

                    for i in element_indexes:
                        for j in orbital_indexes:
                            contrib_bands[b,k,color_idx] += data[Spin.up][k][b][i][j]**2
                            if k == 0 and b == 0:
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


    # plot bands using rgb mapping
    for b in range(bands.nb_bands):
        rgbline(ax1,
                KPOINTS,
                [e - bands.efermi for e in bands.bands[Spin.up][b]],
                contrib_bands[b,:,0],
                contrib_bands[b,:,1],
                contrib_bands[b,:,2])   #this is np.zeros if only two colors are defined

    # plot DOS
    ax2.plot(contrib_dos[0],dosrun.tdos.energies - dosrun.efermi, \
            c=(1.0,0.0,0.0), label = f'${el_orbs_labels[0]}$', linewidth = 1)
    ax2.plot(contrib_dos[1],dosrun.tdos.energies - dosrun.efermi, \
            c=(0.0,1.0,0.0), label = f'${el_orbs_labels[1]}$', linewidth = 1)
    if len(el_orbs) == 3:
        ax2.plot(contrib_dos[2],dosrun.tdos.energies - dosrun.efermi, \
            c=(0.0,0.0,1.0), label = f'${el_orbs_labels[2]}$', linewidth = 1)












#----- Program starts -------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print('--- plot_fatbands.py --------------------------------------')
    print(f'\tPlotting fatbands into {args.output_file}.{args.format}')
    print(f'\tAdditional data is printed into plot_fatbands.log')

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


    # general options for plot
    font = {'family': 'serif', 'size': args.font_size}
    plt.rc('font', **font)

    # set up 2 graph with aspec ratio args.ratio/1
    # plot 1: bands diagram
    # plot 2: DOS
    gs = GridSpec(1, 2, width_ratios=[args.ratio,1], wspace=0.1)
    fig = plt.figure(figsize=(args.width, args.height))
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1]) #, sharey=ax1)

    # Set both fermi levels equal to the band fermi level
    bands.efermi =  dosrun.efermi #= 0

    # set y limits for the plot
    emin = args.emin
    emax = args.emax

    # if either is not defined, get the min and/or the max of energy from bands.bands.keys()
    if emin is None and emax is None:
        emin=100
        emax=-100
        for spin in bands.bands.keys():
            for b in range(bands.nb_bands):
                emin = min(emin, min(bands.bands[spin][b]))
                emax = max(emax, max(bands.bands[spin][b]))
    elif emin is None:
        emin=100
        for spin in bands.bands.keys():
            for b in range(bands.nb_bands):
                emin = min(emin, min(bands.bands[spin][b]))
    elif emax is None:
        emax=-100
        for spin in bands.bands.keys():
            for b in range(bands.nb_bands):
                emax = max(emax, max(bands.bands[spin][b]))
    
    # set y-axis limit
    ax1.set_ylim(emin, emax)
    ax2.set_ylim(emin, emax)


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


    # style
    ax1.set_ylabel(r"$E - E_f$ (eV)",labelpad=-2) 	#labelpad might work bad
    ax1.grid(lw=args.glw,alpha=0.5)

    # fermi level line at 0
    ax1.hlines(y=0, xmin=0, xmax=len(bands.kpoints), color="k", lw=args.flw)

    TICKS = [0.0]
    for i in range(step,len(KPOINTS)+step,step):
        ax1.vlines(KPOINTS[i-1], emin, emax, "k",lw=args.vlw)
        TICKS.append(KPOINTS[i-1])
    ax1.set_xticks(TICKS)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis='x', which='both', length=0, pad=5)
    ax1.set_xlim(0, KPOINTS[-1])


    if no_proj is False:
        CalculateProjections()
	DOSlabel = 'total'
    else:
        print('\tNo projection requested. Plotting normal bands.')
        for b in range(bands.nb_bands): 
            ax1.plot(KPOINTS,[e - bands.efermi for e in bands.bands[Spin.up][b]], lw=args.plw, color='k')
        DOSlabel = None

    ax2.fill_betweenx(dosrun.tdos.energies - dosrun.efermi,
        0,dosrun.tdos.densities[Spin.up],
        color = (0.7, 0.7, 0.7),
        facecolor = (0.7, 0.7, 0.7))
    ax2.plot(dosrun.tdos.densities[Spin.up],
        dosrun.tdos.energies - dosrun.efermi,
        color = (0.6, 0.6, 0.6),
        label = DOSlabel, lw=args.plw)
    ax2.legend(fancybox=False, shadow=False, prop={'size': args.font_size-1},labelspacing=0.15,borderpad=0.20,handlelength=1.2,framealpha=0.6)

    # scaling factor for the x axis limit, if the peaks are too high
    maxdos = max(dosrun.tdos.densities[Spin.up])/scale

    
    ax2.set_yticklabels([])
    ax2.grid(lw=args.glw,alpha=0.5)
    ax2.set_xticks([])
    ax2.set_xlim(0,maxdos)
    ax2.hlines(y=0, xmin=0, xmax=maxdos, color="k", lw=args.flw)
    ax2.set_xlabel("DOS")

    # Plotting 
    # -----------------
    plt.savefig(f"{args.output_file}.{args.format}", format=args.format, bbox_inches='tight')

    print('\tFile saved.')
    print('-----------------------------------------------------------')
