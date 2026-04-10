import sys
import argparse
import logging
import os.path
import numpy as np
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.core import Element
from pymatgen.electronic_structure.core import Spin, Orbital, OrbitalType

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class SaneFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter): pass

parser = argparse.ArgumentParser(
    prog='plot_dos_landscape.py',
    description=("Plot projected Density of States from a VASP calculation in landscape format.\n" +
                 "Adapted to plot Energy on the X-axis and DOS on the Y-axis.\n"),
    formatter_class=SaneFormatter
)

parser.add_argument('-D','--vasprun-file-dos', type=str, help='Path of the vasprun.xml file of the dos calculation', default='vasprun.xml')
parser.add_argument('-P','--POTCAR-file', type=str, help='Path of the POTCAR file', default='../../POTCAR') #'../
parser.add_argument('-p','--project', help='DOS projections. Accepts 1-5 arguments. Nomenclature:\n'
                    '\t- E: all orbitals of element with symbol E (H, C, N, ...)\n'
                    '\t- E.o: o-orbital (s, p, px, py, pz, d, ...) of element with symbol E\n'
                    '\t- X.o: o-orbital of all elements (literally X to indicate all elements)\n'
                    '\t- Example: O.s.pz+N.pz means sum of O(s,pz) and N(pz)\n',
                    nargs='+', default=['X.px', 'X.py', 'X.s.pz'])
parser.add_argument('-N','--no-projection', help='Do not perform any projection', action='store_true')
parser.add_argument('-T','--no-total', help='Do not plot the total DOS', action='store_true')
parser.add_argument('-m','--emin', type=float, help='Minimum of energy in the plot.', default=-6.0)
parser.add_argument('-M','--emax', type=float, help='Maximum of energy in the plot.', default=6.0)
parser.add_argument('-s','--scale', type=float, help='DOS scale factor for upper limit', default=1.0)
parser.add_argument('-H','--height', type=float, help='Height of the plot in inches', default=3.0)
parser.add_argument('-W','--width', type=float, help='Width of the plot in inches', default=4.0)
parser.add_argument('--dlw','--dos-lw', type=float, help='Linewidth of DOS', default=1.25)
parser.add_argument('--flw','--Fermi-lw', type=float, help='Linewidth of Fermi level. Set it to 0 to remove it', default=1.0)
parser.add_argument('--glw','--grid-lw', type=float, help='Linewidth of grid. Set it to 0 to remove them', default=0.0)
parser.add_argument('--gla','--grid-alpha', type=float, help='alpha value (transparency) of grid lines.', default=0.5)
parser.add_argument('--nofermi', help='Show E_max instead of E_F as the Fermi level', action='store_true')
parser.add_argument('-f','--font-size', type=float, help='Fontsize', default=7)
parser.add_argument('--cmap','--colormap', type=str, help='Color scheme for projections.',
                    choices=['rgb','saturated','normal','pale','dark','alt','darkalt','accent','plt','sumo'], default='normal')
parser.add_argument('--cord','--color-order', nargs='+', type=int, help='Change the color order. E.g. --cord 3 2 1 4 5.',
                    default=[1,2,3,4,5])
parser.add_argument('--custc','--custom-colors', nargs='+', help='Custom colors, as in matplotlib. E.g. --custc red green blue', default='None')
parser.add_argument('-o','--output-file', type=str, help='Path and name of the output file, excluding the format', default='hdos')
parser.add_argument('--format', type=str, help='Output file format', choices=['pdf','png'], default='pdf')
parser.add_argument('--no-legend', help='Do not show the legend.', action='store_true')
parser.add_argument('--clabels', '--custom-labels', nargs='+', type=str, help='Custom labels for the legend. Must match the number of projections.', default=None)
parser.add_argument('--redo','--readlog', help='Rerun the last command from the .log file.', action='store_true')

args = parser.parse_args()
redo = args.redo

if args.no_total and args.no_projection:
    raise ValueError(f'You do not want to plot anything?')

if redo is True:
    if os.path.isfile(f'plot_{args.output_file}.log') is False:
        raise ValueError(f'I cannot rerun: plot_{args.output_file}.log does not exist')
    
    with open(f'plot_{args.output_file}.log') as f:
        argstr = f.readline()
    
    args = parser.parse_args(argstr.split()[1:]) 

logging.basicConfig(
    filename=f'plot_{args.output_file}.log', 
    level=logging.INFO, 
    filemode="w", 
    format="%(message)s",
)

if redo is True:
    logging.info(argstr)
else:
    logging.info(" ".join(sys.argv[:]))

# -----------------------------------------------------------------------------
# Color definitions
# -----------------------------------------------------------------------------
if args.custc == 'None':
    match args.cmap:
        case 'rgb':
            colors = [[1,0,0], [0,1,0], [0,0,1], [1, 0.50, 0.14], [0.58,0.40,0.74]]
        case 'saturated':
            colors = [mcolors.BASE_COLORS['r'], mcolors.BASE_COLORS['g'], mcolors.BASE_COLORS['b'], mcolors.TABLEAU_COLORS['tab:orange'], mcolors.BASE_COLORS['m']]
        case 'pale':
            colors = [mcolors.TABLEAU_COLORS['tab:red'], mcolors.TABLEAU_COLORS['tab:green'], mcolors.TABLEAU_COLORS['tab:blue'], mcolors.TABLEAU_COLORS['tab:orange'], mcolors.TABLEAU_COLORS['tab:purple']]
        case 'normal':
            colors = [plt.cm.Set1.colors[0], plt.cm.Set1.colors[2], plt.cm.Set1.colors[1], plt.cm.Set1.colors[4], plt.cm.Set1.colors[3]]
        case 'dark':
            colors = [mcolors.CSS4_COLORS['darkred'], mcolors.CSS4_COLORS['darkgreen'], mcolors.CSS4_COLORS['darkblue'],  mcolors.CSS4_COLORS['chocolate'], mcolors.CSS4_COLORS['purple']]
        case 'alt':
            colors = [plt.cm.Set2.colors[i] for i in range(5)]
        case 'darkalt':
            colors = [plt.cm.Dark2.colors[i] for i in range(5)]
        case 'accent':
            colors = [plt.cm.Accent.colors[i] for i in range(5)]
        case 'plt':
            colors = [plt.cm.tab10.colors[i] for i in range(5)]
        case 'sumo':
            colors = ["#3952A3", "#FAA41A", "#67BC47", "#6ECCDD", "#ED2025"]
else:
    colors = [ mcolors.CSS4_COLORS[cval] for cval in args.custc ] 

color_order = {0: args.cord[0]-1, 1: args.cord[1]-1, 2: args.cord[2]-1, 3: args.cord[3]-1, 4: args.cord[4]-1}

# -----------------------------------------------------------------------------
# Read Data
# -----------------------------------------------------------------------------
print(f'Reading DOS from {args.vasprun_file_dos}')
try:
    dosrun = Vasprun(args.vasprun_file_dos,parse_potcar_file=args.POTCAR_file)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find {args.vasprun_file_dos}")

cdos = dosrun.complete_dos
efermi = dosrun.efermi
energies = cdos.energies - efermi
tdos = cdos.densities[Spin.up]

emin, emax = args.emin, args.emax

# -----------------------------------------------------------------------------
# Projections
# -----------------------------------------------------------------------------
def parse_and_sum_dos(proj_str, cdos):
    total_densities = np.zeros_like(cdos.energies)
    components = proj_str.split('+')
    all_elements = [str(el) for el in cdos.structure.composition.elements]
    
    for comp in components:
        parts = comp.split('.')
        target_str = parts[0]
        orbs_str = parts[1:] if len(parts) > 1 else []
        
        if target_str.isdigit():
            # Atom index projection (1-based to match VASP/PROCAR)
            site_idx = int(target_str) - 1
            if site_idx < 0 or site_idx >= len(cdos.structure):
                print(f"Warning: Atom index {target_str} is out of bounds.")
                continue
            
            site = cdos.structure[site_idx]
            site_dos = cdos.pdos[site]
            
            if not orbs_str:
                # All orbitals for this specific site
                for orb_enum, dos_vals in site_dos.items():
                    total_densities += dos_vals[Spin.up]
            else:
                # Specific orbitals for this specific site
                for orb in orbs_str:
                    try:
                        orb_enum = Orbital[orb]
                        if orb_enum in site_dos:
                            total_densities += site_dos[orb_enum][Spin.up]
                    except KeyError:
                        print(f"Warning: Could not parse orbital '{orb}' for atom {target_str}")
                        
        else:
            # Element projection
            target_elements = all_elements if target_str == 'X' else [target_str]
            for el in target_elements:
                if el not in all_elements: 
                    continue
                
                if not orbs_str:
                    # Full element DOS
                    el_dos = cdos.get_element_dos()[Element(el)]
                    total_densities += el_dos.densities[Spin.up]
                else:
                    # Orbital specific DOS
                    for orb in orbs_str:
                        try:
                            # Try general spd DOS first (s, p, d)
                            orb_type = OrbitalType[orb]
                            spd_dos = cdos.get_element_spd_dos(el)[orb_type]
                            total_densities += spd_dos.densities[Spin.up]
                        except KeyError:
                            # Fallback to specific orbitals (px, py, dxy, etc.)
                            try:
                                orb_enum = Orbital[orb]
                                for site in cdos.structure:
                                    if site.species_string == el:
                                        pdos = cdos.pdos[site]
                                        if orb_enum in pdos:
                                            total_densities += pdos[orb_enum][Spin.up]
                            except KeyError:
                                print(f"Warning: Could not parse orbital '{orb}' for element '{el}'")
                                
    return total_densities

# -----------------------------------------------------------------------------
# Plot Setup
# -----------------------------------------------------------------------------
plt.rcParams.update({'font.size': args.font_size})
fig, ax_DOS = plt.subplots(figsize=(args.width, args.height))

if not args.no_total:
    ax_DOS.fill_between(energies, 0, tdos, color=(0.7, 0.7, 0.7), facecolor=(0.7, 0.7, 0.7),alpha=0.7)
    DOSlabel = 'total'
    ax_DOS.plot(energies, tdos, color=(0.6, 0.6, 0.6), label=DOSlabel, lw=args.dlw)

if not args.no_projection:
    print('Calculating DOS projections...')
    
    # Verify custom labels match the number of projections
    use_custom_labels = False
    if args.clabels:
        if len(args.clabels) == len(args.project):
            use_custom_labels = True
        else:
            print("Warning: Number of custom labels does not match number of projections. Using default labels.")

    for i, proj in enumerate(args.project):
        idx = color_order[i % 5] % len(colors)
        color = colors[idx]
        proj_dens = parse_and_sum_dos(proj, cdos)
        
        if use_custom_labels:
            formatted_label = args.clabels[i]
        else:
            # 1. Format "Element.orb1.orb2" into "Element(orb1,orb2)"
            formatted_comps = []
            for comp in proj.split('+'):
                parts = comp.split('.')
                if len(parts) > 1:
                    formatted_comps.append(f"{parts[0]}({','.join(parts[1:])})")
                else:
                    formatted_comps.append(parts[0])
            
            formatted_label = "+".join(formatted_comps)
            
            # 2. Apply subscripts for the legend
            formatted_label = (formatted_label.replace('px', '$p_x$')
                                              .replace('py', '$p_y$')
                                              .replace('pz', '$p_z$')
                                              .replace('dxy', '$d_{xy}$')
                                              .replace('dyz', '$d_{yz}$')
                                              .replace('dxz', '$d_{xz}$')
                                              .replace('dz2', '$d_{z^2}$')
                                              .replace('dx2-y2', '$d_{x^2-y^2}$'))
        
        ax_DOS.plot(energies, proj_dens, color=color, label=formatted_label, lw=args.dlw)

# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------

# Create a boolean mask for the energies within the plot range
mask = (energies >= emin) & (energies <= emax)

# Safely calculate the maximum DOS within the window, ignoring NaNs/Infs
if np.any(mask):
    maxdos = np.nanmax(tdos[mask]) / args.scale
else:
    maxdos = np.nanmax(tdos) / args.scale

# Fallback in case the array is entirely NaNs or flat
if np.isnan(maxdos) or np.isinf(maxdos) or maxdos <= 0:
    maxdos = 1.0

ax_DOS.set_xlim(emin, emax)
ax_DOS.set_ylim(0, maxdos)

ax_DOS.grid(lw=args.glw, alpha=args.gla, zorder=0)
ax_DOS.set_axisbelow(True)

if not args.nofermi:
    ax_DOS.set_xlabel(r"$E - E_F$ (eV)", labelpad=5)
    ax_DOS.vlines(0, 0, maxdos, color="k", lw=args.flw, zorder=6)
else:
    ax_DOS.set_xlabel(r"$E - E_{max}$ (eV)", labelpad=5)

ax_DOS.set_ylabel("DOS", labelpad=5)
ax_DOS.tick_params(axis='y', length=0, width=1, which='major')
ax_DOS.set_yticklabels([]) # Replicates the style of removing DOS Y-axis text
ax_DOS.tick_params(axis='x', which='both', pad=5)

if not args.no_projection or not args.no_legend:
    ax_DOS.legend(fancybox=False, shadow=False, prop={'size': args.font_size-1}, 
                  labelspacing=0.15, borderpad=0.20, handlelength=1.2, framealpha=0.6)

plt.savefig(f"{args.output_file}.{args.format}", format=args.format, bbox_inches='tight', dpi=400)
print(f'\tFile saved as {args.output_file}.{args.format}.')
print('-----------------------------------------------------------')
