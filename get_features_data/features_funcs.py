import os, numpy as np, pandas as pd
import xlsxwriter
from sklearn.metrics import auc

from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.electronic_structure.cohp import  CompleteCohp
from pymatgen.electronic_structure import cohp
from pymatgen.io.lobster import Cohpcar
from pymatgen.electronic_structure.core import Orbital, Spin
from monty.io import zopen
from pymatgen.electronic_structure.cohp import Cohp
from pymatgen.io.lobster.inputs import Lobsterin

from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from jarvis.io.vasp.outputs import Outcar



def get_cohp_cobi_feats(COHPCAR_path, POSCAR_path, lobsterin_path, are_cobis=False):
	pmg_structure=Structure.from_file(POSCAR_path)

	completecohp=CompleteCohp.from_file(fmt="LOBSTER",filename=COHPCAR_path,structure_file=POSCAR_path, are_cobis=are_cobis)
	cohp_bonds=completecohp.bonds; #print(list(cohp_bonds.keys()))	
	labelist=list(cohp_bonds.keys())
	cohp_data=completecohp.get_summed_cohp_by_label_list(label_list=labelist, divisor=1)

	if Spin.down in cohp_data.cohp:
		populations = cohp_data.cohp[Spin.up]+cohp_data.cohp[Spin.down]
	else:
		populations = cohp_data.cohp[Spin.up]
	
	number_energies_below_efermi = len([x for x in cohp_data.energies if x <= cohp_data.efermi])
	cohp_below_fermiE=populations[0:number_energies_below_efermi]

	cohp_energies_below_fermiE=[x for x in cohp_data.energies if x <= cohp_data.efermi]
	#cohp_bond_strength=auc(cohp_energies_below_fermiE, cohp_below_fermiE)
	icohp=auc(cohp_energies_below_fermiE, cohp_below_fermiE)

	lob_cls=Lobsterin.from_file(lobsterin_path)
	nbasis_funcs=lob_cls._get_nbands(structure=pmg_structure)

	norm_ICOHP=icohp/nbasis_funcs

	return norm_ICOHP

COBICAR_1spin_filename = "COBICAR_spin1.lobster"
COBICAR_2spin_filename = "COBICAR.lobster"
COHPCAR_1spin_filename = "COHPCAR_spin1.lobster"
COHPCAR_2spin_filename = "COHPCAR.lobster"

POSCAR_filename = "POSCAR"
lobsterin_filename='lobsterin'
#COHPCAR_path=os.path.join(os.getcwd(), COHPCAR_filename)
POSCAR_path = os.path.join(os.getcwd(), POSCAR_filename)
lobsterin_path=os.path.join(os.getcwd(), lobsterin_filename)
pmg_structure=Structure.from_file('POSCAR')


norm_cobi_1spin=get_cohp_cobi_feats(POSCAR_path=POSCAR_path, lobsterin_path=lobsterin_path, are_cobis=True, COHPCAR_path=os.path.join(os.getcwd(), COBICAR_1spin_filename))
norm_cobi_2spin=get_cohp_cobi_feats(POSCAR_path=POSCAR_path, lobsterin_path=lobsterin_path, are_cobis=True, COHPCAR_path=os.path.join(os.getcwd(), COBICAR_2spin_filename))
norm_cohp_1spin=get_cohp_cobi_feats(POSCAR_path=POSCAR_path, lobsterin_path=lobsterin_path, are_cobis=False, COHPCAR_path=os.path.join(os.getcwd(), COHPCAR_1spin_filename))
norm_cohp_2spin=get_cohp_cobi_feats(POSCAR_path=POSCAR_path, lobsterin_path=lobsterin_path, are_cobis=False, COHPCAR_path=os.path.join(os.getcwd(), COHPCAR_2spin_filename))


print('norm ICOBI with 1 spin:', norm_cobi_1spin)
print('norm ICOBI with 2 spins:', norm_cobi_2spin)
print('norm ICOHP with 1 spin:', norm_cohp_1spin)
print('norm ICOHP with 2 spins:', norm_cohp_2spin)


"""
norm ICOBI with 1 spin: 0.46616574723750004
norm ICOBI with 2 spins: 0.46627223146875013
norm ICOHP with 1 spin: -4.78699544943125
norm ICOHP with 2 spins: -4.7871338880500005
"""

