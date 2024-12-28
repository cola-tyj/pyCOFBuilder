# importing module
import sys
 
# appending a path
sys.path.append('{PATH_TO_PYCOFBUILDER}/pyCOFBuilder/src')

import pycofbuilder as pcb

cof = pcb.Framework('H6_HECO_CHO_OH-L2_BENZ_NH2_H-HXL_A-AA')
cof.save(fmt='cif', supercell = [1, 1, 2], save_dir = '.')