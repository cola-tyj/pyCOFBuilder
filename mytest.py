# # importing module
# import sys
 
# # appending a path
# sys.path.append('{PATH_TO_PYCOFBUILDER}/pyCOFBuilder/src')

import pycofbuilder as pcb

# cof = pcb.Framework('H6_HECO_CHO_OH-H6_HECO_NH2_H-HXL-AA')    #失败的
# cof = pcb.Framework('S4_PHPR_CHO_H-S4_PTCA_CH2CN_H-FXT-AA')  #成功的
# cof = pcb.Framework('S4_PHPR_CHO_OH-L2_DFDB_NH2_H-FXT_A-AA')  #成功的
# cof = pcb.Framework('S4_PHPR_BOH2_OH-L2_DFDB_OH2_H-FXT_A-AA')   #失败的
cof = pcb.Framework('H6_HECO_CHO_OH-H6_HECO_NH2_H-HXL-AA')

cof.save(fmt='cif', supercell = [1, 1, 2], save_dir = '.')