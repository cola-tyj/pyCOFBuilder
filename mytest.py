# # importing module
# import sys
 
# # appending a path
# sys.path.append('{PATH_TO_PYCOFBUILDER}/pyCOFBuilder/src')

import pycofbuilder as pcb

# cof = pcb.Framework('H6_HECO_CHO_OH-H6_HECO_NH2_H-HXL-AA')    #失败的
# cof = pcb.Framework('S4_PHPR_CHO_H-S4_PTCA_CH2CN_H-FXT-AA')  #成功的
# cof = pcb.Framework('S4_PHPR_CHO_OH-L2_DFDB_NH2_H-FXT_A-AA')  #成功的
# cof = pcb.Framework('T3_DBA2_CHO_OH-T3_TPAM_NH2_H-HNB_A-AA')  #成功
# cof = pcb.Framework('T3_DBA2_CHO_OH-L2_DFDB_NH2_H-HCA_A-AA')
# cof = pcb.Framework('L2_BDFN_CHO_CHO-L2_BENZ_NH2_NH2-HGB-AA')
cof = pcb.Framework('T3_TPTZ_CONHNH2_OH-L2_BPNY_CHO_H-FESa-AA')

# cof = pcb.Framework('R4_ETKB_CHO_H-R4_ETKB_NH2_H-KGM-AA')
# cof = pcb.Framework('T3_DBA2_CHO_OH-T3_DBA2_NH2_H-HCA-AA')    #成功的

# cof = pcb.Framework('R4_ETKB_CH2CN_OH-L2_DPEL_CHO_H-KGM_A-AA')  

cof.save(fmt='cif', supercell = [1, 1, 2], save_dir = '.')