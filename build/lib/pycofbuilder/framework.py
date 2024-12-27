# -*- coding: utf-8 -*-
# Created by Felipe Lopes de Oliveira
# Distributed under the terms of the MIT License.

"""
The Framework class implements definitions and methods for a Framework buiding
"""

import os
import copy
import numpy as np

# Import pymatgen
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation

from scipy.spatial.transform import Rotation as R

# Import pycofbuilder exceptions
from pycofbuilder.exceptions import (BondLenghError,
                                     BBConnectivityError)

# Import pycofbuilder building_block
from pycofbuilder.building_block import BuildingBlock

# Import pycofbuilder topology data
from pycofbuilder.data.topology import TOPOLOGY_DICT

# Import pycofbuilder tools
from pycofbuilder.tools import (get_bond_atom,
                                cell_to_cellpar,
                                cellpar_to_cell,
                                rotation_matrix_from_vectors,
                                unit_vector,
                                angle,
                                get_framework_symm_text,
                                get_bonds)

# Import pycofbuilder io_tools
from pycofbuilder.io_tools import (save_chemjson,
                                   save_cif,
                                   save_xyz,
                                   save_turbomole,
                                   save_vasp,
                                   save_xsf,
                                   save_pdb,
                                   save_pqr,
                                   save_qe,
                                   save_gjf)

from pycofbuilder.logger import create_logger


class Framework():
    """
    A class used to represent a Covalent Organic Framework as a reticular entity.

    ...

    Attributes
    ----------
    name : str
        Name of the material
    out_dir : str
        Path to save the results.
        If not defined, a `out` folder will be created in the current directory.
    verbosity : str
        Control the printing options. Can be 'none', 'normal', or 'debug'.
        Default: 'normal'
    save_bb : bool
        Control the saving of the building blocks.
        Default: True
    lib_path : str
        Path for saving the building block files.
        If not defined, a `building_blocks` folder will be created in the current directory.
    topology : str = None
    dimention : str = None
    lattice : str = None
    lattice_sgs : str = None
    space_group : str = None
    space_group_n : str = None
    stacking : str = None
    mass : str = None
    composition : str = None
    charge : int  = 0
    multiplicity : int = 1
    chirality : bool = False
    atom_types : list = []
    atom_pos : list = []
    lattice : list = [[], [], []]
    symm_tol : float = 0.2
    angle_tol : float = 0.2
    dist_threshold : float = 0.8
    available_2D_topologies : list
        List of available 2D topologies
    available_3D_topologies : list
        List of available 3D topologies
    available_topologies : list
        List of all available topologies
    available_stacking : list
        List of available stakings for all 2D topologies
    lib_bb : str
        String with the name of the folder containing the building block files
        Default: bb_lib
    """

    def __init__(self, name: str = None, **kwargs):

        self.name: str = name
        #name 是一个可选参数，用于指定框架的名称。

        self.out_path: str = kwargs.get('out_dir', os.path.join(os.getcwd(), 'out'))
        # 输出目录的路径，默认存储在当前工作目录下的out文件夹
        self.save_bb: bool = kwargs.get('save_bb', True)
        # 决定是否保存 building blocks 构建块，默认为True
        self.bb_out_path: str = kwargs.get('bb_out_path', os.path.join(self.out_path, 'building_blocks'))
        # 是保存构建块的路径

        self.logger = create_logger(level=kwargs.get('log_level', 'info'),
                                    format=kwargs.get('log_format', 'simple'),
                                    save_to_file=kwargs.get('save_to_file', False),
                                    log_filename=kwargs.get('log_filename', 'pycofbuilder.log'))
        # 初始化日志记录器，记录框架的各种操作和信息

# frame的几何和物理属性
        self.symm_tol = kwargs.get('symm_tol', 0.1)
        # 对称性容差，用于处理框架的对称性
        self.angle_tol = kwargs.get('angle_tol', 0.5)
        # 角度容差，调整框架中原子和化学键之间的角度
        self.dist_threshold = kwargs.get('dist_threshold', 0.8)
        # 距离阈值，筛选原子间是否存在有效的相互作用
        self.bond_threshold = kwargs.get('bond_threshold', 1.3)
        # 化学键的距离阈值
    
# frame的基本结构属性
        self.bb1_name = None    
        self.bb2_name = None
        self.topology = None
        # 框架的拓扑结构
        self.stacking = None
        # 框架的堆叠类型
        self.smiles = None
        # 存储SMILES字符串，表示分子的化学结构

        self.atom_types = []
        # 存储原子的类型（C H O
        self.atom_pos = []
        # 原子的空间坐标 position
        self.atom_labels = []
        # 原子的标签或名称
# 晶胞矩阵和晶胞参数
        self.cellMatrix = np.eye(3)
        # cellMatrix 是3x3的单位矩阵
        self.cellParameters = np.array([1, 1, 1, 90, 90, 90]).astype(float)
        # 是一个包含晶胞参数的数组，包括晶胞的三个边长和三个角度
        self.bonds = []
        # 化学键 bond

# 晶体学相关属性
        self.lattice_sgs = None
        # 晶格对称群
        self.space_group = None
        # 晶体的空间群信息
        self.space_group_n = None
        # 空间群的编号 num

# frame的其他物理属性
        self.dimention = None
        # 维度 2D or 3D
        self.n_atoms = self.get_n_atoms()
        # 框架中的原子数量
        self.mass = None
        # frame 质量
        self.composition = None
        # 组成
        self.charge = 0
        # 电荷，默认为0
        self.multiplicity = 1
        # frame的多重性（自旋多重性）
        self.chirality = False
        # frame是否具备手性（手对称

# 可用的拓扑结构和堆叠类型
        self.available_2D_top = ['HCB', 'HCB_A',
                                 'SQL', 'SQL_A',
                                 'KGD',
                                 'HXL', 'HXL_A',
                                 'FXT', 'FXT_A']

        # To add: ['dia', 'bor', 'srs', 'pts', 'ctn', 'rra', 'fcc', 'lon', 'stp', 'acs', 'tbo', 'bcu', 'fjh', 'ceq']
        self.available_3D_top = ['DIA', 'DIA_A', 'BOR']  # Temporary
        self.available_topologies = self.available_2D_top + self.available_3D_top
        # 所支持的拓扑结构类型

        # Define available stackings for all 2D topologies
        self.available_stacking = {
            'HCB': ['A', 'AA', 'AB1', 'AB2', 'AAl', 'AAt', 'ABC1', 'ABC2'],
            'HCB_A': ['A', 'AA', 'AB1', 'AB2', 'AAl', 'AAt', 'ABC1', 'ABC2'],
            'SQL': ['A', 'AA', 'AB1', 'AB2', 'AAl', 'AAt', 'ABC1', 'ABC2'],
            'SQL_A': ['A', 'AA', 'AB1', 'AB2', 'AAl', 'AAt', 'ABC1', 'ABC2'],
            'KGD': ['A', 'AA', 'AB1', 'AB2', 'AAl', 'AAt', 'ABC1', 'ABC2'],
            'HXL': ['A', 'AA', 'AB1', 'AB2', 'AAl', 'AAt', 'ABC1', 'ABC2'],
            'HXL_A': ['A', 'AA', 'AB1', 'AB2', 'AAl', 'AAt', 'ABC1', 'ABC2'],
            'FXT': ['A', 'AA', 'AB1', 'AB2', 'AAl', 'AAt', 'ABC1', 'ABC2'],
            'FXT_A': ['A', 'AA', 'AB1', 'AB2', 'AAl', 'AAt', 'ABC1', 'ABC2'],
            'DIA': [str(i + 1) for i in range(15)],
            'DIA_A': [str(i + 1) for i in range(15)],
            'BOR': [str(i + 1) for i in range(15)]
        }
        # 所支持的堆叠类型

        if self.name is not None:
            self.from_name(self.name)
        # 根据框架名称加载相应数据

# 用于返回对象的“用户友好”字符串表示，易于理解的
    def __str__(self) -> str:
        return self.as_string()

# 返回对象的“开发者友好”的字符串表示，目的是提供一个尽可能明确和详细的对象描述，方便调试和开发。
    def __repr__(self) -> str:
        return f'Framework({self.bb1_name}, {self.bb2_name}, {self.topology}, {self.stacking})'

# 用于生成一个详细的框架信息字符串。
    def as_string(self) -> str:
        """
        Returns a string with the Framework information.
        """

        fram_str = f'Name: {self.name}\n'
        # 框架名称

        # Get the formula of the framework

        if self.composition is not None:
            fram_str += f'Full Formula ({self.composition})\n'
            fram_str += f'Reduced Formula: ({self.composition})\n'
        else:
            fram_str += 'Full Formula ()\n'
            fram_str += 'Reduced Formula: \n'
        # 若有 self.composition，则显示完整的化学式，否则显示空

        fram_str += 'abc   :  {:11.6f}  {:11.6f}  {:11.6f}\n'.format(*self.cellParameters[:3])
        fram_str += 'angles:  {:11.6f}  {:11.6f}  {:11.6f}\n'.format(*self.cellParameters[3:])
        fram_str += 'A: {:11.6f}  {:11.6f}   {:11.6f}\n'.format(*self.cellMatrix[0])
        fram_str += 'B: {:11.6f}  {:11.6f}   {:11.6f}\n'.format(*self.cellMatrix[1])
        fram_str += 'C: {:11.6f}  {:11.6f}   {:11.6f}\n'.format(*self.cellMatrix[2])
        # 格式化输出frame的晶胞参数（3个边长和3个角度）以及晶胞的晶胞矩阵

        fram_str += f'Cartesian Sites ({self.n_atoms})\n'
        fram_str += '  #  Type         a         b         c    label\n'
        fram_str += '---  ----  --------  --------  --------  -------\n'

        for i in range(len(self.atom_types)):
            fram_str += '{:3d}  {:4s}  {:8.5f}  {:8.5f}  {:8.5f}  {:>7}\n'.format(i,
                                                                                  self.atom_types[i],
                                                                                  self.atom_pos[i][0],
                                                                                  self.atom_pos[i][1],
                                                                                  self.atom_pos[i][2],
                                                                                  self.atom_labels[i])

        # 输出frame中每个原子的1类型、位置（abc坐标）和标签
        return fram_str

# 返回frame中单元格内原子数量，单元格只单位晶胞 unitary cell
    def get_n_atoms(self) -> int:
        """
        Returns the number of atoms in the unitary cell
        """
        return len(self.atom_types)

# 返回支持的拓扑结构列表，可以根据维度进行筛选
    def get_available_topologies(self, dimensionality: str = 'all', print_result: bool = True):
        """
        Get the available topologies implemented in the class.

        Parameters
        ----------

        dimensionality : str, optional
            The dimensionality of the topologies to be printed. Can be 'all', '2D' or '3D'.
            Default: 'all'
        print_result: bool, optional
            If True, the available topologies are printed.

        Returns
        -------
        dimensionality_list: list
            A list with the available topologies.
        """

        dimensionality_error = f'Dimensionality must be one of the following: all, 2D, 3D, not {dimensionality}'
        assert dimensionality in ['all', '2D', '3D'], dimensionality_error

        dimensionality_list = []

        if dimensionality == 'all' or dimensionality == '2D':
            if print_result:
                print('Available 2D Topologies:')
            for i in self.available_2D_top:
                if print_result:
                    print(i.upper())
                dimensionality_list.append(i)

        if dimensionality == 'all' or dimensionality == '3D':
            if print_result:
                print('Available 3D Topologies:')
            for i in self.available_3D_top:
                if print_result:
                    print(i.upper())
                dimensionality_list.append(i)

        return dimensionality_list

# 检查frame的名称是否符合正确的格式，并返回一个元组，包含构建块名称、网格类型和堆叠方式
    def check_name_concistency(self, FrameworkName) -> tuple[str, str, str, str]:
        """
        Checks if the name is in the correct format and returns a
        tuple with the building blocks names, the net and the stacking.

        In case the name is not in the correct format, an error is raised.
        如果名字不是正确格式，则会报错

        Parameters
        ----------
        FrameworkName : str, required
            The name of the COF to be created
        框架名称，必须是一个字符串。格式为构建块1-构建块2-网格类型-堆叠方式：BB1-BB2-Net-Stacking
        Returns
        -------
        tuple[str, str, str, str]
            A tuple with the building blocks names, the net and the stacking.
        """

        string_error = 'FrameworkName must be a string'
        assert isinstance(FrameworkName, str), string_error
        # 首先验证是否为字符串

        name_error = 'FrameworkName must be in the format: BB1-BB2-Net-Stacking'
        assert len(FrameworkName.split('-')) == 4, name_error
        # 检查是否包含由‘-’分割的4个部分

        bb1_name, bb2_name, Net, Stacking = FrameworkName.split('-')
        # 将Name进行分解

        net_error = f'{Net} not in the available list: {self.available_topologies}'
        assert Net in self.available_topologies, net_error
        # 检查Net是否在可用的拓扑列表：self.available_topologies中

        stacking_error = f'{Stacking} not in the available list: {self.available_stacking[Net]}'
        assert Stacking in self.available_stacking[Net], stacking_error
        # 检查Stacking是否在对应拓扑的堆叠类型列表中

        return bb1_name, bb2_name, Net, Stacking

# 根据frame的名称创建一个新的 COF
    def from_name(self, FrameworkName, **kwargs) -> None:
        """Creates a COF from a given FrameworkName.

        Parameters
        ----------
        FrameworkName : str, required
            The name of the COF to be created

        Returns
        -------
        COF : Framework
            The COF object
        """
        bb1_name, bb2_name, Net, Stacking = self.check_name_concistency(FrameworkName)
        # 调用方法验证并解析框架名称

        bb1 = BuildingBlock(name=bb1_name, bb_out_path=self.bb_out_path, save_bb=self.save_bb)
        bb2 = BuildingBlock(name=bb2_name, bb_out_path=self.bb_out_path, save_bb=self.save_bb)
        # 创建两个 Building Block 对象，表示frame的两个构建块

        self.from_building_blocks(bb1, bb2, Net, Stacking, **kwargs)
        # 调用方法根据这些构建块和拓扑信息构建COF

# 根据给定的构建块和网络类型、堆叠方式创建COF结构
    def from_building_blocks(self,
                             bb1: BuildingBlock,
                             bb2: BuildingBlock,
                             net: str,
                             stacking: str,
                             **kwargs):
        """Creates a COF from the building blocks.

        Parameters
        ----------
        BB1 : BuildingBlock, required
            The first building block
        BB2 : BuildingBlock, required
            The second building block
        Net : str, required
            The network of the COF
        Stacking : str, required
            The stacking of the COF

        Returns
        -------
        COF : Framework
            The COF object
        """
        self.name = f'{bb1.name}-{bb2.name}-{net}-{stacking}'
        self.bb1_name = bb1.name
        self.bb2_name = bb2.name
        self.topology = net
        self.stacking = stacking

        # Check if the BB1 has the smiles attribute
        if hasattr(bb1, 'smiles') and hasattr(bb2, 'smiles'):
            self.smiles = f'{bb1.smiles}.{bb2.smiles}'
        else:
            print('WARNING: The smiles attribute is not available for the building blocks')
        # 检查bb1 和 bb2 对象是否具有smile属性，如果有则将信息合并并存储为框架的smile属性

        net_build_dict = {
            'HCB': self.create_hcb_structure,
            'HCB_A': self.create_hcb_a_structure,
            'SQL': self.create_sql_structure,
            'SQL_A': self.create_sql_a_structure,
            'KGD': self.create_kgd_structure,
            'HXL': self.create_hxl_structure,
            'HXL_A': self.create_hxl_a_structure,
            'FXT': self.create_fxt_structure,
            'FXT_A': self.create_fxt_a_structure,
            'DIA': self.create_dia_structure,
            'DIA_A': self.create_dia_a_structure,
            'BOR': self.create_bor_structure
            }
        # 根据 net 选择合适的函数来创建具体的结构（通过net_build_dict字典映射

        result = net_build_dict[net](bb1, bb2, stacking, **kwargs)

        structure = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        )

        self.bonds = get_bonds(structure, self.bond_threshold)

        return result

# 将 frame结构保存为指定格式文件，如 cif,xyz,json等
    def save(self,
             fmt: str = 'cif',
             supercell: list = (1, 1, 1),
             save_dir=None,
             primitive=False,
             save_bonds=True) -> None:
            #  save_bonds：是否保存化学键
        
        """
        Save the structure in a specif file format.

        Parameters
        ----------
        fmt : str, optional
        # fmt：文件格式，默认为cif
            The file format to be saved
            Can be `json`, `cif`, `xyz`, `turbomole`, `vasp`, `xsf`, `pdb`, `pqr`, `qe`.
            Default: 'cif'
        supercell : list, optional
        # 列表，指定保存结构时的supercell大小
            The supercell to be used to save the structure.
            Default: [1,1,1]
        save_dir : str, optional
        # 保存文件的路径
            The path to save the structure. By default, the structure is saved in a
            `out` folder created in the current directory.
        primitive : bool, optional
        # 如果为True，保存原始晶胞，否则保存常规晶胞
            If True, the primitive cell is saved. Otherwise, the conventional cell is saved.
            Default: False
        """

        save_dict = {
            # 使用字典映射文件格式到具体的保存函数
            'cjson': save_chemjson,
            'cif': save_cif,
            'xyz': save_xyz,
            'turbomole': save_turbomole,
            'vasp': save_vasp,
            'xsf': save_xsf,
            'pdb': save_pdb,
            'pqr': save_pqr,
            'qe': save_qe,
            'gjf': save_gjf
         }

        file_format_error = f'Format must be one of the following: {save_dict.keys()}'
        assert fmt in save_dict.keys(), file_format_error

        if primitive:
            structure = self.prim_structure
            # 保存原始结构

        else:
            structure = Structure(
                # 使用frame的cellMatrix等信息创建结构
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

        structure.make_supercell(supercell)
        # 生成supercell（超胞

        if save_bonds:
            bonds = get_bonds(structure, self.bond_threshold)
        else:
            bonds = []

        structure_dict = structure.as_dict()

        cell = structure_dict['lattice']['matrix']

        atom_types = [site['species'][0]['element'] for site in structure_dict['sites']]
        atom_labels = [site['properties']['source'] for site in structure_dict['sites']]
        atom_pos = [site['xyz'] for site in structure_dict['sites']]

        if save_dir is None:
            save_path = self.out_path
        else:
            save_path = save_dir

        save_dict[fmt](path=save_path,
                       file_name=self.name,
                       cell=cell,
                       atom_types=atom_types,
                       atom_labels=atom_labels,
                       atom_pos=atom_pos,
                       bonds=bonds)

    def make_cubic(self,
                   min_length=10,
                   force_diagonal=False,
                   force_90_degrees=True,
                   min_atoms=None,
                   max_atoms=None,
                   angle_tolerance=1e-3):
        # 将frame的晶胞转换为一个立方体supercell。该方法会调整frame的晶胞，使其成为一个包含至少min_length长度立方体，并满足原子数限制
        """
        Transform the primitive structure into a supercell with alpha, beta, and
        gamma equal to 90 degrees. The algorithm will iteratively increase the size
        of the supercell until the largest inscribed cube's side length is at least 'min_length'
        and the number of atoms in the supercell falls in the range ``min_atoms < n < max_atoms``.

        Parameters
        ----------
        min_length : float, optional
            Minimum length of the cubic cell (default is 10)
        # supercell的最小边长，默认为10
        force_diagonal : bool, optional
            If True, generate a transformation with a diagonal transformation matrix (default is False)
        # 是否强制使用对角矩阵进行转换
        force_90_degrees : bool, optional
            If True, force the angles to be 90 degrees (default is True)
        # 是否强制所有角度为90度
        min_atoms : int, optional
            Minimum number of atoms in the supercell (default is None)
        # csupercell中的最小原子数
        max_atoms : int, optional
            Maximum number of atoms in the supercell (default is None)
        # 最大原子数
        angle_tolerance : float, optional
            The angle tolerance for the transformation (default is 1e-3)
        # 角度公差，默认为1e-3
        """

        # 使用CubicSupercellTransformation类将原始结构转换为一个立方体supercell，
        # 且根据指定条件（最小边长、原子数）等进行调整
        cubic_dict = CubicSupercellTransformation(
            min_length=min_length,
            force_90_degrees=force_90_degrees,
            force_diagonal=force_diagonal,
            min_atoms=min_atoms,
            max_atoms=max_atoms,
            angle_tolerance=angle_tolerance
            ).apply_transformation(self.prim_structure).as_dict()

        # 更新frame的晶胞矩阵和晶胞参数
        self.cellMatrix = np.array(cubic_dict['lattice']['matrix']).astype(float)
        self.cellParameters = cell_to_cellpar(self.cellMatrix)

        # 更新frame的原子类型、坐标和标签
        self.atom_types = [i['label'] for i in cubic_dict['sites']]
        self.atom_pos = [i['xyz'] for i in cubic_dict['sites']]
        self.atom_labels = [i['properties']['source'] for i in cubic_dict['sites']]

# --------------- Net creation methods -------------------------- #

    def create_hcb_structure(self,
                             BB_T3_A,
                             BB_T3_B,
                             stacking: str = 'AA',
                             slab: float = 10.0,
                             shift_vector: list = (1.0, 1.0, 0),
                             tilt_angle: float = 5.0):
        # 用于创建一个具有HCB网络（蜂窝网络）的共价有机框架，这种网络由两个三脚型（tripodal）
        # 构建块（BB_T_A和 BB_T_B）组成，并支持多种层间堆叠模式（如AA，AB，ABC等）
        # 该方法主要完成以下任务：
        # 1.验证构建块的连通性是否为3
        # 2.计算晶胞参数，并生成晶格结构
        # 3.根据堆叠模式排列构建块
        # 4.返回frame的晶体学和对称性信息
        """Creates a COF with HCB network.

        The HCB net is composed of two tripodal building blocks.

        Parameters
        ----------
        BB_T3_1 : BuildingBlock, required
            The BuildingBlock object of the tripodal Buiding Block A
        BB_T3_2 : BuildingBlock, required
            The BuildingBlock object of the tripodal Buiding Block B
        # 分别是两个三脚型构建块,包含了构建块的原子类型,原子位置和连通性等信息.
        stacking : str, optional
            The stacking pattern of the COF layers (default is 'AA')
        # 决定了COF层的堆叠方式,堆叠方式包括:A,AA:简单平移叠层;AB1,AB2:二层交错;ABC1，ABC2，三层交错；AAl，AAt：平移和倾斜变体。
        slab : float, optional
            Default parameter for the interlayer slab (default is 10.0)
        # 控制层间的默认间距。
        shift_vector: list, optional
            Shift vector for the AAl and AAt stakings (defatult is [1.0,1.0,0])
        # 平移向量，用于AAl和AAt的堆叠模式
        tilt_angle: float, optional
            Tilt angle for the AAt staking in degrees (default is 5.0)
        # 倾斜角度，用于AAt的堆叠模式

        Returns
        -------
        list
            A list of strings containing:
                1. the structure name,
                2. lattice type,
                3. hall symbol of the cristaline structure,
                4. space group,
                5. number of the space group,
                6. number of operation symmetry
        """

        connectivity_error = 'Building block {} must present connectivity {} not {}'

        # 检查构建块的连通性是否为3，如果不是就抛出错误
        if BB_T3_A.connectivity != 3:
            self.logger.error(connectivity_error.format('A', 3, BB_T3_A.connectivity))
            raise BBConnectivityError(3, BB_T3_A.connectivity)
        if BB_T3_B.connectivity != 3:
            self.logger.error(connectivity_error.format('B', 3, BB_T3_B.connectivity))
            raise BBConnectivityError(3, BB_T3_B.connectivity)

        # 初始化框架属性
        self.name = f'{BB_T3_A.name}-{BB_T3_B.name}-HCB-{stacking}'
        self.topology = 'HCB'
        self.staking = stacking
        self.dimension = 2

        self.charge = BB_T3_A.charge + BB_T3_B.charge
        self.chirality = BB_T3_A.chirality or BB_T3_B.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_T3_A.conector, BB_T3_B.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_T3_A.conector,
                                                                                 BB_T3_B.conector))

        # Replace "X" the building block
        BB_T3_A.replace_X(bond_atom)

        # Remove the "X" atoms from the the building block
        BB_T3_A.remove_X()
        BB_T3_B.remove_X()

# 计算晶胞参数
        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]
        # 从拓扑信息中获得HCB网络的基本信息

        # Measure the base size of the building blocks
        size = BB_T3_A.size[0] + BB_T3_B.size[0]

        # Calculate the delta size to add to the c parameter
        delta_a = abs(max(np.transpose(BB_T3_A.atom_pos)[2])) + abs(min(np.transpose(BB_T3_A.atom_pos)[2]))
        delta_b = abs(max(np.transpose(BB_T3_B.atom_pos)[2])) + abs(min(np.transpose(BB_T3_B.atom_pos)[2]))

        delta_max = max([delta_a, delta_b])

        # 根据构建块的大小（size）和位置，计算frame的晶胞边长（a,b,c）和角度（alpha,beta,gamma）
        # Calculate the cell parameters
        a = topology_info['a'] * size
        b = topology_info['b'] * size
        c = topology_info['c'] + delta_max
        alpha = topology_info['alpha']
        beta = topology_info['beta']
        gamma = topology_info['gamma']

        if self.stacking == 'A':
            c = slab
        # 如果堆叠模式是A，就将c设置为slab的值

# 创建晶格
        # Create the lattice
        self.cellMatrix = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        self.cellParameters = np.array([a, b, c, alpha, beta, gamma]).astype(float)

# 创建structure
        # Create the structure
        self.atom_types = []
        self.atom_labels = []
        self.atom_pos = []

# 添加构建块到structure
        # Add the A1 building blocks to the structure
        vertice_data = topology_info['vertices'][0]
        self.atom_types += BB_T3_A.atom_types
        vertice_pos = np.array(vertice_data['position'])*a

        R_Matrix = R.from_euler('z',
                                vertice_data['angle'],
                                degrees=True).as_matrix()

        rotated_pos = np.dot(BB_T3_A.atom_pos, R_Matrix) + vertice_pos
        self.atom_pos += rotated_pos.tolist()

        self.atom_labels += ['C1' if i == 'C' else i for i in BB_T3_A.atom_labels]

        # Add the A2 building block to the structure
        vertice_data = topology_info['vertices'][1]
        self.atom_types += BB_T3_B.atom_types
        vertice_pos = np.array(vertice_data['position'])*a

        R_Matrix = R.from_euler('z',
                                vertice_data['angle'],
                                degrees=True).as_matrix()

        rotated_pos = np.dot(BB_T3_B.atom_pos, R_Matrix) + vertice_pos
        self.atom_pos += rotated_pos.tolist()

        self.atom_labels += ['C2' if i == 'C' else i for i in BB_T3_B.atom_labels]

        # Creates a pymatgen structure
        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        # Translates the structure to the center of the cell
        StartingFramework.translate_sites(
            range(len(StartingFramework.as_dict()['sites'])),
            [0, 0, 0.5],
            frac_coords=True,
            to_unit_cell=True)

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)
        self.cellParameters = cell_to_cellpar(self.cellMatrix)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]

        if stacking == 'A' or stacking == 'AA':
            stacked_structure = StartingFramework

        if stacking == 'AB1':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [2/3, 1/3, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'AB2':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/2, 0, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/2, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'ABC1':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (2/3, 1/3, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (4/3, 2/3, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'ABC2':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 0, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 0, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'AAl':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            sv = np.array(shift_vector)
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos + sv))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        # Create AA tilted stacking.
        if stacking == 'AAt':
            cell = StartingFramework.as_dict()['lattice']

            # Shift the cell by the tilt angle
            a_cell = cell['a']
            b_cell = cell['b']
            c_cell = cell['c'] * 2
            alpha = cell['alpha'] - tilt_angle
            beta = cell['beta'] - tilt_angle
            gamma = cell['gamma']

            self.cellMatrix = cellpar_to_cell([a_cell, b_cell, c_cell, alpha, beta, gamma])
            self.cellParameters = np.array([a_cell, b_cell, c_cell, alpha, beta, gamma]).astype(float)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        dict_structure = stacked_structure.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)
        self.cellParameters = cell_to_cellpar(self.cellMatrix)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = stacked_structure.formula

# 检查原子间的距离
        dist_matrix = StartingFramework.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

# 计算对称性信息
        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(stacked_structure,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_refined_structure(keep_site_properties=True)

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()
        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]

    def create_hcb_a_structure(self,
                               BB_T3: str,
                               BB_L2: str,
                               stacking: str = 'AA',
                               slab: float = 10.0,
                               shift_vector: list = (1.0, 1.0, 0),
                               tilt_angle: float = 5.0):
        """Creates a COF with HCB-A network.

        The HCB-A net is composed of one tripodal and one linear building blocks.

        Parameters
        ----------
        BB_T3 : BuildingBlock, required
            The BuildingBlock object of the tripodal Buiding Block
        BB_L2 : BuildingBlock, required
            The BuildingBlock object of the linear Buiding Block
        stacking : str, optional
            The stacking pattern of the COF layers (default is 'AA')
        c_parameter_base : float, optional
            The base value for interlayer distance in angstroms (default is 3.6)
        print_result : bool, optional
            Parameter for the control for printing the result (default is True)
        slab : float, optional
            Default parameter for the interlayer slab (default is 10.0)
        shift_vector: list, optional
            Shift vector for the AAl and AAt stakings (defatult is [1.0,1.0,0])
        tilt_angle: float, optional
            Tilt angle for the AAt staking in degrees (default is 5.0)

        Returns
        -------
        list
            A list of strings containing:
                1. the structure name
                2. lattice type
                3. hall symbol of the cristaline structure
                4. space group
                5. number of the space group,
                6. number of operation symmetry
        """

        connectivity_error = 'Building block {} must present connectivity {} not {}'
        if BB_T3.connectivity != 3:
            self.logger.error(connectivity_error.format('A', 3, BB_T3.connectivity))
            raise BBConnectivityError(3, BB_T3.connectivity)
        if BB_L2.connectivity != 2:
            self.logger.error(connectivity_error.format('B', 3, BB_L2.connectivity))
            raise BBConnectivityError(2, BB_L2.connectivity)

        self.name = f'{BB_T3.name}-{BB_L2.name}-HCB_A-{stacking}'
        self.topology = 'HCB_A'
        self.staking = stacking
        self.dimension = 2

        self.charge = BB_L2.charge + BB_T3.charge
        self.chirality = BB_L2.chirality or BB_T3.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_T3.conector, BB_L2.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_T3.conector,
                                                                                 BB_L2.conector))

        # Replace "X" the building block
        BB_L2.replace_X(bond_atom)

        # Remove the "X" atoms from the the building block
        BB_T3.remove_X()
        BB_L2.remove_X()

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        # Measure the base size of the building blocks
        size = BB_T3.size[0] + BB_L2.size[0]

        # Calculate the delta size to add to the c parameter
        delta_a = abs(max(np.transpose(BB_T3.atom_pos)[2])) + abs(min(np.transpose(BB_T3.atom_pos)[2]))
        delta_b = abs(max(np.transpose(BB_L2.atom_pos)[2])) + abs(min(np.transpose(BB_L2.atom_pos)[2]))

        delta_max = max([delta_a, delta_b])

        # Calculate the cell parameters
        a = topology_info['a'] * size
        b = topology_info['b'] * size
        c = topology_info['c'] + delta_max
        alpha = topology_info['alpha']
        beta = topology_info['beta']
        gamma = topology_info['gamma']

        if self.stacking == 'A':
            c = slab

        # Create the lattice
        self.cellMatrix = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        self.cellParameters = np.array([a, b, c, alpha, beta, gamma]).astype(float)

        # Create the structure
        self.atom_types = []
        self.atom_labels = []
        self.atom_pos = []

        # Add the building blocks to the structure
        for vertice_data in topology_info['vertices']:
            self.atom_types += BB_T3.atom_types
            vertice_pos = np.array(vertice_data['position'])*a

            R_Matrix = R.from_euler('z', vertice_data['angle'], degrees=True).as_matrix()

            rotated_pos = np.dot(BB_T3.atom_pos, R_Matrix) + vertice_pos
            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C1' if i == 'C' else i for i in BB_T3.atom_labels]

        # Add the building blocks to the structure
        for edge_data in topology_info['edges']:
            self.atom_types += BB_L2.atom_types

            R_Matrix = R.from_euler('z', edge_data['angle'], degrees=True).as_matrix()

            rotated_pos = np.dot(BB_L2.atom_pos, R_Matrix) + np.array(edge_data['position'])*a

            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C2' if i == 'C' else i for i in BB_L2.atom_labels]

        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        # Translates the structure to the center of the cell
        StartingFramework.translate_sites(
            range(len(StartingFramework.as_dict()['sites'])),
            [0, 0, 0.5],
            frac_coords=True,
            to_unit_cell=True
        )

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]

        if stacking == 'A' or stacking == 'AA':
            stacked_structure = StartingFramework

        if stacking == 'AB1':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [2/3, 1/3, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'AB2':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/2, 0, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/2, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'ABC1':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (2/3, 1/3, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (4/3, 2/3, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'ABC2':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 0, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 0, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'AAl':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            sv = np.array(shift_vector)
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos + sv))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        # Create AA tilted stacking.
        if stacking == 'AAt':
            cell = StartingFramework.as_dict()['lattice']

            # Shift the cell by the tilt angle
            a_cell = cell['a']
            b_cell = cell['b']
            c_cell = cell['c'] * 2
            alpha = cell['alpha'] - tilt_angle
            beta = cell['beta'] - tilt_angle
            gamma = cell['gamma']

            self.cellMatrix = cellpar_to_cell([a_cell, b_cell, c_cell, alpha, beta, gamma])
            self.cellParameters = np.array([a_cell, b_cell, c_cell, alpha, beta, gamma]).astype(float)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        dict_structure = stacked_structure.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = stacked_structure.formula

        dist_matrix = stacked_structure.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(stacked_structure,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_refined_structure(keep_site_properties=True)

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()

        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]

    def create_sql_structure(self,
                             BB_A: str,
                             BB_B: str,
                             stacking: str = 'AA',
                             slab: float = 10.0,
                             shift_vector: list = (1.0, 1.0, 0),
                             tilt_angle: float = 5.0):
        """Creates a COF with SQL network.

        The SQL net is composed of two tetrapodal building blocks.

        Parameters
        ----------
        BB_S4_A : BuildingBlock, required
            The BuildingBlock object of the tetrapodal Buiding Block A
        BB_S4_B : BuildingBlock, required
            The BuildingBlock object of the tetrapodal Buiding Block B
        stacking : str, optional
            The stacking pattern of the COF layers (default is 'AA')
        print_result : bool, optional
            Parameter for the control for printing the result (default is True)
        slab : float, optional
            Default parameter for the interlayer slab (default is 10.0)
        shift_vector: list, optional
            Shift vector for the AAl and AAt stakings (defatult is [1.0,1.0,0])
        tilt_angle: float, optional
            Tilt angle for the AAt staking in degrees (default is 5.0)

        Returns
        -------
        list
            A list of strings containing:
                1. the structure name,
                2. lattice type,
                3. hall symbol of the cristaline structure,
                4. space group,
                5. number of the space group,
                6. number of operation symmetry
        """

        connectivity_error = 'Building block {} must present connectivity {} not {}'
        if BB_A.connectivity != 4:
            self.logger.error(connectivity_error.format('A', 4, BB_A.connectivity))
            raise BBConnectivityError(4, BB_A.connectivity)
        if BB_B.connectivity != 4:
            self.logger.error(connectivity_error.format('B', 4, BB_B.connectivity))
            raise BBConnectivityError(4, BB_B.connectivity)

        self.name = f'{BB_A.name}-{BB_B.name}-SQL-{stacking}'
        self.topology = 'SQL'
        self.staking = stacking
        self.dimension = 2

        self.charge = BB_A.charge + BB_B.charge
        self.chirality = BB_A.chirality or BB_B.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Get the positions of the "X" atoms
        _, X_A = BB_A.get_X_points()
        _, X_B = BB_B.get_X_points()

        # Calculate the alpha angle for the rotation of the building blocks
        alpha_A = -np.arctan2(X_A[0][1] - X_A[-1][1], X_A[0][0] - X_A[-1][0])
        alpha_B = -np.arctan2(X_B[0][1] - X_B[-1][1], X_B[0][0] - X_B[-1][0])

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_A.conector, BB_B.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_A.conector,
                                                                                 BB_B.conector))

        # Replace "X" the building block
        BB_A.replace_X(bond_atom)

        # Remove the "X" atoms from the the building block
        BB_A.remove_X()
        BB_B.remove_X()

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        # Measure the base size of the building blocks
        size_A = (np.abs(BB_A.size[0] * np.sin(alpha_A)) + np.abs(BB_B.size[0] * np.cos(alpha_B))) * 2
        size_B = (np.abs(BB_A.size[0] * np.cos(alpha_A)) + np.abs(BB_B.size[0] * np.cos(alpha_B))) * 2

        # Calculate the delta size to add to the c parameter
        delta_a = abs(max(np.transpose(BB_A.atom_pos)[2])) + abs(min(np.transpose(BB_B.atom_pos)[2]))
        delta_b = abs(max(np.transpose(BB_A.atom_pos)[2])) + abs(min(np.transpose(BB_B.atom_pos)[2]))

        delta_max = max([delta_a, delta_b])

        # Calculate the cell parameters
        a = topology_info['a'] * size_A
        b = topology_info['b'] * size_B
        c = topology_info['c'] + delta_max
        alpha = topology_info['alpha']
        beta = topology_info['beta']
        gamma = topology_info['gamma']

        if self.stacking == 'A':
            c = slab

        # Create the lattice
        self.cellMatrix = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        self.cellParameters = np.array([a, b, c, alpha, beta, gamma]).astype(float)

        # Create the structure
        self.atom_types = []
        self.atom_labels = []
        self.atom_pos = []

        # Add the first building block to the structure
        self.atom_types += BB_A.atom_types
        vertice_pos = np.array([0, 0, 0])

        R_Matrix = R.from_euler('z', -alpha_A, degrees=False).as_matrix()

        rotated_pos = np.dot(BB_A.atom_pos, R_Matrix)
        self.atom_pos += rotated_pos.tolist()

        self.atom_labels += ['C1' if i == 'C' else i for i in BB_A.atom_labels]

        # Add the second building block to the structure
        self.atom_types += BB_B.atom_types
        vertice_pos = np.array([a/2, b/2, 0])

        R_Matrix = R.from_euler('z', -alpha_B, degrees=False).as_matrix()

        rotated_pos = np.dot(BB_B.atom_pos, R_Matrix) + vertice_pos
        self.atom_pos += rotated_pos.tolist()

        self.atom_labels += ['C2' if i == 'C' else i for i in BB_B.atom_labels]

        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        # Translates the structure to the center of the cell
        StartingFramework.translate_sites(
            range(len(StartingFramework.as_dict()['sites'])),
            [0, 0, 0.5],
            frac_coords=True,
            to_unit_cell=True
        )

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]

        if stacking == 'A' or stacking == 'AA':
            stacked_structure = StartingFramework

        if stacking == 'AB1':

            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/4, 1/4, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/4, 1/4, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'AB2':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/2, 0, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/2, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'ABC1':

            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (1/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 1/3, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 2/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 2/3, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'ABC2':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (1/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 0, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 2/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 0, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        # Create AAl stacking. Tetragonal cell with two sheets
        # per cell shifited by the shift_vector in angstroms.
        if stacking == 'AAl':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            sv = np.array(shift_vector)
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos + sv))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        # Create AA tilted stacking.
        # Tilted tetragonal cell with two sheets per cell tilted by tilt_angle.
        if stacking == 'AAt':
            cell = StartingFramework.as_dict()['lattice']

            # Shift the cell by the tilt angle
            a_cell = cell['a']
            b_cell = cell['b']
            c_cell = cell['c'] * 2
            alpha = cell['alpha'] - tilt_angle
            beta = cell['beta'] - tilt_angle
            gamma = cell['gamma']

            self.cellMatrix = cellpar_to_cell([a_cell, b_cell, c_cell, alpha, beta, gamma])
            self.cellParameters = np.array([a_cell, b_cell, c_cell, alpha, beta, gamma]).astype(float)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        dict_structure = stacked_structure.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = stacked_structure.formula

        dist_matrix = stacked_structure.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(stacked_structure,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_refined_structure(keep_site_properties=True)

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()

        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]

    def create_sql_a_structure(self,
                               BB_S4: str,
                               BB_L2: str,
                               stacking: str = 'AA',
                               c_parameter_base: float = 3.6,
                               slab: float = 10.0,
                               shift_vector: list = (1.0, 1.0, 0),
                               tilt_angle: float = 5.0):
        """Creates a COF with SQL-A network.

        The SQL-A net is composed of one tetrapodal and one linear building blocks.

        Parameters
        ----------
        BB_S4 : BuildingBlock, required
            The BuildingBlock object of the tetrapodal Buiding Block
        BB_L2 : BuildingBlock, required
            The BuildingBlock object of the bipodal Buiding Block
        stacking : str, optional
            The stacking pattern of the COF layers (default is 'AA')
        slab : float, optional
            Default parameter for the interlayer slab (default is 10.0)
        shift_vector: list, optional
            Shift vector for the AAl and AAt stakings (defatult is [1.0,1.0,0])
        tilt_angle: float, optional
            Tilt angle for the AAt staking in degrees (default is 5.0)

        Returns
        -------
        list
            A list of strings containing:
                1. the structure name,
                2. lattice type,
                3. hall symbol of the cristaline structure,
                4. space group,
                5. number of the space group,
                6. number of operation symmetry
        """

        connectivity_error = 'Building block {} must present connectivity {} not {}'
        if BB_S4.connectivity != 4:
            self.logger.error(connectivity_error.format('A', 4, BB_S4.connectivity))
            raise BBConnectivityError(4, BB_S4.connectivity)
        if BB_L2.connectivity != 2:
            self.logger.error(connectivity_error.format('B', 3, BB_L2.connectivity))
            raise BBConnectivityError(2, BB_L2.connectivity)

        self.name = f'{BB_S4.name}-{BB_L2.name}-SQL_A-{stacking}'
        self.topology = 'SQL_A'
        self.staking = stacking
        self.dimension = 2

        self.charge = BB_S4.charge + BB_L2.charge
        self.chirality = BB_S4.chirality or BB_L2.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Get the position of the "X" atoms
        _, X_S4 = BB_S4.get_X_points()

        # Calculate the alpha angle for the rotation of the building blocks
        alpha_S4 = -np.arctan2(X_S4[0][1] - X_S4[-1][1], X_S4[0][0] - X_S4[-1][0])

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_S4.conector, BB_L2.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_S4.conector,
                                                                                 BB_L2.conector))

        # Replace "X" the building block
        BB_L2.replace_X(bond_atom)

        # Remove the "X" atoms from the the building block
        BB_S4.remove_X()
        BB_L2.remove_X()

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        # Measure the base size of the building blocks
        size_A = ((BB_S4.size[0] + BB_L2.size[0]) * np.abs(np.sin(alpha_S4))) * 4
        size_B = ((BB_S4.size[0] + BB_L2.size[0]) * np.abs(np.cos(alpha_S4))) * 4

        # Calculate the delta size to add to the c parameter
        delta_a = abs(max(np.transpose(BB_S4.atom_pos)[2])) + abs(min(np.transpose(BB_S4.atom_pos)[2]))
        delta_b = abs(max(np.transpose(BB_L2.atom_pos)[2])) + abs(min(np.transpose(BB_L2.atom_pos)[2]))

        delta_max = max([delta_a, delta_b])

        # Calculate the cell parameters
        a = topology_info['a'] * size_A
        b = topology_info['b'] * size_B
        c = topology_info['c'] + delta_max
        alpha = topology_info['alpha']
        beta = topology_info['beta']
        gamma = topology_info['gamma']

        if self.stacking == 'A':
            c = slab

        # Create the lattice
        self.cellMatrix = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        self.cellParameters = np.array([a, b, c, alpha, beta, gamma]).astype(float)

        # Create the structure
        self.atom_types = []
        self.atom_labels = []
        self.atom_pos = []

        # Add the S4 building blocks to the structure
        for vertice_data in topology_info['vertices']:
            self.atom_types += BB_S4.atom_types
            vertice_pos = np.array(vertice_data['position']) * np.array([a, b, c])

            R_Matrix = R.from_euler('z', -alpha_S4, degrees=False).as_matrix()

            rotated_pos = np.dot(BB_S4.atom_pos, R_Matrix) + vertice_pos
            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C1' if i == 'C' else i for i in BB_S4.atom_labels]

        L2_angle_list = [-alpha_S4, alpha_S4, -alpha_S4, alpha_S4]

        # Add the L2 building blocks to the structure
        for i, edge_data in enumerate(topology_info['edges']):
            self.atom_types += BB_L2.atom_types
            edge_pos = np.array(edge_data['position']) * np.array([a, b, c])

            R_Matrix = R.from_euler('z', L2_angle_list[i], degrees=False).as_matrix()

            rotated_pos = np.dot(BB_L2.atom_pos, R_Matrix) + edge_pos
            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C2' if i == 'C' else i for i in BB_L2.atom_labels]

        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        # Translates the structure to the center of the cell
        StartingFramework.translate_sites(
            range(len(StartingFramework.as_dict()['sites'])),
            [0, 0, 0.5],
            frac_coords=True,
            to_unit_cell=True
        )

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]

        if stacking == 'A' or stacking == 'AA':
            stacked_structure = StartingFramework

        if stacking == 'AB1':

            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/4, 1/4, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/4, 1/4, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'AB2':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/2, 0, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/2, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'ABC1':

            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (1/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 1/3, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 2/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 2/3, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'ABC2':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (1/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 0, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 2/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 0, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        # Create AAl stacking. Tetragonal cell with two sheets
        # per cell shifited by the shift_vector in angstroms.
        if stacking == 'AAl':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            sv = np.array(shift_vector)
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos + sv))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        # Create AA tilted stacking.
        # Tilted tetragonal cell with two sheets per cell tilted by tilt_angle.
        if stacking == 'AAt':
            cell = StartingFramework.as_dict()['lattice']

            # Shift the cell by the tilt angle
            a_cell = cell['a']
            b_cell = cell['b']
            c_cell = cell['c'] * 2
            alpha = cell['alpha'] - tilt_angle
            beta = cell['beta'] - tilt_angle
            gamma = cell['gamma']

            self.cellMatrix = cellpar_to_cell([a_cell, b_cell, c_cell, alpha, beta, gamma])
            self.cellParameters = np.array([a_cell, b_cell, c_cell, alpha, beta, gamma]).astype(float)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        dict_structure = stacked_structure.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = stacked_structure.formula

        dist_matrix = stacked_structure.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(stacked_structure,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_refined_structure(keep_site_properties=True)

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()

        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]

    def create_kgd_structure(self,
                             BB_H6: str,
                             BB_T3: str,
                             stacking: str = 'AA',
                             print_result: bool = True,
                             slab: float = 10.0,
                             shift_vector: list = (1.0, 1.0, 0),
                             tilt_angle: float = 5.0):
        """Creates a COF with KGD network.

        The KGD net is composed of one hexapodal and one tripodal building blocks.

        Parameters
        ----------
        BB_H6 : BuildingBlock, required
            The BuildingBlock object of the hexapodal Buiding Block
        BB_T3 : BuildingBlock, required
            The BuildingBlock object of the tripodal Buiding Block
        stacking : str, optional
            The stacking pattern of the COF layers (default is 'AA')
        c_parameter_base : float, optional
            The base value for interlayer distance in angstroms (default is 3.6)
        print_result : bool, optional
            Parameter for the control for printing the result (default is True)
        slab : float, optional
            Default parameter for the interlayer slab (default is 10.0)
        shift_vector: list, optional
            Shift vector for the AAl and AAt stakings (defatult is [1.0,1.0,0])
        tilt_angle: float, optional
            Tilt angle for the AAt staking in degrees (default is 5.0)

        Returns
        -------
        list
            A list of strings containing:
                1. the structure name,
                2. lattice type,
                3. hall symbol of the cristaline structure,
                4. space group,
                5. number of the space group,
                6. number of operation symmetry
        """
        connectivity_error = 'Building block {} must present connectivity {} not {}'
        if BB_H6.connectivity != 6:
            self.logger.error(connectivity_error.format('A', 6, BB_H6.connectivity))
            raise BBConnectivityError(6, BB_H6.connectivity)
        if BB_T3.connectivity != 3:
            self.logger.error(connectivity_error.format('B', 3, BB_T3.connectivity))
            raise BBConnectivityError(3, BB_T3.connectivity)

        self.name = f'{BB_H6.name}-{BB_T3.name}-KGD-{stacking}'
        self.topology = 'KGD'
        self.staking = stacking
        self.dimension = 2

        self.charge = BB_H6.charge + BB_T3.charge
        self.chirality = BB_H6.chirality or BB_T3.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_H6.conector, BB_T3.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_H6.conector,
                                                                                 BB_T3.conector))

        # Replace "X" the building block
        BB_H6.replace_X(bond_atom)

        # Remove the "X" atoms from the the building block
        BB_H6.remove_X()
        BB_T3.remove_X()

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        # Measure the base size of the building blocks
        size = BB_H6.size[0] + BB_T3.size[0]

        # Calculate the delta size to add to the c parameter
        delta_a = abs(max(np.transpose(BB_H6.atom_pos)[2])) + abs(min(np.transpose(BB_H6.atom_pos)[2]))
        delta_b = abs(max(np.transpose(BB_T3.atom_pos)[2])) + abs(min(np.transpose(BB_T3.atom_pos)[2]))

        delta_max = max([delta_a, delta_b])

        # Calculate the cell parameters
        a = topology_info['a'] * size
        b = topology_info['b'] * size
        c = topology_info['c'] + delta_max
        alpha = topology_info['alpha']
        beta = topology_info['beta']
        gamma = topology_info['gamma']

        if self.stacking == 'A':
            c = slab

        # Create the lattice
        self.cellMatrix = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        self.cellParameters = np.array([a, b, c, alpha, beta, gamma]).astype(float)

        # Create the structure
        self.atom_types = []
        self.atom_labels = []
        self.atom_pos = []

        # Add the building blocks to the structure
        for vertice_data in topology_info['vertices']:
            self.atom_types += BB_H6.atom_types
            vertice_pos = np.array(vertice_data['position'])*a

            R_Matrix = R.from_euler('z',
                                    vertice_data['angle'],
                                    degrees=True).as_matrix()

            rotated_pos = np.dot(BB_H6.atom_pos, R_Matrix) + vertice_pos
            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C1' if i == 'C' else i for i in BB_H6.atom_labels]

        # Add the building blocks to the structure
        for edge_data in topology_info['edges']:
            self.atom_types += BB_T3.atom_types
            edge_pos = np.array(edge_data['position'])*a

            R_Matrix = R.from_euler('z',
                                    edge_data['angle'],
                                    degrees=True).as_matrix()

            rotated_pos = np.dot(BB_T3.atom_pos, R_Matrix) + edge_pos
            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C2' if i == 'C' else i for i in BB_T3.atom_labels]

        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        # Translates the structure to the center of the cell
        StartingFramework.translate_sites(
            range(len(StartingFramework.as_dict()['sites'])),
            [0, 0, 0.5],
            frac_coords=True,
            to_unit_cell=True
        )

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]

        if stacking == 'A' or stacking == 'AA':
            stacked_structure = StartingFramework

        if stacking == 'AB1':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [2/3, 1/3, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'AB2':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/2, 0, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/2, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'ABC1':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (2/3, 1/3, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (4/3, 2/3, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'ABC2':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 0, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 0, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'AAl':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            sv = np.array(shift_vector)
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos + sv))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        # Create AA tilted stacking.
        if stacking == 'AAt':
            cell = StartingFramework.as_dict()['lattice']

            # Shift the cell by the tilt angle
            a_cell = cell['a']
            b_cell = cell['b']
            c_cell = cell['c'] * 2
            alpha = cell['alpha'] - tilt_angle
            beta = cell['beta'] - tilt_angle
            gamma = cell['gamma']

            self.cellMatrix = cellpar_to_cell([a_cell, b_cell, c_cell, alpha, beta, gamma])
            self.cellParameters = np.array([a_cell, b_cell, c_cell, alpha, beta, gamma]).astype(float)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        dict_structure = stacked_structure.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = stacked_structure.formula

        dist_matrix = stacked_structure.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(stacked_structure,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_refined_structure(keep_site_properties=True)

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()
        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]

    def create_hxl_a_structure(self,
                               BB_H6: str,
                               BB_L2: str,
                               stacking: str = 'AA',
                               print_result: bool = True,
                               slab: float = 10.0,
                               shift_vector: list = (1.0, 1.0, 0),
                               tilt_angle: float = 5.0):
        """Creates a COF with HXL-A network.

        The HXL-A net is composed of one hexapodal and one linear building blocks.

        Parameters
        ----------
        BB_H6 : BuildingBlock, required
            The BuildingBlock object of the tetrapodal Buiding Block A
        BB_L2 : BuildingBlock, required
            The BuildingBlock object of the tetrapodal Buiding Block B
        stacking : str, optional
            The stacking pattern of the COF layers (default is 'AA')
        print_result : bool, optional
            Parameter for the control for printing the result (default is True)
        slab : float, optional
            Default parameter for the interlayer slab (default is 10.0)
        shift_vector: list, optional
            Shift vector for the AAl and AAt stakings (defatult is [1.0,1.0,0])
        tilt_angle: float, optional
            Tilt angle for the AAt staking in degrees (default is 5.0)

        Returns
        -------
        list
            A list of strings containing:
                1. the structure name,
                2. lattice type,
                3. hall symbol of the cristaline structure,
                4. space group,
                5. number of the space group,
                6. number of operation symmetry
        """

        connectivity_error = 'Building block {} must present connectivity {} not {}'
        if BB_H6.connectivity != 6:
            self.logger.error(connectivity_error.format('A', 6, BB_H6.connectivity))
            raise BBConnectivityError(6, BB_H6.connectivity)
        if BB_L2.connectivity != 2:
            self.logger.error(connectivity_error.format('B', 3, BB_L2.connectivity))
            raise BBConnectivityError(2, BB_L2.connectivity)

        self.name = f'{BB_H6.name}-{BB_L2.name}-HXL_A-{stacking}'
        self.topology = 'HXL_A'
        self.staking = stacking
        self.dimension = 2

        self.charge = BB_H6.charge + BB_L2.charge
        self.chirality = BB_H6.chirality or BB_L2.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_H6.conector, BB_L2.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_H6.conector,
                                                                                 BB_L2.conector))

        # Replace "X" the building block
        BB_L2.replace_X(bond_atom)

        # Remove the "X" atoms from the the building block
        BB_H6.remove_X()
        BB_L2.remove_X()

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        # Measure the base size of the building blocks
        size = BB_H6.size[0] + BB_L2.size[0]

        # Calculate the delta size to add to the c parameter
        delta_a = abs(max(np.transpose(BB_H6.atom_pos)[2])) + abs(min(np.transpose(BB_H6.atom_pos)[2]))
        delta_b = abs(max(np.transpose(BB_L2.atom_pos)[2])) + abs(min(np.transpose(BB_L2.atom_pos)[2]))

        delta_max = max([delta_a, delta_b])

        # Calculate the cell parameters
        a = topology_info['a'] * size
        b = topology_info['b'] * size
        c = topology_info['c'] + delta_max
        alpha = topology_info['alpha']
        beta = topology_info['beta']
        gamma = topology_info['gamma']

        if self.stacking == 'A':
            c = slab

        # Create the lattice
        self.cellMatrix = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        self.cellParameters = np.array([a, b, c, alpha, beta, gamma]).astype(float)

        # Create the structure
        self.atom_types = []
        self.atom_labels = []
        self.atom_pos = []

        # Add the building blocks to the structure
        for vertice_data in topology_info['vertices']:
            self.atom_types += BB_H6.atom_types
            vertice_pos = np.array(vertice_data['position'])*a

            R_Matrix = R.from_euler('z',
                                    vertice_data['angle'],
                                    degrees=True).as_matrix()

            rotated_pos = np.dot(BB_H6.atom_pos, R_Matrix) + vertice_pos
            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C1' if i == 'C' else i for i in BB_H6.atom_labels]

        # Add the building blocks to the structure
        for edge_data in topology_info['edges']:
            self.atom_types += BB_L2.atom_types
            edge_pos = np.array(edge_data['position'])*a

            R_Matrix = R.from_euler('z',
                                    edge_data['angle'],
                                    degrees=True).as_matrix()

            rotated_pos = np.dot(BB_L2.atom_pos, R_Matrix) + edge_pos
            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C2' if i == 'C' else i for i in BB_L2.atom_labels]

        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        # Translates the structure to the center of the cell
        StartingFramework.translate_sites(
            range(len(StartingFramework.as_dict()['sites'])),
            [0, 0, 0.5],
            frac_coords=True,
            to_unit_cell=True
        )

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]

        if stacking == 'A' or stacking == 'AA':
            stacked_structure = StartingFramework

        if stacking == 'AB1':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [2/3, 1/3, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'AB2':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/2, 0, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/2, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'ABC1':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (2/3, 1/3, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (4/3, 2/3, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'ABC2':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 0, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 0, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'AAl':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            sv = np.array(shift_vector)
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos + sv))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        # Create AA tilted stacking.
        if stacking == 'AAt':
            cell = StartingFramework.as_dict()['lattice']

            # Shift the cell by the tilt angle
            a_cell = cell['a']
            b_cell = cell['b']
            c_cell = cell['c'] * 2
            alpha = cell['alpha'] - tilt_angle
            beta = cell['beta'] - tilt_angle
            gamma = cell['gamma']

            self.cellMatrix = cellpar_to_cell([a_cell, b_cell, c_cell, alpha, beta, gamma])
            self.cellParameters = np.array([a_cell, b_cell, c_cell, alpha, beta, gamma]).astype(float)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        dict_structure = stacked_structure.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = stacked_structure.formula

        dist_matrix = stacked_structure.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(stacked_structure,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_refined_structure(keep_site_properties=True)

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()
        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]

    def create_fxt_structure(self,
                             BB_R4_A: str,
                             BB_R4_B: str,
                             stacking: str = 'AA',
                             print_result: bool = True,
                             slab: float = 10.0,
                             shift_vector: list = (1.0, 1.0, 0),
                             tilt_angle: float = 5.0):
        """Creates a COF with FXT network.

        The FXT net is composed of two tetrapodal building blocks.

        Parameters
        ----------
        BB_R4_A : BuildingBlock, required
            The BuildingBlock object of the rectangular tetrapodal Buiding Block A
        BB_R4_B : BuildingBlock, required
            The BuildingBlock object of the rectangular tetrapodal Buiding Block B
        stacking : str, optional
            The stacking pattern of the COF layers (default is 'AA')
        print_result : bool, optional
            Parameter for the control for printing the result (default is True)
        slab : float, optional
            Default parameter for the interlayer slab (default is 10.0)
        shift_vector: list, optional
            Shift vector for the AAl and AAt stakings (defatult is [1.0,1.0,0])
        tilt_angle: float, optional
            Tilt angle for the AAt staking in degrees (default is 5.0)

        Returns
        -------
        list
            A list of strings containing:
                1. the structure name,
                2. lattice type,
                3. hall symbol of the cristaline structure,
                4. space group,
                5. number of the space group,
                6. number of operation symmetry
        """

        connectivity_error = 'Building block {} must present connectivity {} not {}'
        if BB_R4_A.connectivity != 4:
            self.logger.error(connectivity_error.format('A', 4, BB_R4_A.connectivity))
            raise BBConnectivityError(4, BB_R4_A.connectivity)
        if BB_R4_B.connectivity != 4:
            self.logger.error(connectivity_error.format('B', 4, BB_R4_B.connectivity))
            raise BBConnectivityError(4, BB_R4_B.connectivity)

        self.name = f'{BB_R4_A.name}-{BB_R4_B.name}-FXT-{stacking}'
        self.topology = 'FXT'
        self.staking = stacking
        self.dimension = 2

        self.charge = BB_R4_A.charge + BB_R4_B.charge
        self.chirality = BB_R4_A.chirality or BB_R4_B.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_R4_A.conector, BB_R4_B.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_R4_A.conector,
                                                                                 BB_R4_B.conector))

        # Get the position of the X atom in the building blocks
        BB_R4_A.get_X_points(bond_atom)
        BB_R4_B.get_X_points(bond_atom)

        # Replace "X" the building block
        BB_R4_A.replace_X(bond_atom)

        # Remove the "X" atoms from the the building block
        BB_R4_A.remove_X()
        BB_R4_B.remove_X()

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        # Measure the base size of the building blocks
        size = 2 * (BB_R4_A.size[0] + BB_R4_B.size[0])

        # Calculate the delta size to add to the c parameter
        delta_a = abs(max(np.transpose(BB_R4_A.atom_pos)[2])) + abs(min(np.transpose(BB_R4_B.atom_pos)[2]))
        delta_b = abs(max(np.transpose(BB_R4_A.atom_pos)[2])) + abs(min(np.transpose(BB_R4_B.atom_pos)[2]))

        delta_max = max([delta_a, delta_b])

        # Calculate the cell parameters
        a = topology_info['a'] * size
        b = topology_info['b'] * size
        c = topology_info['c'] + delta_max
        alpha = topology_info['alpha']
        beta = topology_info['beta']
        gamma = topology_info['gamma']

        if self.stacking == 'A':
            c = slab

        # Create the lattice
        self.cellMatrix = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        self.cellParameters = np.array([a, b, c, alpha, beta, gamma]).astype(float)

        # Create the structure
        self.atom_types = []
        self.atom_labels = []
        self.atom_pos = []

        # Add the first building block to the structure
        vertice_data = topology_info['vertices'][0]
        self.atom_types += BB_R4_A.atom_types
        vertice_pos = np.array(vertice_data['position'])*a

        R_Matrix = R.from_euler('z', vertice_data['angle'], degrees=True).as_matrix()

        rotated_pos = np.dot(BB_R4_A.atom_pos, R_Matrix) + vertice_pos
        self.atom_pos += rotated_pos.tolist()

        self.atom_labels += ['C1' if i == 'C' else i for i in BB_R4_A.atom_labels]

        # Add the second building block to the structure
        for vertice_data in topology_info['vertices'][1:]:
            self.atom_types += BB_R4_B.atom_types
            vertice_pos = np.array(vertice_data['position'])*a

            R_Matrix = R.from_euler('z', vertice_data['angle'], degrees=True).as_matrix()

            rotated_pos = np.dot(BB_R4_B.atom_pos, R_Matrix) + vertice_pos
            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C2' if i == 'C' else i for i in BB_R4_B.atom_labels]

        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        # Translates the structure to the center of the cell
        StartingFramework.translate_sites(
            range(len(StartingFramework.as_dict()['sites'])),
            [0, 0, 0.5],
            frac_coords=True,
            to_unit_cell=True
        )

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]

        if stacking == 'A' or stacking == 'AA':
            stacked_structure = StartingFramework

        if stacking == 'AB1':

            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/4, 1/4, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/4, 1/4, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'AB2':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/2, 0, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/2, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'ABC1':

            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (1/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 1/3, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 2/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 2/3, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'ABC2':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (1/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 0, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 2/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 0, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        # Create AAl stacking. Tetragonal cell with two sheets
        # per cell shifited by the shift_vector in angstroms.
        if stacking == 'AAl':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            sv = np.array(shift_vector)
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos + sv))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        # Create AA tilted stacking.
        # Tilted tetragonal cell with two sheets per cell tilted by tilt_angle.
        if stacking == 'AAt':
            cell = StartingFramework.as_dict()['lattice']

            # Shift the cell by the tilt angle
            a_cell = cell['a']
            b_cell = cell['b']
            c_cell = cell['c'] * 2
            alpha = cell['alpha'] - tilt_angle
            beta = cell['beta'] - tilt_angle
            gamma = cell['gamma']

            self.cellMatrix = cellpar_to_cell([a_cell, b_cell, c_cell, alpha, beta, gamma])
            self.cellParameters = np.array([a_cell, b_cell, c_cell, alpha, beta, gamma]).astype(float)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        dict_structure = stacked_structure.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = stacked_structure.formula

        dist_matrix = stacked_structure.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(stacked_structure,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_refined_structure(keep_site_properties=True)

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()

        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]

    def create_fxt_a_structure(self,
                               BB_S4: str,
                               BB_L2: str,
                               stacking: str = 'AA',
                               c_parameter_base: float = 3.6,
                               print_result: bool = True,
                               slab: float = 10.0,
                               shift_vector: list = (1.0, 1.0, 0),
                               tilt_angle: float = 5.0):
        """Creates a COF with FXT-A network.

        The FXT-A net is composed of one tetrapodal and one linear building blocks.

        Parameters
        ----------
        BB_S4 : BuildingBlock, required
            The BuildingBlock object of the tetrapodal Buiding Block
        BB_L2 : BuildingBlock, required
            The BuildingBlock object of the bipodal Buiding Block
        stacking : str, optional
            The stacking pattern of the COF layers (default is 'AA')
        print_result : bool, optional
            Parameter for the control for printing the result (default is True)
        slab : float, optional
            Default parameter for the interlayer slab (default is 10.0)
        shift_vector: list, optional
            Shift vector for the AAl and AAt stakings (defatult is [1.0,1.0,0])
        tilt_angle: float, optional
            Tilt angle for the AAt staking in degrees (default is 5.0)

        Returns
        -------
        list
            A list of strings containing:
                1. the structure name,
                2. lattice type,
                3. hall symbol of the cristaline structure,
                4. space group,
                5. number of the space group,
                6. number of operation symmetry
        """

        connectivity_error = 'Building block {} must present connectivity {} not {}'
        if BB_S4.connectivity != 4:
            self.logger.error(connectivity_error.format('A', 4, BB_S4.connectivity))
            raise BBConnectivityError(4, BB_S4.connectivity)
        if BB_L2.connectivity != 2:
            self.logger.error(connectivity_error.format('B', 3, BB_L2.connectivity))
            raise BBConnectivityError(2, BB_L2.connectivity)

        self.name = f'{BB_S4.name}-{BB_L2.name}-FXT_A-{stacking}'
        self.topology = 'FXT_A'
        self.staking = stacking
        self.dimension = 2

        self.charge = BB_S4.charge + BB_L2.charge
        self.chirality = BB_S4.chirality or BB_L2.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_S4.conector, BB_L2.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_S4.conector,
                                                                                 BB_L2.conector))

        # Replace "X" the building block
        BB_L2.replace_X(bond_atom)

        # Remove the "X" atoms from the the building block
        BB_S4.remove_X()
        BB_L2.remove_X()

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        # Measure the base size of the building blocks
        size = 2 * (BB_S4.size[0] + BB_L2.size[0])

        # Calculate the delta size to add to the c parameter
        delta_a = abs(max(np.transpose(BB_S4.atom_pos)[2])) + abs(min(np.transpose(BB_S4.atom_pos)[2]))
        delta_b = abs(max(np.transpose(BB_L2.atom_pos)[2])) + abs(min(np.transpose(BB_L2.atom_pos)[2]))

        delta_max = max([delta_a, delta_b])

        # Calculate the cell parameters
        a = topology_info['a'] * size
        b = topology_info['b'] * size
        c = topology_info['c'] + delta_max
        alpha = topology_info['alpha']
        beta = topology_info['beta']
        gamma = topology_info['gamma']

        if self.stacking == 'A':
            c = slab

        # Create the lattice
        self.cellMatrix = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        self.cellParameters = np.array([a, b, c, alpha, beta, gamma]).astype(float)

        # Create the structure
        self.atom_types = []
        self.atom_labels = []
        self.atom_pos = []

        # Add the building blocks to the structure
        for vertice_data in topology_info['vertices']:
            self.atom_types += BB_S4.atom_types
            vertice_pos = np.array(vertice_data['position'])*a

            R_Matrix = R.from_euler('z', vertice_data['angle'], degrees=True).as_matrix()

            rotated_pos = np.dot(BB_S4.atom_pos, R_Matrix) + vertice_pos
            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C1' if i == 'C' else i for i in BB_S4.atom_labels]

        # Add the building blocks to the structure
        for edge_data in topology_info['edges']:
            self.atom_types += BB_L2.atom_types
            edge_pos = np.array(edge_data['position'])*a

            R_Matrix = R.from_euler('z', edge_data['angle'], degrees=True).as_matrix()

            rotated_pos = np.dot(BB_L2.atom_pos, R_Matrix) + edge_pos
            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C2' if i == 'C' else i for i in BB_L2.atom_labels]

        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        # Translates the structure to the center of the cell
        StartingFramework.translate_sites(
            range(len(StartingFramework.as_dict()['sites'])),
            [0, 0, 0.5],
            frac_coords=True,
            to_unit_cell=True
        )

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]

        if stacking == 'A' or stacking == 'AA':
            stacked_structure = StartingFramework

        if stacking == 'AB1':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [2/3, 1/3, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'AB2':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/2, 0, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/2, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'ABC1':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (2/3, 1/3, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (4/3, 2/3, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'ABC2':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 0, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 0, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'AAl':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            sv = np.array(shift_vector)
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos + sv))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        # Create AA tilted stacking.
        if stacking == 'AAt':
            cell = StartingFramework.as_dict()['lattice']

            # Shift the cell by the tilt angle
            a_cell = cell['a']
            b_cell = cell['b']
            c_cell = cell['c'] * 2
            alpha = cell['alpha'] - tilt_angle
            beta = cell['beta'] - tilt_angle
            gamma = cell['gamma']

            self.cellMatrix = cellpar_to_cell([a_cell, b_cell, c_cell, alpha, beta, gamma])
            self.cellParameters = np.array([a_cell, b_cell, c_cell, alpha, beta, gamma]).astype(float)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        dict_structure = stacked_structure.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = stacked_structure.formula

        dist_matrix = stacked_structure.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(stacked_structure,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_refined_structure(keep_site_properties=True)

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()

        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]

    def create_dia_structure(self,
                             BB_D41: str,
                             BB_D42: str,
                             interp_dg: str = '1',
                             d_param_base: float = 7.2,
                             print_result: bool = True,
                             **kwargs):
        """Creates a COF with DIA network.

        The DIA net is composed of two tetrapodal tetrahedical building blocks.

        Parameters
        ----------
        BB_D41 : BuildingBlock, required
            The BuildingBlock object of the tetrapodal tetrahedical Buiding Block
        BB_D42 : BuildingBlock, required
            The BuildingBlock object of the tetrapodal tetrahedical Buiding Block
        interp_dg : str, optional
            The degree of interpenetration of the framework (default is '1')
        d_param_base : float, optional
            The base value for interlayer distance in angstroms (default is 7.2)
        print_result : bool, optional
            Parameter for the control for printing the result (default is True)

        Returns
        -------
        list
            A list of strings containing:
                1. the structure name,
                2. lattice type,
                3. hall symbol of the cristaline structure,
                4. space group,
                5. number of the space group,
                6. number of operation symmetry
        """
        connectivity_error = 'Building block {} must present connectivity {} not {}'
        if BB_D41.connectivity != 4:
            self.logger.error(connectivity_error.format('A', 4, BB_D41.connectivity))
            raise BBConnectivityError(4, BB_D41.connectivity)
        if BB_D42.connectivity != 4:
            self.logger.error(connectivity_error.format('B', 4, BB_D42.connectivity))
            raise BBConnectivityError(4, BB_D42.connectivity)

        self.name = f'{BB_D41.name}-{BB_D42.name}-DIA-{interp_dg}'
        self.topology = 'DIA'
        self.staking = interp_dg
        self.dimension = 3

        self.charge = BB_D41.charge + BB_D42.charge
        self.chirality = BB_D41.chirality or BB_D42.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_D41.conector, BB_D42.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_D41.conector,
                                                                                 BB_D42.conector))

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        # Measure the base size of the building blocks
        size = np.average(BB_D41.size) + np.average(BB_D42.size)

        # Calculate the primitive cell vector assuming tetrahedical building blocks
        a_prim = np.sqrt(2)*size*np.sqrt((1 - np.cos(1.9106316646041868)))
        a_conv = np.sqrt(2)*a_prim

        # Create the primitive lattice
        self.cellMatrix = Lattice(a_conv/2*np.array(topology_info['lattice']))
        self.cellParameters = np.array([a_prim, a_prim, a_prim, 60, 60, 60]).astype(float)

        # Create the structure
        self.atom_types = []
        self.atom_labels = []
        self.atom_pos = []

        # Align and rotate the building block 1 to their respective positions
        BB_D41.align_to(topology_info['vertices'][0]['align_v'])

        # Determine the angle that alings the X[1] to one of the vertices of the tetrahedron
        vertice_pos = unit_vector(np.array([1, 0, 1]))
        Q_vertice_pos = BB_D41.get_X_points()[1][1]

        rotated_list = [
             R.from_rotvec(
                 a * unit_vector(topology_info['vertices'][0]['align_v']), degrees=False
                 ).apply(Q_vertice_pos)
             for a in np.linspace(0, 2*np.pi, 360)
             ]

        # Calculate the angle between the vertice_pos and the elements of rotated_list
        angle_list = [angle(vertice_pos, i) for i in rotated_list]

        rot_angle = np.linspace(0, 360, 360)[np.argmax(angle_list)]

        BB_D41.rotate_around(rotation_axis=np.array(topology_info['vertices'][0]['align_v']),
                             angle=rot_angle,
                             degree=True)

        BB_D41.shift(np.array(topology_info['vertices'][0]['position'])*a_conv)
        BB_D41.remove_X()

        # Add the building block 1 to the structure
        self.atom_types += BB_D41.atom_types
        self.atom_pos += BB_D41.atom_pos.tolist()
        self.atom_labels += ['C1' if i == 'C' else i for i in BB_D41.atom_labels]

        # Align and rotate the building block 1 to their respective positions
        BB_D42.align_to(topology_info['vertices'][0]['align_v'])

        # Determine the angle that alings the X[1] to one of the vertices of the tetrahedron
        vertice_pos = unit_vector(np.array([1, 0, 1]))
        Q_vertice_pos = BB_D42.get_X_points()[1][1]

        rotated_list = [
             R.from_rotvec(
                 a * unit_vector(topology_info['vertices'][0]['align_v']), degrees=False
                 ).apply(Q_vertice_pos)
             for a in np.linspace(0, 2*np.pi, 360)
             ]

        # Calculate the angle between the vertice_pos and the elements of rotated_list
        angle_list = [angle(vertice_pos, i) for i in rotated_list]

        rot_angle = np.linspace(0, 360, 360)[np.argmax(angle_list)]

        BB_D42.rotate_around(rotation_axis=np.array(topology_info['vertices'][0]['align_v']),
                             angle=rot_angle,
                             degree=True)

        BB_D42.atom_pos = -BB_D42.atom_pos

        BB_D42.shift(np.array(topology_info['vertices'][1]['position'])*a_conv)

        BB_D42.replace_X(bond_atom)
        BB_D42.remove_X()

        # Add the building block 2 to the structure
        self.atom_types += BB_D42.atom_types
        self.atom_pos += BB_D42.atom_pos.tolist()
        self.atom_labels += ['C2' if i == 'C' else i for i in BB_D42.atom_labels]

        atom_types, atom_labels, atom_pos = [], [], []
        for n_int in range(int(self.stacking)):
            int_direction = np.array([0, 1, 0]) * d_param_base * n_int

            atom_types += self.atom_types
            atom_pos += (np.array(self.atom_pos) + int_direction).tolist()
            atom_labels += self.atom_labels

        self.atom_types = atom_types
        self.atom_pos = atom_pos
        self.atom_labels = atom_labels

        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        StartingFramework.to(os.path.join(os.getcwd(), 'TESTE_DIA.cif'), fmt='cif')

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = StartingFramework.formula

        dist_matrix = StartingFramework.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(StartingFramework,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_primitive_standard_structure(keep_site_properties=True)

            dict_structure = symm.get_refined_structure(keep_site_properties=True).as_dict()

            self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)
            self.cellParameters = np.array([dict_structure['lattice']['a'],
                                            dict_structure['lattice']['b'],
                                            dict_structure['lattice']['c'],
                                            dict_structure['lattice']['alpha'],
                                            dict_structure['lattice']['beta'],
                                            dict_structure['lattice']['gamma']]).astype(float)

            self.atom_types = [i['label'] for i in dict_structure['sites']]
            self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
            self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
            self.n_atoms = len(dict_structure['sites'])
            self.composition = self.prim_structure.formula

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()

        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]

    def create_dia_a_structure(self,
                               BB_D4: str,
                               BB_L2: str,
                               interp_dg: str = '1',
                               d_param_base: float = 7.2,
                               print_result: bool = True,
                               **kwargs):
        """Creates a COF with DIA-A network.

        The DIA net is composed of two tetrapodal tetrahedical building blocks.

        Parameters
        ----------
        BB_D4 : BuildingBlock, required
            The BuildingBlock object of the tetrapodal tetrahedical Buiding Block
        BB_L2 : BuildingBlock, required
            The BuildingBlock object of the dipodal linear Buiding Block
        interp_dg : str, optional
            The degree of interpenetration of the framework (default is '1')
        d_param_base : float, optional
            The base value for interlayer distance in angstroms (default is 7.2)
        print_result : bool, optional
            Parameter for the control for printing the result (default is True)

        Returns
        -------
        list
            A list of strings containing:
                1. the structure name,
                2. lattice type,
                3. hall symbol of the cristaline structure,
                4. space group,
                5. number of the space group,
                6. number of operation symmetry
        """
        connectivity_error = 'Building block {} must present connectivity {} not {}'
        if BB_D4.connectivity != 4:
            self.logger.error(connectivity_error.format('A', 4, BB_D4.connectivity))
            raise BBConnectivityError(4, BB_D4.connectivity)
        if BB_L2.connectivity != 2:
            self.logger.error(connectivity_error.format('B', 2, BB_L2.connectivity))
            raise BBConnectivityError(2, BB_L2.connectivity)

        self.name = f'{BB_D4.name}-{BB_L2.name}-DIA_A-{interp_dg}'
        self.topology = 'DIA_A'
        self.staking = interp_dg
        self.dimension = 3

        self.charge = BB_D4.charge + BB_L2.charge
        self.chirality = BB_D4.chirality or BB_L2.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_D4.conector, BB_L2.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_D4.conector,
                                                                                 BB_L2.conector))

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        # Measure the base size of the building blocks
        size = 2 * (np.average(BB_D4.size) + np.average(BB_L2.size))

        # Calculate the primitive cell vector assuming tetrahedical building blocks
        a_prim = np.sqrt(2)*size*np.sqrt((1 - np.cos(1.9106316646041868)))
        a_conv = np.sqrt(2)*a_prim

        # Create the primitive lattice
        self.cellMatrix = Lattice(a_conv/2*np.array(topology_info['lattice']))
        self.cellParameters = np.array([a_prim, a_prim, a_prim, 60, 60, 60]).astype(float)

        # Create the structure
        self.atom_types = []
        self.atom_labels = []
        self.atom_pos = []

        # Align and rotate the building block 1 to their respective positions
        BB_D4.align_to(topology_info['vertices'][0]['align_v'])

        # Determine the angle that alings the X[1] to one of the vertices of the tetrahedron
        vertice_pos = unit_vector(np.array([1, 0, 1]))
        Q_vertice_pos = BB_D4.get_X_points()[1][1]

        rotated_list = [
             R.from_rotvec(
                 a * unit_vector(topology_info['vertices'][0]['align_v']), degrees=False
                 ).apply(Q_vertice_pos)
             for a in np.linspace(0, 2*np.pi, 360)
             ]

        # Calculate the angle between the vertice_pos and the elements of rotated_list
        angle_list = [angle(vertice_pos, i) for i in rotated_list]

        rot_angle = np.linspace(0, 360, 360)[np.argmax(angle_list)]

        BB_D4.rotate_around(rotation_axis=np.array(topology_info['vertices'][0]['align_v']),
                            angle=rot_angle,
                            degree=True)

        BB_D4.shift(np.array(topology_info['vertices'][0]['position'])*a_conv)
        BB_D4.remove_X()

        # Add the building block 1 to the structure
        self.atom_types += BB_D4.atom_types
        self.atom_pos += BB_D4.atom_pos.tolist()
        self.atom_labels += ['C1' if i == 'C' else i for i in BB_D4.atom_labels]

        # Add the building block 1 to the structure
        self.atom_types += BB_D4.atom_types
        self.atom_pos += list(-np.array(BB_D4.atom_pos) + np.array(topology_info['vertices'][1]['position'])*a_conv)
        self.atom_labels += ['C1' if i == 'C' else i for i in BB_D4.atom_labels]

        # Add the building blocks to the structure
        for edge_data in topology_info['edges']:
            # Copy the building block 2 object
            BB = copy.deepcopy(BB_L2)

            # Align, rotate and shift the building block 2 to their respective positions
            BB.align_to(edge_data['align_v'])
            BB.rotate_around(rotation_axis=edge_data['align_v'],
                             angle=edge_data['angle'])
            BB.shift(np.array(edge_data['position']) * a_conv)

            # Replace "X" the building block with the correct atom dicated by the connection group
            BB.replace_X(bond_atom)
            BB.remove_X()

            # Update the structure
            self.atom_types += BB.atom_types
            self.atom_pos += BB.atom_pos.tolist()
            self.atom_labels += ['C2' if i == 'C' else i for i in BB.atom_labels]

        atom_types, atom_labels, atom_pos = [], [], []
        for n_int in range(int(self.stacking)):
            int_direction = np.array([0, 1, 0]) * d_param_base * n_int

            atom_types += self.atom_types
            atom_pos += (np.array(self.atom_pos) + int_direction).tolist()
            atom_labels += self.atom_labels

        self.atom_types = atom_types
        self.atom_pos = atom_pos
        self.atom_labels = atom_labels

        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        StartingFramework.translate_sites(
            np.ones(len(self.atom_types)).astype(int).tolist(),
            [0, 0, 0],
            frac_coords=True,
            to_unit_cell=True
            )

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)
        self.cellParameters = np.array([dict_structure['lattice']['a'],
                                        dict_structure['lattice']['b'],
                                        dict_structure['lattice']['c'],
                                        dict_structure['lattice']['alpha'],
                                        dict_structure['lattice']['beta'],
                                        dict_structure['lattice']['gamma']]).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = StartingFramework.formula

        StartingFramework.to(os.path.join(os.getcwd(), 'TESTE_DIA-A.cif'), fmt='cif')

        dist_matrix = StartingFramework.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(StartingFramework,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_primitive_standard_structure(keep_site_properties=True)

            dict_structure = symm.get_refined_structure(keep_site_properties=True).as_dict()

            self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)
            self.cellParameters = np.array([dict_structure['lattice']['a'],
                                            dict_structure['lattice']['b'],
                                            dict_structure['lattice']['c'],
                                            dict_structure['lattice']['alpha'],
                                            dict_structure['lattice']['beta'],
                                            dict_structure['lattice']['gamma']]).astype(float)

            self.atom_types = [i['label'] for i in dict_structure['sites']]
            self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
            self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
            self.n_atoms = len(dict_structure['sites'])
            self.composition = self.prim_structure.formula

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()

        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]

    def create_bor_structure(self,
                             BB_D4: str,
                             BB_T3: str,
                             interp_dg: str = '1',
                             d_param_base: float = 7.2,
                             print_result: bool = True,
                             **kwargs):
        """Creates a COF with BOR network.

        The DIA net is composed of one tetrapodal tetrahedical building block and
        one tripodal triangular building block.

        Parameters
        ----------
        BB_D4 : BuildingBlock, required
            The BuildingBlock object of the tetrapodal tetrahedical Buiding Block
        BB_T3 : BuildingBlock, required
            The BuildingBlock object of the tripodal triangular Buiding Block
        interp_dg : str, optional
            The degree of interpenetration of the framework (default is '1')
        d_param_base : float, optional
            The base value for interlayer distance in angstroms (default is 7.2)
        print_result : bool, optional
            Parameter for the control for printing the result (default is True)

        Returns
        -------
        list
            A list of strings containing:
                1. the structure name,
                2. lattice type,
                3. hall symbol of the cristaline structure,
                4. space group,
                5. number of the space group,
                6. number of operation symmetry
        """
        connectivity_error = 'Building block {} must present connectivity {} not {}'
        if BB_D4.connectivity != 4:
            self.logger.error(connectivity_error.format('A', 4, BB_D4.connectivity))
            raise BBConnectivityError(4, BB_D4.connectivity)
        if BB_T3.connectivity != 3:
            self.logger.error(connectivity_error.format('B', 3, BB_T3.connectivity))
            raise BBConnectivityError(3, BB_T3.connectivity)

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        self.name = f'{BB_D4.name}-{BB_T3.name}-BOR-{interp_dg}'
        self.topology = 'BOR'
        self.staking = interp_dg
        self.dimension = 3

        self.charge = BB_D4.charge + BB_T3.charge
        self.chirality = BB_D4.chirality or BB_T3.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_D4.conector, BB_T3.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_D4.conector,
                                                                                 BB_T3.conector))

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        # Measure the base size of the building blocks
        d_size = (np.array(BB_D4.size).mean() + np.array(BB_T3.size).mean())

        # Calculate the primitive cell vector assuming tetrahedical building blocks
        a_conv = np.sqrt(6) * d_size

        # Create the primitive lattice
        self.cellMatrix = Lattice(a_conv * np.array(topology_info['lattice']))
        self.cellParameters = np.array([a_conv, a_conv, a_conv, 90, 90, 90]).astype(float)

        # Create the structure
        atom_types = []
        atom_labels = []
        atom_pos = []

        for D_site in topology_info['vertices']:
            D4 = BB_D4.copy()
            D4.align_to(
                np.array(D_site['align_v'])
                )

            D4.rotate_around(
                rotation_axis=D_site['align_v'],
                angle=D_site['angle'])

            D4.shift(np.array(D_site['position'])*a_conv)

            atom_types += D4.atom_types
            atom_pos += D4.atom_pos.tolist()
            atom_labels += D4.atom_labels.tolist()

        # Translate all atoms to inside the cell
        for i, pos in enumerate(atom_pos):
            for j, coord in enumerate(pos):
                if coord < 0:
                    atom_pos[i][j] += a_conv

        X_pos = [atom_pos[i] for i in np.where(np.array(atom_types) == 'X')[0]]

        T_site = topology_info['edges'][0]

        _, X = BB_T3.get_X_points()
        BB_T3.rotate_around([0, 0, 1], T_site['angle'], True)

        R_matrix = rotation_matrix_from_vectors([0, 0, 1],
                                                T_site['align_v'])

        BB_T3.atom_pos = np.dot(BB_T3.atom_pos, R_matrix.T)

        BB_T3.replace_X('O')

        # Get the 3 atoms that are closer to T_site['position'])*a_conv
        X_pos_temp = sorted(X_pos, key=lambda x: np.linalg.norm(x - np.array(T_site['position'])*a_conv))

        X_center = np.array(X_pos_temp[:3]).mean(axis=0)

        BB_T3.shift(X_center)

        atom_types += BB_T3.atom_types
        atom_pos += BB_T3.atom_pos.tolist()
        atom_labels += BB_T3.atom_labels.tolist()

        T4 = BB_T3.copy()
        T4.rotate_around([0, 0, 1], 180, True)

        atom_types += T4.atom_types
        atom_pos += T4.atom_pos.tolist()
        atom_labels += T4.atom_labels.tolist()

        T2 = BB_T3.copy()
        T2.rotate_around([0, 0, 1], 90, True)
        T2.rotate_around([1, 0, 0], -90, True)

        atom_types += T2.atom_types
        atom_pos += T2.atom_pos.tolist()
        atom_labels += T2.atom_labels.tolist()

        T3 = BB_T3.copy()
        T3.rotate_around([0, 0, 1], -90, True)

        T3.atom_pos *= np.array([1, 1, -1])

        atom_types += T3.atom_types
        atom_pos += T3.atom_pos.tolist()
        atom_labels += T3.atom_labels.tolist()

        # Translate all atoms to inside the cell
        for i, pos in enumerate(atom_pos):
            for j, coord in enumerate(pos):
                if coord < 0:
                    atom_pos[i][j] += a_conv

        # Remove the X atoms from the list
        X_index = np.where(np.array(atom_types) == 'X')[0]

        self.atom_types = [atom_types[i] for i in range(len(atom_types)) if i not in X_index]
        self.atom_pos = [atom_pos[i] for i in range(len(atom_pos)) if i not in X_index]
        self.atom_labels = [atom_labels[i] for i in range(len(atom_labels)) if i not in X_index]

        atom_types, atom_labels, atom_pos = [], [], []
        for n_int in range(int(self.stacking)):
            int_direction = np.array([0, 1, 0]) * d_param_base * n_int

            atom_types += self.atom_types
            atom_pos += (np.array(self.atom_pos) + int_direction).tolist()
            atom_labels += self.atom_labels

        self.atom_types = atom_types
        self.atom_pos = atom_pos
        self.atom_labels = atom_labels

        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        StartingFramework.translate_sites(
            np.ones(len(self.atom_types)).astype(int).tolist(),
            [0, 0, 0],
            frac_coords=True,
            to_unit_cell=True
            )

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)
        self.cellParameters = np.array([dict_structure['lattice']['a'],
                                        dict_structure['lattice']['b'],
                                        dict_structure['lattice']['c'],
                                        dict_structure['lattice']['alpha'],
                                        dict_structure['lattice']['beta'],
                                        dict_structure['lattice']['gamma']]).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = StartingFramework.formula

        StartingFramework.to('TESTE_BOR.cif', fmt='cif')

        dist_matrix = StartingFramework.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(StartingFramework,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_primitive_standard_structure(keep_site_properties=True)

            dict_structure = symm.get_refined_structure(keep_site_properties=True).as_dict()

            self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)
            self.cellParameters = np.array([dict_structure['lattice']['a'],
                                            dict_structure['lattice']['b'],
                                            dict_structure['lattice']['c'],
                                            dict_structure['lattice']['alpha'],
                                            dict_structure['lattice']['beta'],
                                            dict_structure['lattice']['gamma']]).astype(float)

            self.atom_types = [i['label'] for i in dict_structure['sites']]
            self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
            self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
            self.n_atoms = len(dict_structure['sites'])
            self.composition = self.prim_structure.formula

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()

        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]


    def create_hxl_structure(self,
                             BB_H6_A:str,
                             BB_H6_B:str,
                             stacking: str = 'AA',
                             print_result: bool = True,
                             slab: float = 10.0,
                             shift_vector: list = (1.0, 1.0, 0),
                             tilt_angle: float = 5.0):
        connectivity_error = 'Building block {} must present connectivity {} not {}'
        if BB_H6_A.connectivity != 6:
            self.logger.error(connectivity_error.format('A', 6, BB_H6_A.connectivity))
            raise BBConnectivityError(6, BB_H6_A.connectivity)
        if BB_H6_B.connectivity != 6:
            self.logger.error(connectivity_error.format('B', 6, BB_H6_B.connectivity))
            raise BBConnectivityError(6, BB_H6_B.connectivity)

        self.name = f'{BB_H6_A.name}-{BB_H6_B.name}-HXL-{stacking}'
        self.topology = 'HXL'
        self.staking = stacking
        self.dimension = 2

        self.charge = BB_H6_A.charge + BB_H6_B.charge
        self.chirality = BB_H6_A.chirality or BB_H6_B.chirality

        self.logger.debug(f'Starting the creation of {self.name}')

        # Detect the bond atom from the connection groups type
        bond_atom = get_bond_atom(BB_H6_A.conector, BB_H6_B.conector)

        self.logger.debug('{} detected as bond atom for groups {} and {}'.format(bond_atom,
                                                                                 BB_H6_A.conector,
                                                                                 BB_H6_B.conector))

        # Replace "X" the building block
        BB_H6_A.replace_X(bond_atom)

        # Remove the "X" atoms from the the building block
        BB_H6_A.remove_X()
        BB_H6_B.remove_X()

        # Get the topology information
        topology_info = TOPOLOGY_DICT[self.topology]

        # Measure the base size of the building blocks
        size = BB_H6_A.size[0] + BB_H6_B.size[0]

        # Calculate the delta size to add to the c parameter
        delta_a = abs(max(np.transpose(BB_H6_A.atom_pos)[2])) + abs(min(np.transpose(BB_H6_A.atom_pos)[2]))
        delta_b = abs(max(np.transpose(BB_H6_B.atom_pos)[2])) + abs(min(np.transpose(BB_H6_B.atom_pos)[2]))

        delta_max = max([delta_a, delta_b])

        # Calculate the cell parameters
        a = topology_info['a'] * size
        b = topology_info['b'] * size
        c = topology_info['c'] + delta_max
        alpha = topology_info['alpha']
        beta = topology_info['beta']
        gamma = topology_info['gamma']

        if self.stacking == 'A':
            c = slab

        # Create the lattice
        self.cellMatrix = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        self.cellParameters = np.array([a, b, c, alpha, beta, gamma]).astype(float)

        # Create the structure
        self.atom_types = []
        self.atom_labels = []
        self.atom_pos = []

        # Add the A1 building blocks to the structure
        vertice_data = topology_info['vertices'][0]
        self.atom_types += BB_H6_A.atom_types
        vertice_pos = np.array(vertice_data['position'])*a

        R_Matrix = R.from_euler('z',
                                vertice_data['angle'],
                                degrees=True).as_matrix()

        rotated_pos = np.dot(BB_H6_A.atom_pos, R_Matrix) + vertice_pos
        self.atom_pos += rotated_pos.tolist()

        self.atom_labels += ['C1' if i == 'C' else i for i in BB_H6_A.atom_labels]

        # Add the A2 building block to the structure
        vertice_data = topology_info['vertices'][1]
        self.atom_types += BB_H6_B.atom_types
        vertice_pos = np.array(vertice_data['position'])*a

            R_Matrix = R.from_euler('z',
                                    edge_data['angle'],
                                    degrees=True).as_matrix()

            rotated_pos = np.dot(BB_H6_B.atom_pos, R_Matrix) + edge_pos
            self.atom_pos += rotated_pos.tolist()

            self.atom_labels += ['C2' if i == 'C' else i for i in BB_H6_B.atom_labels]

        StartingFramework = Structure(
            self.cellMatrix,
            self.atom_types,
            self.atom_pos,
            coords_are_cartesian=True,
            site_properties={'source': self.atom_labels}
        ).get_sorted_structure()

        # Translates the structure to the center of the cell
        StartingFramework.translate_sites(
            range(len(StartingFramework.as_dict()['sites'])),
            [0, 0, 0.5],
            frac_coords=True,
            to_unit_cell=True
        )

        dict_structure = StartingFramework.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]

        if stacking == 'A' or stacking == 'AA':
            stacked_structure = StartingFramework

        if stacking == 'AB1':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [2/3, 1/3, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'AB2':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [1/2, 0, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [1/2, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        if stacking == 'ABC1':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (2/3, 1/3, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (4/3, 2/3, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'ABC2':
            self.cellMatrix *= (1, 1, 3)
            self.cellParameters *= (1, 1, 3, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            _, B_list, C_list = np.split(np.arange(len(self.atom_types)), 3)

            # Translate the second sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                (1/3, 0, 1/3),
                frac_coords=True,
                to_unit_cell=True
                )

            # Translate the third sheet by the vector (2/3, 1/3, 0) to generate the B positions
            stacked_structure.translate_sites(
                C_list,
                (2/3, 0, 2/3),
                frac_coords=True,
                to_unit_cell=True
            )

        if stacking == 'AAl':
            self.cellMatrix *= (1, 1, 2)
            self.cellParameters *= (1, 1, 2, 1, 1, 1)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            sv = np.array(shift_vector)
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos + sv))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        # Create AA tilted stacking.
        if stacking == 'AAt':
            cell = StartingFramework.as_dict()['lattice']

            # Shift the cell by the tilt angle
            a_cell = cell['a']
            b_cell = cell['b']
            c_cell = cell['c'] * 2
            alpha = cell['alpha'] - tilt_angle
            beta = cell['beta'] - tilt_angle
            gamma = cell['gamma']

            self.cellMatrix = cellpar_to_cell([a_cell, b_cell, c_cell, alpha, beta, gamma])
            self.cellParameters = np.array([a_cell, b_cell, c_cell, alpha, beta, gamma]).astype(float)

            self.atom_types = np.concatenate((self.atom_types, self.atom_types))
            self.atom_pos = np.concatenate((self.atom_pos, self.atom_pos))
            self.atom_labels = np.concatenate((self.atom_labels, self.atom_labels))

            stacked_structure = Structure(
                self.cellMatrix,
                self.atom_types,
                self.atom_pos,
                coords_are_cartesian=True,
                site_properties={'source': self.atom_labels}
            )

            # Get the index of the atoms in the second sheet
            B_list = np.split(np.arange(len(self.atom_types)), 2)[1]

            # Translate the second sheet by the vector [2/3, 1/3, 0.5] to generate the B positions
            stacked_structure.translate_sites(
                B_list,
                [0, 0, 0.5],
                frac_coords=True,
                to_unit_cell=True
                )

        dict_structure = stacked_structure.as_dict()

        self.cellMatrix = np.array(dict_structure['lattice']['matrix']).astype(float)

        self.atom_types = [i['label'] for i in dict_structure['sites']]
        self.atom_pos = [i['xyz'] for i in dict_structure['sites']]
        self.atom_labels = [i['properties']['source'] for i in dict_structure['sites']]
        self.n_atoms = len(dict_structure['sites'])
        self.composition = stacked_structure.formula

        dist_matrix = stacked_structure.distance_matrix

        # Check if there are any atoms closer than 0.8 A
        for i in range(len(dist_matrix)):
            for j in range(i+1, len(dist_matrix)):
                if dist_matrix[i][j] < self.dist_threshold:
                    raise BondLenghError(i, j, dist_matrix[i][j], self.dist_threshold)

        # Get the simmetry information of the generated structure
        symm = SpacegroupAnalyzer(stacked_structure,
                                  symprec=self.symm_tol,
                                  angle_tolerance=self.angle_tol)

        try:
            self.prim_structure = symm.get_refined_structure(keep_site_properties=True)

            self.logger.debug(self.prim_structure)

            self.lattice_type = symm.get_lattice_type()
            self.space_group = symm.get_space_group_symbol()
            self.space_group_n = symm.get_space_group_number()

            symm_op = symm.get_point_group_operations()
            self.hall = symm.get_hall()
        except Exception as e:
            self.logger.exception(e)

            self.lattice_type = 'Triclinic'
            self.space_group = 'P1'
            self.space_group_n = '1'

            symm_op = [1]
            self.hall = 'P 1'

        symm_text = get_framework_symm_text(self.name,
                                            str(self.lattice_type),
                                            str(self.hall[0:2]),
                                            str(self.space_group),
                                            str(self.space_group_n),
                                            len(symm_op))

        self.logger.info(symm_text)

        return [self.name,
                str(self.lattice_type),
                str(self.hall[0:2]),
                str(self.space_group),
                str(self.space_group_n),
                len(symm_op)]