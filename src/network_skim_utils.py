# ============================================================================
# network_skim_utils.py
# ============================================================================
# 模块职责：从MATSim格式的路网文件构建ActivitySim所需的Skim矩阵
# 包括：
# - 解析MATSim network.xml文件
# - 构建NetworkX有向图
# - 将TAZ质心映射到路网节点
# - 计算OD对的最短路径时间和距离
# - 输出OpenMatrix (OMX)格式的Skim矩阵
# ============================================================================

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import Point

try:
    import openmatrix as omx
except ImportError:
    warnings.warn("openmatrix未安装，将使用替代方案保存skim矩阵")
    omx = None

from .io_utils import logger, ensure_dir, ProgressCallback
from .shapefile_utils import transform_to_wgs84, CRS_WGS84


# ----------------------------------------------------------------------------
# MATSim network.xml 解析
# ----------------------------------------------------------------------------

class MATSimNetwork:
    """
    MATSim路网解析器

    属性:
        nodes: 节点DataFrame (node_id, x, y)
        links: 路段DataFrame (link_id, from_node, to_node, length, freespeed, capacity, modes)
        graph: NetworkX有向图
    """

    def __init__(self, network_file: Union[str, Path]):
        """
        初始化MATSim路网解析器

        参数:
            network_file: MATSim network.xml文件路径
        """
        self.network_file = Path(network_file)
        self.nodes = None
        self.links = None
        self.graph = None
        self.crs = None  # 路网坐标系

        if not self.network_file.exists():
            raise FileNotFoundError(f"路网文件不存在: {network_file}")

    def parse(self, target_crs: str = CRS_WGS84) -> None:
        """
        解析MATSim network.xml文件

        参数:
            target_crs: 目标坐标系，默认WGS84

        说明:
            - 提取nodes和links数据
            - 检测并转换坐标系统
        """
        logger.info(f"正在解析MATSim路网文件: {self.network_file}")

        # 解析XML
        tree = ET.parse(self.network_file)
        root = tree.getroot()

        # 提取nodes
        nodes_data = []
        nodes_elem = root.find('nodes')
        if nodes_elem is not None:
            for node in nodes_elem.findall('node'):
                nodes_data.append({
                    'node_id': node.get('id'),
                    'x': float(node.get('x')),
                    'y': float(node.get('y'))
                })

        self.nodes = pd.DataFrame(nodes_data)
        logger.info(f"已读取 {len(self.nodes)} 个路网节点")

        # 提取links
        links_data = []
        links_elem = root.find('links')
        if links_elem is not None:
            for link in links_elem.findall('link'):
                # 提取基本属性
                link_dict = {
                    'link_id': link.get('id'),
                    'from_node': link.get('from'),
                    'to_node': link.get('to'),
                    'length': float(link.get('length', 0)),
                    'freespeed': float(link.get('freespeed', 13.89)),  # 默认50km/h = 13.89m/s
                    'capacity': float(link.get('capacity', 1000)),
                    'permlanes': float(link.get('permlanes', 1)),
                    'modes': link.get('modes', 'car')
                }
                links_data.append(link_dict)

        self.links = pd.DataFrame(links_data)
        logger.info(f"已读取 {len(self.links)} 条路网路段")

        # 检测坐标系
        self._detect_crs()

        # 坐标转换
        if self.crs != target_crs:
            self._transform_coordinates(target_crs)

        # 计算出行时间
        self._compute_travel_times()

    def _detect_crs(self) -> None:
        """
        检测路网坐标系

        说明:
            - 通过坐标数值范围推断CRS
            - MATSim路网可能使用投影坐标系（米）或地理坐标系（度）
        """
        x_range = self.nodes['x'].max() - self.nodes['x'].min()
        y_range = self.nodes['y'].max() - self.nodes['y'].min()

        # 如果坐标范围在-180到180之间，可能是经纬度
        if (self.nodes['x'].min() >= -180 and self.nodes['x'].max() <= 180 and
                self.nodes['y'].min() >= -90 and self.nodes['y'].max() <= 90):
            self.crs = CRS_WGS84
            logger.info("检测到路网使用WGS84地理坐标系")
        else:
            # 假设为投影坐标系（米）
            # 这里简化处理，实际应从XML中读取CRS信息
            self.crs = "EPSG:3857"  # Web Mercator作为默认投影
            logger.info(f"检测到路网使用投影坐标系（假设为{self.crs}）")

    def _transform_coordinates(self, target_crs: str) -> None:
        """
        转换路网节点坐标到目标CRS

        参数:
            target_crs: 目标坐标系
        """
        logger.info(f"正在转换路网坐标系: {self.crs} -> {target_crs}")

        # 创建GeoDataFrame进行坐标转换
        geometry = gpd.points_from_xy(self.nodes['x'], self.nodes['y'])
        gdf = gpd.GeoDataFrame(self.nodes, geometry=geometry, crs=self.crs)
        gdf = gdf.to_crs(target_crs)

        # 更新坐标
        self.nodes['x'] = gdf.geometry.x
        self.nodes['y'] = gdf.geometry.y
        self.crs = target_crs

    def _compute_travel_times(self) -> None:
        """
        计算路段的自由流出行时间

        说明:
            - travel_time (秒) = length (米) / freespeed (米/秒)
        """
        self.links['travel_time'] = self.links['length'] / self.links['freespeed']
        self.links['travel_time_minutes'] = self.links['travel_time'] / 60.0

    def build_graph(self, mode: str = 'car') -> nx.DiGraph:
        """
        构建NetworkX有向图

        参数:
            mode: 交通方式，用于过滤路段

        返回:
            NetworkX有向图

        说明:
            - 边权重使用travel_time（秒）
            - 仅包含允许指定mode通行的路段
        """
        logger.info(f"正在构建{mode}模式的路网图...")

        # 过滤允许指定mode的路段
        mode_links = self.links[self.links['modes'].str.contains(mode, na=False)]

        # 创建有向图
        G = nx.DiGraph()

        # 添加节点
        for _, node in self.nodes.iterrows():
            G.add_node(
                node['node_id'],
                x=node['x'],
                y=node['y']
            )

        # 添加边
        for _, link in mode_links.iterrows():
            G.add_edge(
                link['from_node'],
                link['to_node'],
                link_id=link['link_id'],
                length=link['length'],
                time=link['travel_time'],
                freespeed=link['freespeed'],
                capacity=link['capacity']
            )

        logger.info(f"路网图包含 {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")

        # 检查连通性
        if not nx.is_weakly_connected(G):
            logger.warning("路网图不是弱连通的，可能存在孤立子图")
            # 找到最大连通分量
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            logger.info(f"最大连通分量包含 {len(largest_cc)} 个节点")

        self.graph = G
        return G

    def get_node_coordinates(self) -> pd.DataFrame:
        """
        获取所有节点坐标

        返回:
            包含node_id, x, y的DataFrame
        """
        return self.nodes[['node_id', 'x', 'y']].copy()


# ----------------------------------------------------------------------------
# TAZ质心到路网节点的映射
# ----------------------------------------------------------------------------

def map_centroids_to_network(
        zone_centroids: gpd.GeoDataFrame,
        network_nodes: pd.DataFrame,
        max_distance_km: float = 5.0
) -> pd.DataFrame:
    """
    将TAZ质心映射到最近的路网节点

    参数:
        zone_centroids: TAZ质心GeoDataFrame，包含zone_id和geometry(Point)
        network_nodes: 路网节点DataFrame，包含node_id, x, y
        max_distance_km: 最大映射距离（公里），超过此距离将警告

    返回:
        映射表DataFrame，包含zone_id, node_id, distance_m

    说明:
        - 使用KDTree加速最近邻搜索
        - 所有坐标必须在相同的CRS中
    """
    logger.info("正在将TAZ质心映射到路网节点...")

    # 确保zone_centroids使用WGS84
    zone_centroids = transform_to_wgs84(zone_centroids)

    # 提取质心坐标
    centroids_coords = np.array([
        [point.x, point.y] for point in zone_centroids.geometry
    ])

    # 提取路网节点坐标
    nodes_coords = network_nodes[['x', 'y']].values

    # 构建KDTree
    tree = cKDTree(nodes_coords)

    # 查询最近节点
    distances, indices = tree.query(centroids_coords, k=1)

    # 构建映射表
    mapping = pd.DataFrame({
        'zone_id': zone_centroids['zone_id'].values,
        'node_id': network_nodes.iloc[indices]['node_id'].values,
        'distance_m': distances * 111000  # 度转米（粗略估算）
    })

    # 检查映射距离
    max_dist_m = max_distance_km * 1000
    far_zones = mapping[mapping['distance_m'] > max_dist_m]
    if len(far_zones) > 0:
        logger.warning(
            f"{len(far_zones)}个TAZ质心距离最近路网节点超过{max_distance_km}km，"
            f"最大距离: {mapping['distance_m'].max() / 1000:.2f}km"
        )

    logger.info(f"成功映射 {len(mapping)} 个TAZ到路网节点")

    return mapping


# ----------------------------------------------------------------------------
# Skim矩阵计算
# ----------------------------------------------------------------------------

def compute_skim_matrix(
        graph: nx.DiGraph,
        zone_mapping: pd.DataFrame,
        weight: str = 'time',
        progress_callback: Optional[ProgressCallback] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算OD Skim矩阵

    参数:
        graph: NetworkX路网图
        zone_mapping: TAZ到节点的映射表 (zone_id, node_id)
        weight: 边权重属性名 ('time' 或 'length')
        progress_callback: 进度回调对象

    返回:
        (skim_matrix, od_matrix) 元组
        - skim_matrix: 长格式DataFrame (origin_zone, dest_zone, value)
        - od_matrix: 方阵DataFrame (行=origin, 列=dest)

    说明:
        - 使用Dijkstra算法计算最短路径
        - 对于不可达的OD对，值设为-1或一个大数
        - 计算时间复杂度为O(n²logn)，大规模路网可能耗时较长
    """
    logger.info(f"正在计算Skim矩阵 (权重: {weight})...")

    zones = zone_mapping['zone_id'].unique()
    n_zones = len(zones)

    # 初始化矩阵
    skim_dict = {}

    # 为每个起点计算到所有终点的最短路径
    for i, origin_zone in enumerate(zones):
        if progress_callback:
            progress_callback.update(
                step=i + 1,
                message=f"正在计算zone {origin_zone}的最短路径...",
                progress=(i + 1) / n_zones
            )

        # 获取起点对应的路网节点
        origin_node = zone_mapping[
            zone_mapping['zone_id'] == origin_zone
            ]['node_id'].iloc[0]

        # 检查节点是否在图中
        if origin_node not in graph:
            logger.warning(f"Zone {origin_zone}的节点{origin_node}不在路网图中")
            continue

        # 计算从origin_node到所有其他节点的最短路径
        try:
            lengths = nx.single_source_dijkstra_path_length(
                graph, origin_node, weight=weight
            )
        except nx.NetworkXError as e:
            logger.warning(f"计算zone {origin_zone}的最短路径失败: {e}")
            continue

        # 对每个终点，查找对应节点并获取距离
        for dest_zone in zones:
            dest_node = zone_mapping[
                zone_mapping['zone_id'] == dest_zone
                ]['node_id'].iloc[0]

            if dest_node in lengths:
                skim_dict[(origin_zone, dest_zone)] = lengths[dest_node]
            else:
                # 不可达，设为-1
                skim_dict[(origin_zone, dest_zone)] = -1

    # 转换为DataFrame
    skim_list = [
        {'origin_zone': oz, 'dest_zone': dz, weight: value}
        for (oz, dz), value in skim_dict.items()
    ]
    skim_df = pd.DataFrame(skim_list)

    # 创建方阵
    od_matrix = skim_df.pivot(
        index='origin_zone',
        columns='dest_zone',
        values=weight
    )

    # 填充缺失值（不可达）
    od_matrix = od_matrix.fillna(-1)

    logger.info(f"Skim矩阵计算完成: {n_zones} x {n_zones}")

    return skim_df, od_matrix


def compute_multiple_skims(
        graph: nx.DiGraph,
        zone_mapping: pd.DataFrame,
        weights: List[str] = ['time', 'length'],
        progress_callback: Optional[ProgressCallback] = None
) -> Dict[str, pd.DataFrame]:
    """
    计算多个权重的Skim矩阵

    参数:
        graph: NetworkX路网图
        zone_mapping: TAZ到节点的映射表
        weights: 权重列表，如 ['time', 'length']
        progress_callback: 进度回调

    返回:
        字典，键为权重名称，值为方阵DataFrame
    """
    skims = {}

    for weight in weights:
        logger.info(f"正在计算 {weight} 矩阵...")
        _, od_matrix = compute_skim_matrix(
            graph, zone_mapping, weight, progress_callback
        )
        skims[weight] = od_matrix

    return skims


# ----------------------------------------------------------------------------
# Skim矩阵后处理
# ----------------------------------------------------------------------------

def postprocess_skim_matrix(
        skim_matrix: pd.DataFrame,
        fill_unreachable: float = 999999,
        apply_symmetry: bool = False
) -> pd.DataFrame:
    """
    后处理Skim矩阵

    参数:
        skim_matrix: 原始Skim方阵
        fill_unreachable: 用于填充不可达OD对的值
        apply_symmetry: 是否强制对称（取往返平均值）

    返回:
        处理后的Skim矩阵
    """
    matrix = skim_matrix.copy()

    # 填充不可达（-1）为大数
    matrix = matrix.replace(-1, fill_unreachable)

    # 对角线设为0（同一zone内部出行）
    np.fill_diagonal(matrix.values, 0)

    # 可选：强制对称
    if apply_symmetry:
        matrix = (matrix + matrix.T) / 2

    return matrix


def convert_time_units(
        time_matrix: pd.DataFrame,
        from_unit: str = 'seconds',
        to_unit: str = 'minutes'
) -> pd.DataFrame:
    """
    转换时间矩阵的单位

    参数:
        time_matrix: 时间矩阵
        from_unit: 原始单位 ('seconds', 'minutes', 'hours')
        to_unit: 目标单位

    返回:
        转换后的矩阵
    """
    conversion_factors = {
        ('seconds', 'minutes'): 1 / 60,
        ('seconds', 'hours'): 1 / 3600,
        ('minutes', 'seconds'): 60,
        ('minutes', 'hours'): 1 / 60,
        ('hours', 'seconds'): 3600,
        ('hours', 'minutes'): 60,
    }

    if from_unit == to_unit:
        return time_matrix

    factor = conversion_factors.get((from_unit, to_unit))
    if factor is None:
        raise ValueError(f"不支持的单位转换: {from_unit} -> {to_unit}")

    return time_matrix * factor


def convert_distance_units(
        dist_matrix: pd.DataFrame,
        from_unit: str = 'meters',
        to_unit: str = 'kilometers'
) -> pd.DataFrame:
    """
    转换距离矩阵的单位

    参数:
        dist_matrix: 距离矩阵
        from_unit: 原始单位 ('meters', 'kilometers', 'miles')
        to_unit: 目标单位

    返回:
        转换后的矩阵
    """
    conversion_factors = {
        ('meters', 'kilometers'): 1 / 1000,
        ('meters', 'miles'): 1 / 1609.34,
        ('kilometers', 'meters'): 1000,
        ('kilometers', 'miles'): 1 / 1.60934,
        ('miles', 'meters'): 1609.34,
        ('miles', 'kilometers'): 1.60934,
    }

    if from_unit == to_unit:
        return dist_matrix

    factor = conversion_factors.get((from_unit, to_unit))
    if factor is None:
        raise ValueError(f"不支持的单位转换: {from_unit} -> {to_unit}")

    return dist_matrix * factor


# ----------------------------------------------------------------------------
# OMX文件写入
# ----------------------------------------------------------------------------

def save_skims_to_omx(
        skims: Dict[str, pd.DataFrame],
        output_file: Union[str, Path],
        zone_ids: Optional[List[int]] = None
) -> Path:
    """
    将Skim矩阵保存为OMX格式

    参数:
        skims: 字典，键为矩阵名称（如'TIME', 'DIST'），值为方阵DataFrame
        output_file: 输出OMX文件路径
        zone_ids: zone ID列表（如为None则从矩阵索引获取）

    返回:
        输出文件路径

    说明:
        - OMX是交通建模常用的HDF5格式矩阵文件
        - ActivitySim可以直接读取OMX格式的skim文件
    """
    output_file = Path(output_file)
    ensure_dir(output_file.parent)

    if omx is None:
        logger.warning("openmatrix未安装，将使用CSV格式保存")
        return _save_skims_to_csv(skims, output_file.with_suffix('.csv'))

    logger.info(f"正在保存Skim矩阵到OMX文件: {output_file}")

    # 获取zone IDs
    if zone_ids is None:
        # 从第一个矩阵获取
        first_matrix = list(skims.values())[0]
        zone_ids = first_matrix.index.tolist()

    # 创建OMX文件
    with omx.open_file(str(output_file), 'w') as omx_file:
        # 设置维度（zone映射）
        omx_file.create_mapping('zone_id', zone_ids)

        # 写入每个矩阵
        for name, matrix in skims.items():
            # 确保矩阵按zone_ids顺序排列
            matrix = matrix.reindex(index=zone_ids, columns=zone_ids)

            # 转换为numpy数组
            data = matrix.values.astype(np.float32)

            # 写入OMX
            omx_file[name] = data

            logger.info(f"已写入矩阵: {name} ({data.shape})")

    logger.info(f"OMX文件保存成功: {output_file}")
    return output_file


def _save_skims_to_csv(
        skims: Dict[str, pd.DataFrame],
        output_file: Path
) -> Path:
    """
    将Skim矩阵保存为CSV格式（备选方案）

    参数:
        skims: Skim矩阵字典
        output_file: 输出CSV文件路径

    返回:
        输出文件路径
    """
    logger.info(f"正在保存Skim矩阵到CSV: {output_file}")

    # 合并所有矩阵为长格式
    all_skims = []

    for name, matrix in skims.items():
        # 转为长格式
        long_df = matrix.stack().reset_index()
        long_df.columns = ['origin_zone', 'dest_zone', name]
        all_skims.append(long_df)

    # 合并
    result = all_skims[0]
    for df in all_skims[1:]:
        result = result.merge(
            df,
            on=['origin_zone', 'dest_zone'],
            how='outer'
        )

    result.to_csv(output_file, index=False)
    logger.info(f"CSV文件保存成功: {output_file}")

    return output_file


# ----------------------------------------------------------------------------
# 完整流程
# ----------------------------------------------------------------------------

def build_skims_from_matsim_network(
        network_file: Union[str, Path],
        zone_centroids: gpd.GeoDataFrame,
        output_file: Union[str, Path],
        mode: str = 'car',
        progress_callback: Optional[ProgressCallback] = None
) -> Path:
    """
    从MATSim路网文件构建ActivitySim Skim矩阵的完整流程

    参数:
        network_file: MATSim network.xml文件路径
        zone_centroids: TAZ质心GeoDataFrame (zone_id, geometry)
        output_file: 输出OMX文件路径
        mode: 交通方式
        progress_callback: 进度回调

    返回:
        输出文件路径

    流程:
        1. 解析MATSim路网
        2. 构建NetworkX图
        3. 映射TAZ质心到路网节点
        4. 计算时间和距离Skim矩阵
        5. 后处理并保存为OMX
    """
    logger.info("开始构建Skim矩阵...")

    # 1. 解析MATSim路网
    if progress_callback:
        progress_callback.update(1, "正在解析MATSim路网...")

    network = MATSimNetwork(network_file)
    network.parse(target_crs=CRS_WGS84)

    # 2. 构建NetworkX图
    if progress_callback:
        progress_callback.update(2, "正在构建路网图...")

    graph = network.build_graph(mode=mode)

    # 3. 映射TAZ质心到路网节点
    if progress_callback:
        progress_callback.update(3, "正在映射TAZ到路网节点...")

    zone_mapping = map_centroids_to_network(
        zone_centroids,
        network.get_node_coordinates()
    )

    # 4. 计算Skim矩阵
    if progress_callback:
        progress_callback.update(4, "正在计算出行时间矩阵...")

    # 时间矩阵（秒）
    _, time_matrix_sec = compute_skim_matrix(
        graph, zone_mapping, weight='time', progress_callback=None
    )

    # 转换为分钟
    time_matrix = convert_time_units(time_matrix_sec, 'seconds', 'minutes')

    if progress_callback:
        progress_callback.update(5, "正在计算出行距离矩阵...")

    # 距离矩阵（米）
    _, dist_matrix_m = compute_skim_matrix(
        graph, zone_mapping, weight='length', progress_callback=None
    )

    # 转换为公里
    dist_matrix = convert_distance_units(dist_matrix_m, 'meters', 'kilometers')

    # 5. 后处理
    if progress_callback:
        progress_callback.update(6, "正在后处理Skim矩阵...")

    time_matrix = postprocess_skim_matrix(time_matrix, fill_unreachable=9999)
    dist_matrix = postprocess_skim_matrix(dist_matrix, fill_unreachable=9999)

    # 6. 保存为OMX
    if progress_callback:
        progress_callback.update(7, "正在保存Skim矩阵...")

    skims = {
        'TIME': time_matrix,
        'DIST': dist_matrix
    }

    output_path = save_skims_to_omx(skims, output_file)

    if progress_callback:
        progress_callback.update(8, "Skim矩阵构建完成！")

    logger.info("Skim矩阵构建完成")
    return output_path


# ----------------------------------------------------------------------------
# 简化版Skim构建（用于小规模或测试）
# ----------------------------------------------------------------------------

def build_simple_skims(
        zone_centroids: gpd.GeoDataFrame,
        output_file: Union[str, Path],
        average_speed_kmh: float = 40.0
) -> Path:
    """
    基于欧氏距离构建简化版Skim矩阵（无需路网）

    参数:
        zone_centroids: TAZ质心GeoDataFrame
        output_file: 输出OMX文件路径
        average_speed_kmh: 假设的平均速度（公里/小时）

    返回:
        输出文件路径

    说明:
        - 使用TAZ质心间的欧氏距离
        - 时间 = 距离 / 速度
        - 仅用于快速测试或无路网数据的场景
    """
    logger.info("正在构建简化版Skim矩阵（基于欧氏距离）...")

    # 确保WGS84坐标
    zone_centroids = transform_to_wgs84(zone_centroids)

    zones = zone_centroids['zone_id'].values
    n_zones = len(zones)

    # 初始化矩阵
    dist_matrix = np.zeros((n_zones, n_zones))

    # 计算欧氏距离
    coords = np.array([
        [pt.x, pt.y] for pt in zone_centroids.geometry
    ])

    for i in range(n_zones):
        for j in range(n_zones):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                # 简化的经纬度距离计算（Haversine公式的近似）
                dx = (coords[j, 0] - coords[i, 0]) * 111.0  # 1度经度 ≈ 111km
                dy = (coords[j, 1] - coords[i, 1]) * 111.0  # 1度纬度 ≈ 111km
                dist_matrix[i, j] = np.sqrt(dx ** 2 + dy ** 2)

    # 转为DataFrame
    dist_df = pd.DataFrame(dist_matrix, index=zones, columns=zones)

    # 计算时间矩阵（分钟）
    time_df = dist_df / average_speed_kmh * 60

    # 保存
    skims = {
        'TIME': time_df,
        'DIST': dist_df
    }

    output_path = save_skims_to_omx(skims, output_file, zone_ids=zones.tolist())

    logger.info(f"简化版Skim矩阵构建完成: {output_path}")
    return output_path