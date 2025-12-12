# ============================================================================
# osm_poi_utils.py
# ============================================================================
# 模块职责：使用OSMnx下载和处理OpenStreetMap数据
# 包括：
# - 下载研究区域内的建筑数据
# - 下载POI（兴趣点）数据
# - 按类型分类建筑和POI（办公、商业、住宅、教育、医疗等）
# - 计算建筑面积
# - 为土地利用变量构建提供基础数据
# ============================================================================

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, box
import osmnx as ox

from .io_utils import logger, ensure_dir
from .shapefile_utils import (
    transform_to_wgs84, transform_to_projected,
    detect_optimal_crs, get_study_area_polygon, CRS_WGS84
)

# 抑制OSMnx的一些警告
warnings.filterwarnings('ignore', category=FutureWarning)

# ----------------------------------------------------------------------------
# OSM标签分类定义
# ----------------------------------------------------------------------------
# 以下定义了各类建筑/POI的OSM标签，用于分类和就业岗位估算

# 办公类建筑/POI标签
OFFICE_TAGS = {
    'building': ['office', 'commercial'],
    'office': True,  # 任何office=*标签
    'amenity': ['bank', 'bureau_de_change', 'courthouse', 'embassy',
                'government', 'post_office', 'townhall'],
    'landuse': ['commercial'],
}

# 零售/商业类标签
RETAIL_TAGS = {
    'building': ['retail', 'supermarket', 'kiosk'],
    'shop': True,  # 任何shop=*标签
    'amenity': ['marketplace', 'food_court'],
    'landuse': ['retail'],
}

# 住宅类标签
RESIDENTIAL_TAGS = {
    'building': ['residential', 'apartments', 'house', 'detached',
                 'semidetached_house', 'terrace', 'dormitory', 'bungalow'],
    'landuse': ['residential'],
}

# 教育类标签
EDUCATION_TAGS = {
    'building': ['school', 'university', 'college', 'kindergarten'],
    'amenity': ['school', 'university', 'college', 'kindergarten',
                'library', 'language_school', 'driving_school',
                'music_school', 'prep_school', 'training'],
}

# 医疗类标签
HEALTHCARE_TAGS = {
    'building': ['hospital', 'clinic'],
    'amenity': ['hospital', 'clinic', 'doctors', 'dentist',
                'pharmacy', 'nursing_home', 'veterinary'],
    'healthcare': True,  # 任何healthcare=*标签
}

# 交通枢纽类标签
TRANSPORT_TAGS = {
    'building': ['train_station', 'transportation'],
    'railway': ['station', 'halt', 'subway_entrance'],
    'public_transport': ['station', 'stop_position', 'platform'],
    'amenity': ['bus_station', 'ferry_terminal', 'taxi'],
    'aeroway': ['terminal', 'aerodrome'],
}

# 工业类标签
INDUSTRIAL_TAGS = {
    'building': ['industrial', 'warehouse', 'factory', 'manufacture'],
    'landuse': ['industrial'],
    'man_made': ['works'],
}

# 餐饮/住宿类标签
HOSPITALITY_TAGS = {
    'building': ['hotel', 'motel'],
    'amenity': ['restaurant', 'fast_food', 'cafe', 'bar', 'pub',
                'food_court', 'biergarten', 'ice_cream'],
    'tourism': ['hotel', 'motel', 'hostel', 'guest_house'],
}

# 所有建筑标签（用于下载）
ALL_BUILDING_TAGS = {
    'building': True,  # 下载所有建筑
}

# 所有POI相关标签（用于下载）
ALL_POI_TAGS = {
    'amenity': True,
    'shop': True,
    'office': True,
    'tourism': True,
    'healthcare': True,
    'leisure': True,
}


# ----------------------------------------------------------------------------
# OSM数据下载
# ----------------------------------------------------------------------------

def download_buildings(
        study_area: Union[gpd.GeoDataFrame, Polygon, MultiPolygon],
        buffer_km: float = 1.0,
        timeout: int = 180
) -> gpd.GeoDataFrame:
    """
    下载研究区域内的所有建筑数据

    参数:
        study_area: 研究区域（GeoDataFrame或Polygon）
        buffer_km: 边界缓冲距离（公里），用于减轻边界效应
        timeout: 下载超时时间（秒）

    返回:
        建筑数据GeoDataFrame，包含geometry和各类OSM标签

    说明:
        - 使用osmnx从OpenStreetMap下载建筑多边形
        - 自动处理多边形边界，添加缓冲区
        - 返回的数据使用WGS84坐标系
    """
    logger.info("正在从OpenStreetMap下载建筑数据...")

    # 获取研究区域多边形
    if isinstance(study_area, gpd.GeoDataFrame):
        polygon = get_study_area_polygon(study_area, buffer_km=buffer_km)
    else:
        polygon = study_area

    # 配置osmnx
    ox.settings.timeout = timeout
    ox.settings.log_console = False

    try:
        # 下载建筑
        buildings = ox.features_from_polygon(
            polygon,
            tags=ALL_BUILDING_TAGS
        )

        # 只保留Polygon和MultiPolygon（排除点要素）
        buildings = buildings[
            buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])
        ]

        # 转换为GeoDataFrame并重置索引
        if not isinstance(buildings, gpd.GeoDataFrame):
            buildings = gpd.GeoDataFrame(buildings)

        buildings = buildings.reset_index(drop=True)

        logger.info(f"成功下载 {len(buildings)} 个建筑")
        return buildings

    except Exception as e:
        logger.error(f"下载建筑数据失败: {e}")
        # 返回空的GeoDataFrame
        return gpd.GeoDataFrame(
            columns=['geometry', 'building'],
            geometry='geometry',
            crs=CRS_WGS84
        )


def download_pois(
        study_area: Union[gpd.GeoDataFrame, Polygon, MultiPolygon],
        buffer_km: float = 1.0,
        timeout: int = 180
) -> gpd.GeoDataFrame:
    """
    下载研究区域内的POI数据

    参数:
        study_area: 研究区域（GeoDataFrame或Polygon）
        buffer_km: 边界缓冲距离（公里）
        timeout: 下载超时时间（秒）

    返回:
        POI数据GeoDataFrame

    说明:
        - 下载amenity、shop、office、tourism等类型的POI
        - 包含点要素和面要素
    """
    logger.info("正在从OpenStreetMap下载POI数据...")

    # 获取研究区域多边形
    if isinstance(study_area, gpd.GeoDataFrame):
        polygon = get_study_area_polygon(study_area, buffer_km=buffer_km)
    else:
        polygon = study_area

    # 配置osmnx
    ox.settings.timeout = timeout
    ox.settings.log_console = False

    try:
        # 下载POI
        pois = ox.features_from_polygon(
            polygon,
            tags=ALL_POI_TAGS
        )

        # 转换为GeoDataFrame并重置索引
        if not isinstance(pois, gpd.GeoDataFrame):
            pois = gpd.GeoDataFrame(pois)

        pois = pois.reset_index(drop=True)

        logger.info(f"成功下载 {len(pois)} 个POI")
        return pois

    except Exception as e:
        logger.error(f"下载POI数据失败: {e}")
        # 返回空的GeoDataFrame
        return gpd.GeoDataFrame(
            columns=['geometry', 'amenity'],
            geometry='geometry',
            crs=CRS_WGS84
        )


def download_all_osm_data(
        study_area: Union[gpd.GeoDataFrame, Polygon, MultiPolygon],
        buffer_km: float = 1.0,
        timeout: int = 180
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    下载研究区域内的所有OSM数据（建筑和POI）

    参数:
        study_area: 研究区域
        buffer_km: 边界缓冲距离
        timeout: 下载超时时间

    返回:
        (buildings_gdf, pois_gdf) 元组
    """
    buildings = download_buildings(study_area, buffer_km, timeout)
    pois = download_pois(study_area, buffer_km, timeout)

    return buildings, pois


# ----------------------------------------------------------------------------
# 建筑/POI分类
# ----------------------------------------------------------------------------

def classify_building(row: pd.Series) -> str:
    """
    根据OSM标签对单个建筑进行分类

    参数:
        row: 建筑数据的一行（包含OSM标签）

    返回:
        建筑类型字符串：'office', 'retail', 'residential', 'education',
        'healthcare', 'transport', 'industrial', 'hospitality', 'other'
    """
    # 检查各类标签，按优先级匹配

    # 1. 教育类（优先，因为学校建筑常被标记为其他类型）
    if _match_tags(row, EDUCATION_TAGS):
        return 'education'

    # 2. 医疗类
    if _match_tags(row, HEALTHCARE_TAGS):
        return 'healthcare'

    # 3. 交通枢纽
    if _match_tags(row, TRANSPORT_TAGS):
        return 'transport'

    # 4. 办公类
    if _match_tags(row, OFFICE_TAGS):
        return 'office'

    # 5. 零售/商业类
    if _match_tags(row, RETAIL_TAGS):
        return 'retail'

    # 6. 工业类
    if _match_tags(row, INDUSTRIAL_TAGS):
        return 'industrial'

    # 7. 餐饮/住宿类
    if _match_tags(row, HOSPITALITY_TAGS):
        return 'hospitality'

    # 8. 住宅类
    if _match_tags(row, RESIDENTIAL_TAGS):
        return 'residential'

    # 9. 未分类
    return 'other'


def _match_tags(row: pd.Series, tag_dict: Dict) -> bool:
    """
    检查数据行是否匹配指定的标签字典

    参数:
        row: 数据行
        tag_dict: 标签字典，如 {'building': ['office'], 'office': True}

    返回:
        是否匹配
    """
    for key, values in tag_dict.items():
        if key not in row.index:
            continue

        cell_value = row[key]

        # 跳过空值
        if pd.isna(cell_value):
            continue

        # 如果values为True，表示任何非空值都匹配
        if values is True:
            return True

        # 如果values是列表，检查是否在列表中
        if isinstance(values, list):
            if cell_value in values:
                return True

        # 单个值匹配
        elif cell_value == values:
            return True

    return False


def classify_buildings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    对所有建筑进行分类

    参数:
        gdf: 建筑数据GeoDataFrame

    返回:
        添加了'building_type'列的GeoDataFrame
    """
    logger.info("正在对建筑进行分类...")

    gdf = gdf.copy()
    gdf['building_type'] = gdf.apply(classify_building, axis=1)

    # 统计各类型数量
    type_counts = gdf['building_type'].value_counts()
    logger.info(f"建筑分类结果:\n{type_counts.to_string()}")

    return gdf


def classify_poi(row: pd.Series) -> str:
    """
    根据OSM标签对单个POI进行分类

    参数:
        row: POI数据的一行

    返回:
        POI类型字符串
    """
    # POI分类逻辑与建筑类似
    if _match_tags(row, EDUCATION_TAGS):
        return 'education'

    if _match_tags(row, HEALTHCARE_TAGS):
        return 'healthcare'

    if _match_tags(row, TRANSPORT_TAGS):
        return 'transport'

    if _match_tags(row, OFFICE_TAGS):
        return 'office'

    if _match_tags(row, RETAIL_TAGS):
        return 'retail'

    if _match_tags(row, HOSPITALITY_TAGS):
        return 'hospitality'

    if _match_tags(row, INDUSTRIAL_TAGS):
        return 'industrial'

    return 'other'


def classify_pois(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    对所有POI进行分类

    参数:
        gdf: POI数据GeoDataFrame

    返回:
        添加了'poi_type'列的GeoDataFrame
    """
    logger.info("正在对POI进行分类...")

    gdf = gdf.copy()
    gdf['poi_type'] = gdf.apply(classify_poi, axis=1)

    # 统计各类型数量
    type_counts = gdf['poi_type'].value_counts()
    logger.info(f"POI分类结果:\n{type_counts.to_string()}")

    return gdf


# ----------------------------------------------------------------------------
# 面积计算
# ----------------------------------------------------------------------------

def compute_building_areas(
        gdf: gpd.GeoDataFrame,
        projected_crs: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    计算建筑面积

    参数:
        gdf: 建筑数据GeoDataFrame
        projected_crs: 用于面积计算的投影坐标系；如为None则自动选择

    返回:
        添加了'area_m2'列的GeoDataFrame
    """
    logger.info("正在计算建筑面积...")

    if len(gdf) == 0:
        logger.warning("输入的GeoDataFrame为空，无需计算面积")
        gdf['area_m2'] = []
        return gdf

    gdf = gdf.copy()

    try:
        # 选择投影坐标系
        if projected_crs is None:
            projected_crs = detect_optimal_crs(gdf)

        # 转换到投影坐标系
        gdf_projected = gdf.to_crs(projected_crs)

        # 计算面积
        gdf['area_m2'] = gdf_projected.geometry.area

        # 对于点要素，面积为0，改为默认值
        point_mask = gdf.geometry.type == 'Point'
        if point_mask.any():
            gdf.loc[point_mask, 'area_m2'] = 50.0  # 点要素给默认50平方米
            logger.info(f"{point_mask.sum()}个点要素被赋予默认面积50m²")

        # 确保没有负值或NaN
        gdf['area_m2'] = gdf['area_m2'].fillna(100.0)
        gdf.loc[gdf['area_m2'] < 0, 'area_m2'] = 100.0
        gdf.loc[gdf['area_m2'] == 0, 'area_m2'] = 50.0  # 0面积改为50

        # 统计信息
        total_area = gdf['area_m2'].sum()
        mean_area = gdf['area_m2'].mean()

        logger.info(f"建筑总面积: {total_area / 1e6:.2f} km², 平均面积: {mean_area:.1f} m²")

    except Exception as e:
        logger.error(f"计算面积过程出错: {e}")
        # 出错时使用默认面积
        gdf['area_m2'] = 100.0
        logger.warning("计算失败，所有建筑使用默认面积 100 m²")

    return gdf


def estimate_building_floors(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    估算建筑层数

    参数:
        gdf: 建筑数据GeoDataFrame

    返回:
        添加了'floors'和'gross_floor_area_m2'列的GeoDataFrame

    说明:
        - 优先使用OSM中的building:levels标签
        - 如无该标签，根据建筑类型假设默认层数
        - gross_floor_area_m2 = area_m2 * floors
    """
    gdf = gdf.copy()

    # 默认层数（按建筑类型）
    default_floors = {
        'residential': 6,  # 住宅楼平均6层
        'office': 10,  # 办公楼平均10层
        'retail': 2,  # 商业平均2层
        'education': 4,  # 学校平均4层
        'healthcare': 5,  # 医院平均5层
        'industrial': 2,  # 工业厂房平均2层
        'hospitality': 8,  # 酒店平均8层
        'transport': 2,  # 交通设施平均2层
        'other': 3,  # 其他平均3层
    }

    # 尝试从OSM标签获取层数
    floors_col = None
    for col in ['building:levels', 'building_levels', 'levels']:
        if col in gdf.columns:
            floors_col = col
            break

    if floors_col:
        gdf['floors'] = pd.to_numeric(gdf[floors_col], errors='coerce')
    else:
        gdf['floors'] = np.nan

    # 对于缺失值，使用默认层数
    if 'building_type' in gdf.columns:
        for btype, default in default_floors.items():
            mask = (gdf['floors'].isna()) & (gdf['building_type'] == btype)
            gdf.loc[mask, 'floors'] = default

    # 仍有缺失值的，使用全局默认值
    gdf['floors'] = gdf['floors'].fillna(3)

    # 限制层数范围
    gdf['floors'] = gdf['floors'].clip(lower=1, upper=100)

    # 计算建筑总面积
    if 'area_m2' in gdf.columns:
        gdf['gross_floor_area_m2'] = gdf['area_m2'] * gdf['floors']

    return gdf


# ----------------------------------------------------------------------------
# 数据过滤与清洗
# ----------------------------------------------------------------------------

def filter_buildings_in_study_area(
        buildings: gpd.GeoDataFrame,
        study_area: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    过滤仅保留研究区域内的建筑

    参数:
        buildings: 建筑数据GeoDataFrame
        study_area: 研究区域GeoDataFrame

    返回:
        研究区域内的建筑GeoDataFrame
    """
    # 确保CRS一致
    buildings = transform_to_wgs84(buildings)
    study_area = transform_to_wgs84(study_area)

    # 获取研究区域的合并多边形
    study_polygon = study_area.unary_union

    # 筛选
    mask = buildings.geometry.intersects(study_polygon)
    filtered = buildings[mask].copy()

    logger.info(f"过滤后保留 {len(filtered)}/{len(buildings)} 个建筑")

    return filtered


def remove_duplicate_buildings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    移除重复的建筑（基于几何图形）

    参数:
        gdf: 建筑数据GeoDataFrame

    返回:
        去重后的GeoDataFrame
    """
    original_count = len(gdf)

    # 基于几何图形的WKT表示去重
    gdf = gdf.copy()
    gdf['_geometry_wkt'] = gdf.geometry.apply(lambda g: g.wkt[:100])  # 取前100字符作为近似
    gdf = gdf.drop_duplicates(subset='_geometry_wkt')
    gdf = gdf.drop(columns=['_geometry_wkt'])

    removed_count = original_count - len(gdf)
    if removed_count > 0:
        logger.info(f"移除了 {removed_count} 个重复建筑")

    return gdf


# ----------------------------------------------------------------------------
# 数据处理流水线
# ----------------------------------------------------------------------------

def process_osm_buildings(
        study_area: gpd.GeoDataFrame,
        buffer_km: float = 1.0,
        timeout: int = 180
) -> gpd.GeoDataFrame:
    """
    完整的建筑数据处理流程

    参数:
        study_area: 研究区域GeoDataFrame
        buffer_km: 边界缓冲距离
        timeout: 下载超时时间

    返回:
        处理后的建筑GeoDataFrame，包含以下列:
        - geometry: 建筑几何图形
        - building_type: 建筑类型
        - area_m2: 建筑占地面积（平方米）
        - floors: 估算层数
        - gross_floor_area_m2: 建筑总面积（平方米）
        - 其他OSM原始标签
    """
    try:
        # 1. 下载建筑数据
        buildings = download_buildings(study_area, buffer_km, timeout)

        if len(buildings) == 0:
            logger.warning("未下载到任何建筑数据，返回空GeoDataFrame")
            # 返回带有必需列的空GeoDataFrame
            return _create_empty_buildings_gdf()

        # 2. 过滤到研究区域内
        buildings = filter_buildings_in_study_area(buildings, study_area)

        if len(buildings) == 0:
            logger.warning("过滤后无建筑数据，返回空GeoDataFrame")
            return _create_empty_buildings_gdf()

        # 3. 去重
        buildings = remove_duplicate_buildings(buildings)

        # 4. 分类
        buildings = classify_buildings(buildings)

        # 5. 计算面积（增加异常处理）
        try:
            buildings = compute_building_areas(buildings)
        except Exception as e:
            logger.error(f"计算建筑面积失败: {e}")
            # 添加默认面积列
            buildings['area_m2'] = 100.0  # 默认100平方米
            logger.warning("使用默认面积值 100 m²")

        # 6. 确保area_m2列存在
        if 'area_m2' not in buildings.columns:
            logger.warning("area_m2列缺失，添加默认值")
            buildings['area_m2'] = 100.0

        # 7. 估算层数和总面积
        try:
            buildings = estimate_building_floors(buildings)
        except Exception as e:
            logger.error(f"估算建筑层数失败: {e}")
            buildings['floors'] = 3.0  # 默认3层
            buildings['gross_floor_area_m2'] = buildings['area_m2'] * buildings['floors']

        # 8. 确保gross_floor_area_m2列存在
        if 'gross_floor_area_m2' not in buildings.columns:
            if 'floors' not in buildings.columns:
                buildings['floors'] = 3.0
            buildings['gross_floor_area_m2'] = buildings['area_m2'] * buildings['floors']

        return buildings

    except Exception as e:
        logger.error(f"处理建筑数据时发生错误: {e}")
        # 返回空的但结构正确的GeoDataFrame
        return _create_empty_buildings_gdf()


def _create_empty_buildings_gdf() -> gpd.GeoDataFrame:
    """
    创建空的建筑GeoDataFrame，包含所有必需列

    返回:
        空的GeoDataFrame，带有正确的列结构
    """
    return gpd.GeoDataFrame(
        columns=[
            'geometry', 'building_type', 'area_m2',
            'floors', 'gross_floor_area_m2'
        ],
        geometry='geometry',
        crs=CRS_WGS84
    )


def process_osm_pois(
        study_area: gpd.GeoDataFrame,
        buffer_km: float = 1.0,
        timeout: int = 180
) -> gpd.GeoDataFrame:
    """
    完整的POI数据处理流程

    参数:
        study_area: 研究区域GeoDataFrame
        buffer_km: 边界缓冲距离
        timeout: 下载超时时间

    返回:
        处理后的POI GeoDataFrame
    """
    # 1. 下载POI数据
    pois = download_pois(study_area, buffer_km, timeout)

    if len(pois) == 0:
        logger.warning("未下载到任何POI数据")
        return pois

    # 2. 分类
    pois = classify_pois(pois)

    return pois


# ----------------------------------------------------------------------------
# 数据保存
# ----------------------------------------------------------------------------

def save_osm_data(
        buildings: gpd.GeoDataFrame,
        pois: gpd.GeoDataFrame,
        output_dir: Union[str, Path]
) -> Tuple[Path, Path]:
    """
    保存OSM数据到文件

    参数:
        buildings: 建筑数据
        pois: POI数据
        output_dir: 输出目录

    返回:
        (buildings_path, pois_path) 元组
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    buildings_path = output_dir / 'osm_buildings.geojson'
    pois_path = output_dir / 'osm_pois.geojson'

    # 选择要保存的列（排除过多的OSM原始标签以减小文件大小）
    building_cols = ['geometry', 'building_type', 'area_m2', 'floors',
                     'gross_floor_area_m2', 'building', 'name']
    building_cols = [c for c in building_cols if c in buildings.columns]

    poi_cols = ['geometry', 'poi_type', 'amenity', 'shop', 'name']
    poi_cols = [c for c in poi_cols if c in pois.columns]

    buildings[building_cols].to_file(buildings_path, driver='GeoJSON')
    pois[poi_cols].to_file(pois_path, driver='GeoJSON')

    logger.info(f"已保存建筑数据: {buildings_path}")
    logger.info(f"已保存POI数据: {pois_path}")

    return buildings_path, pois_path


# ----------------------------------------------------------------------------
# 统计与汇总
# ----------------------------------------------------------------------------

def get_building_statistics(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    生成建筑数据统计信息

    参数:
        gdf: 建筑数据GeoDataFrame

    返回:
        统计信息DataFrame
    """
    if 'building_type' not in gdf.columns:
        gdf = classify_buildings(gdf)

    stats = gdf.groupby('building_type').agg({
        'geometry': 'count',
        'area_m2': ['sum', 'mean'],
        'gross_floor_area_m2': 'sum' if 'gross_floor_area_m2' in gdf.columns else 'count'
    }).round(2)

    stats.columns = ['数量', '总占地面积(m²)', '平均占地面积(m²)', '总建筑面积(m²)']
    stats = stats.reset_index()
    stats.columns = ['建筑类型'] + list(stats.columns[1:])

    return stats


def get_poi_statistics(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    生成POI数据统计信息

    参数:
        gdf: POI数据GeoDataFrame

    返回:
        统计信息DataFrame
    """
    if 'poi_type' not in gdf.columns:
        gdf = classify_pois(gdf)

    stats = gdf['poi_type'].value_counts().reset_index()
    stats.columns = ['POI类型', '数量']

    return stats