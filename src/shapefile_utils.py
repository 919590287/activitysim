# ============================================================================
# shapefile_utils.py
# ============================================================================
# 模块职责：研究区域Shapefile的读取、处理和坐标系统转换
# 包括：
# - 读取.shp文件或包含shapefile的.zip压缩包
# - 检测并统一坐标参考系统（CRS）
# - 计算研究区域边界和缓冲区
# - 提取各TAZ的质心点
# - 为OSM下载和路网裁剪提供边界多边形
# ============================================================================

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import pyproj

from .io_utils import (
    logger, ensure_dir, extract_zip, list_files,
    validate_dataframe, write_csv
)

# ----------------------------------------------------------------------------
# 坐标参考系统常量
# ----------------------------------------------------------------------------

# WGS84 地理坐标系（经纬度），OSMnx默认使用
CRS_WGS84 = "EPSG:4326"

# Web墨卡托投影（米），常用于Web地图
CRS_WEB_MERCATOR = "EPSG:3857"

# 中国常用投影坐标系
# CGCS2000 / 3-degree Gauss-Kruger zone 39 (适用于东经117°附近，如北京、天津)
CRS_CGCS2000_39 = "EPSG:4549"

# UTM投影前缀（具体zone根据研究区域确定）
CRS_UTM_PREFIX = "EPSG:326"  # 北半球UTM zones 1-60对应 32601-32660


# ----------------------------------------------------------------------------
# Shapefile读取与解析
# ----------------------------------------------------------------------------

def read_shapefile(
        file_path: Union[str, Path],
        target_crs: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    读取Shapefile或包含Shapefile的ZIP压缩包

    参数:
        file_path: .shp文件路径或.zip压缩包路径
        target_crs: 目标坐标参考系统，如"EPSG:4326"；如为None则保持原始CRS

    返回:
        GeoDataFrame对象，包含研究区域多边形

    异常:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误或无法读取

    说明:
        - 支持直接读取.shp文件（需同目录存在.dbf、.shx、.prj等附属文件）
        - 支持读取.zip压缩包，自动解压后查找.shp文件
        - 自动检测并记录原始CRS信息
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 处理ZIP压缩包
    if file_path.suffix.lower() == '.zip':
        logger.info(f"检测到ZIP压缩包，正在解压: {file_path}")
        extract_dir = extract_zip(file_path)

        # 在解压目录中查找.shp文件
        shp_files = list_files(extract_dir, extensions=['.shp'], recursive=True)

        if not shp_files:
            raise ValueError(f"ZIP压缩包中未找到.shp文件: {file_path}")

        if len(shp_files) > 1:
            logger.warning(f"ZIP中包含多个.shp文件，使用第一个: {shp_files[0]}")

        shp_path = shp_files[0]
    else:
        shp_path = file_path

    # 检查.shp文件及其附属文件
    _validate_shapefile_components(shp_path)

    # 读取Shapefile
    logger.info(f"正在读取Shapefile: {shp_path}")
    gdf = gpd.read_file(shp_path)

    # 记录原始CRS信息
    original_crs = gdf.crs
    if original_crs:
        logger.info(f"原始坐标参考系统: {original_crs}")
    else:
        logger.warning("Shapefile未定义CRS，将假设为WGS84 (EPSG:4326)")
        gdf = gdf.set_crs(CRS_WGS84)

    # 坐标转换
    if target_crs and gdf.crs != target_crs:
        logger.info(f"正在转换坐标系统: {gdf.crs} -> {target_crs}")
        gdf = gdf.to_crs(target_crs)

    # 基本验证
    if len(gdf) == 0:
        raise ValueError("Shapefile不包含任何要素")

    logger.info(f"成功读取Shapefile: {len(gdf)}个要素")

    return gdf


def _validate_shapefile_components(shp_path: Path) -> None:
    """
    验证Shapefile的附属文件是否完整

    参数:
        shp_path: .shp文件路径

    异常:
        ValueError: 缺少必需的附属文件
    """
    required_extensions = ['.shp', '.shx', '.dbf']
    optional_extensions = ['.prj', '.cpg', '.sbn', '.sbx']

    base_path = shp_path.parent / shp_path.stem

    missing_required = []
    for ext in required_extensions:
        if not (base_path.parent / f"{base_path.stem}{ext}").exists():
            # 尝试大小写不敏感匹配
            found = False
            for f in shp_path.parent.iterdir():
                if f.stem.lower() == base_path.stem.lower() and f.suffix.lower() == ext:
                    found = True
                    break
            if not found:
                missing_required.append(ext)

    if missing_required:
        raise ValueError(f"Shapefile缺少必需的附属文件: {missing_required}")

    # 检查.prj文件（可选但重要）
    prj_exists = any(
        f.suffix.lower() == '.prj' and f.stem.lower() == shp_path.stem.lower()
        for f in shp_path.parent.iterdir()
    )
    if not prj_exists:
        logger.warning("Shapefile缺少.prj投影文件，CRS信息可能不完整")


# ----------------------------------------------------------------------------
# 坐标参考系统处理
# ----------------------------------------------------------------------------

def detect_optimal_crs(gdf: gpd.GeoDataFrame) -> str:
    """
    根据研究区域位置自动选择最优的投影坐标系

    参数:
        gdf: 研究区域GeoDataFrame

    返回:
        推荐的投影CRS字符串（如"EPSG:32650"）

    说明:
        - 对于中国区域，优先使用CGCS2000投影
        - 对于其他区域，使用UTM投影
        - 投影坐标系用于精确的距离和面积计算
    """
    # 确保使用WGS84获取边界
    if gdf.crs != CRS_WGS84:
        gdf_wgs84 = gdf.to_crs(CRS_WGS84)
    else:
        gdf_wgs84 = gdf

    # 计算研究区域中心点
    bounds = gdf_wgs84.total_bounds  # [minx, miny, maxx, maxy]
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    logger.debug(f"研究区域中心点: ({center_lon:.4f}, {center_lat:.4f})")

    # 判断是否在中国境内（大致范围：经度73-135，纬度18-54）
    if 73 <= center_lon <= 135 and 18 <= center_lat <= 54:
        # 使用CGCS2000 3度带投影
        # 带号 = (经度 + 1.5) / 3 向下取整
        zone = int((center_lon + 1.5) / 3)
        # CGCS2000 3度带 EPSG代码：4513 + zone（25-45带对应4538-4558）
        # 这里简化使用UTM代替
        utm_zone = int((center_lon + 180) / 6) + 1
        epsg_code = f"EPSG:326{utm_zone:02d}" if center_lat >= 0 else f"EPSG:327{utm_zone:02d}"
        logger.info(f"检测到中国区域，推荐使用UTM投影: {epsg_code}")
    else:
        # 使用UTM投影
        utm_zone = int((center_lon + 180) / 6) + 1
        if center_lat >= 0:
            epsg_code = f"EPSG:326{utm_zone:02d}"
        else:
            epsg_code = f"EPSG:327{utm_zone:02d}"
        logger.info(f"推荐使用UTM Zone {utm_zone}投影: {epsg_code}")

    return epsg_code


def transform_to_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    将GeoDataFrame转换为WGS84坐标系

    参数:
        gdf: 输入GeoDataFrame

    返回:
        WGS84坐标系的GeoDataFrame
    """
    if gdf.crs is None:
        logger.warning("GeoDataFrame无CRS信息，假设为WGS84")
        return gdf.set_crs(CRS_WGS84)

    if gdf.crs == CRS_WGS84:
        return gdf

    return gdf.to_crs(CRS_WGS84)


def transform_to_projected(
        gdf: gpd.GeoDataFrame,
        target_crs: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    将GeoDataFrame转换为投影坐标系（用于距离/面积计算）

    参数:
        gdf: 输入GeoDataFrame
        target_crs: 目标投影CRS；如为None则自动选择

    返回:
        投影坐标系的GeoDataFrame
    """
    if target_crs is None:
        target_crs = detect_optimal_crs(gdf)

    return gdf.to_crs(target_crs)


# ----------------------------------------------------------------------------
# 研究区域边界处理
# ----------------------------------------------------------------------------

def get_study_area_bounds(
        gdf: gpd.GeoDataFrame,
        buffer_km: float = 0.0,
        crs: str = CRS_WGS84
) -> Tuple[float, float, float, float]:
    """
    获取研究区域的边界框

    参数:
        gdf: 研究区域GeoDataFrame
        buffer_km: 向外扩展的缓冲距离（公里）
        crs: 返回边界的坐标系

    返回:
        (minx, miny, maxx, maxy) 边界框坐标
    """
    # 转换到目标CRS
    gdf_target = gdf.to_crs(crs) if gdf.crs != crs else gdf

    bounds = gdf_target.total_bounds

    if buffer_km > 0:
        # 对于WGS84，简单地按经纬度度数估算缓冲
        # 1度纬度 ≈ 111km，1度经度 ≈ 111km * cos(lat)
        if crs == CRS_WGS84:
            center_lat = (bounds[1] + bounds[3]) / 2
            lat_buffer = buffer_km / 111.0
            lon_buffer = buffer_km / (111.0 * np.cos(np.radians(center_lat)))
            bounds = (
                bounds[0] - lon_buffer,
                bounds[1] - lat_buffer,
                bounds[2] + lon_buffer,
                bounds[3] + lat_buffer
            )
        else:
            # 对于投影坐标系，直接用米为单位
            buffer_m = buffer_km * 1000
            bounds = (
                bounds[0] - buffer_m,
                bounds[1] - buffer_m,
                bounds[2] + buffer_m,
                bounds[3] + buffer_m
            )

    return tuple(bounds)


def get_study_area_polygon(
        gdf: gpd.GeoDataFrame,
        buffer_km: float = 0.0
) -> Union[Polygon, MultiPolygon]:
    """
    获取研究区域的合并多边形

    参数:
        gdf: 研究区域GeoDataFrame
        buffer_km: 向外扩展的缓冲距离（公里）

    返回:
        研究区域多边形（Polygon或MultiPolygon）

    说明:
        - 将所有TAZ多边形合并为单个边界多边形
        - 如指定buffer_km，则向外扩展指定距离
        - 返回的多边形使用WGS84坐标系
    """
    # 转到投影坐标系进行缓冲（精确距离）
    projected_crs = detect_optimal_crs(gdf)
    gdf_projected = gdf.to_crs(projected_crs)

    # 合并所有多边形
    unified = unary_union(gdf_projected.geometry)

    # 添加缓冲区
    if buffer_km > 0:
        buffer_m = buffer_km * 1000
        unified = unified.buffer(buffer_m)

    # 转回WGS84
    # 需要使用GeoSeries来进行坐标转换
    gs = gpd.GeoSeries([unified], crs=projected_crs)
    gs_wgs84 = gs.to_crs(CRS_WGS84)

    return gs_wgs84.iloc[0]


# ----------------------------------------------------------------------------
# TAZ（交通分析小区）处理
# ----------------------------------------------------------------------------

def extract_taz_info(
        gdf: gpd.GeoDataFrame,
        zone_id_column: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    提取TAZ信息，包括ID、面积、质心等

    参数:
        gdf: 研究区域GeoDataFrame
        zone_id_column: 指定作为zone_id的列名；如为None则自动检测或创建

    返回:
        包含TAZ信息的GeoDataFrame，包含以下列:
        - zone_id: TAZ唯一标识符
        - area_km2: 面积（平方公里）
        - centroid_lon: 质心经度
        - centroid_lat: 质心纬度
        - geometry: 原始几何图形
    """
    gdf = gdf.copy()

    # 确定zone_id列
    if zone_id_column and zone_id_column in gdf.columns:
        gdf['zone_id'] = gdf[zone_id_column]
    else:
        # 尝试自动检测可能的ID列
        possible_id_cols = ['zone_id', 'ZONE_ID', 'TAZ', 'taz', 'ID', 'id',
                            'FID', 'OBJECTID', 'ZONEID']
        found_col = None
        for col in possible_id_cols:
            if col in gdf.columns:
                found_col = col
                break

        if found_col:
            gdf['zone_id'] = gdf[found_col]
            logger.info(f"使用'{found_col}'列作为zone_id")
        else:
            # 创建新的zone_id
            gdf['zone_id'] = range(1, len(gdf) + 1)
            logger.warning("未找到zone_id列，已自动创建序号ID")

    # 确保zone_id为整数类型
    gdf['zone_id'] = gdf['zone_id'].astype(int)

    # 计算面积（需要投影坐标系）
    projected_crs = detect_optimal_crs(gdf)
    gdf_projected = gdf.to_crs(projected_crs)
    gdf['area_km2'] = gdf_projected.geometry.area / 1e6  # 平方米转平方公里

    # 计算质心（WGS84坐标）
    gdf_wgs84 = transform_to_wgs84(gdf)
    centroids = gdf_wgs84.geometry.centroid
    gdf['centroid_lon'] = centroids.x
    gdf['centroid_lat'] = centroids.y

    # 确保geometry列是WGS84
    gdf = gdf.set_geometry('geometry')
    gdf = transform_to_wgs84(gdf)

    # 选择和排序列
    result_columns = ['zone_id', 'area_km2', 'centroid_lon', 'centroid_lat', 'geometry']
    # 保留原始属性列
    for col in gdf.columns:
        if col not in result_columns:
            result_columns.append(col)

    gdf = gdf[[col for col in result_columns if col in gdf.columns]]

    logger.info(f"已提取{len(gdf)}个TAZ的信息")

    return gdf


def compute_zone_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    计算各TAZ的质心点

    参数:
        gdf: 包含zone_id的研究区域GeoDataFrame

    返回:
        质心点GeoDataFrame，包含zone_id和Point几何图形
    """
    if 'zone_id' not in gdf.columns:
        raise ValueError("GeoDataFrame必须包含'zone_id'列")

    # 确保WGS84坐标系
    gdf_wgs84 = transform_to_wgs84(gdf)

    # 计算质心
    centroids = gdf_wgs84.copy()
    centroids['geometry'] = centroids.geometry.centroid

    return centroids[['zone_id', 'geometry']]


def save_study_area(
        gdf: gpd.GeoDataFrame,
        output_path: Union[str, Path],
        format: str = 'geojson'
) -> Path:
    """
    保存研究区域数据

    参数:
        gdf: 研究区域GeoDataFrame
        output_path: 输出文件路径
        format: 输出格式 ('geojson', 'shp', 'gpkg')

    返回:
        保存的文件路径
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    # 确保使用WGS84保存
    gdf_wgs84 = transform_to_wgs84(gdf)

    if format.lower() == 'geojson':
        gdf_wgs84.to_file(output_path, driver='GeoJSON')
    elif format.lower() == 'shp':
        gdf_wgs84.to_file(output_path, driver='ESRI Shapefile')
    elif format.lower() == 'gpkg':
        gdf_wgs84.to_file(output_path, driver='GPKG')
    else:
        raise ValueError(f"不支持的输出格式: {format}")

    logger.info(f"已保存研究区域数据: {output_path}")
    return output_path


# ----------------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------------

def point_in_study_area(
        lon: float,
        lat: float,
        study_area: gpd.GeoDataFrame
) -> bool:
    """
    判断点是否在研究区域内

    参数:
        lon: 经度
        lat: 纬度
        study_area: 研究区域GeoDataFrame

    返回:
        点是否在研究区域内
    """
    point = Point(lon, lat)
    study_area_wgs84 = transform_to_wgs84(study_area)

    return any(study_area_wgs84.geometry.contains(point))


def find_containing_zone(
        lon: float,
        lat: float,
        gdf: gpd.GeoDataFrame
) -> Optional[int]:
    """
    查找包含给定点的TAZ

    参数:
        lon: 经度
        lat: 纬度
        gdf: 包含zone_id的研究区域GeoDataFrame

    返回:
        包含该点的zone_id，如点不在任何TAZ内则返回None
    """
    if 'zone_id' not in gdf.columns:
        raise ValueError("GeoDataFrame必须包含'zone_id'列")

    point = Point(lon, lat)
    gdf_wgs84 = transform_to_wgs84(gdf)

    for idx, row in gdf_wgs84.iterrows():
        if row.geometry.contains(point):
            return int(row['zone_id'])

    return None


def assign_points_to_zones(
        points_gdf: gpd.GeoDataFrame,
        zones_gdf: gpd.GeoDataFrame,
        zone_id_col: str = 'zone_id'
) -> gpd.GeoDataFrame:
    """
    将点数据分配到对应的TAZ

    参数:
        points_gdf: 点数据GeoDataFrame
        zones_gdf: 包含zone_id的TAZ GeoDataFrame
        zone_id_col: TAZ ID列名

    返回:
        添加了zone_id列的点数据GeoDataFrame
    """
    # 确保两者使用相同的CRS
    points = transform_to_wgs84(points_gdf).copy()
    zones = transform_to_wgs84(zones_gdf)

    # 空间连接
    result = gpd.sjoin(
        points,
        zones[[zone_id_col, 'geometry']],
        how='left',
        predicate='within'
    )

    # 处理不在任何zone内的点
    outside_count = result[zone_id_col].isna().sum()
    if outside_count > 0:
        logger.warning(f"{outside_count}个点不在任何TAZ内")

    # 清理sjoin产生的额外列
    if 'index_right' in result.columns:
        result = result.drop(columns=['index_right'])

    return result


def get_zone_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    生成TAZ汇总统计信息

    参数:
        gdf: 包含TAZ信息的GeoDataFrame

    返回:
        汇总统计DataFrame
    """
    if 'zone_id' not in gdf.columns:
        raise ValueError("GeoDataFrame必须包含'zone_id'列")

    summary = pd.DataFrame({
        '统计项': ['TAZ数量', '总面积(km²)', '平均面积(km²)',
                   '最小面积(km²)', '最大面积(km²)'],
        '值': [
            len(gdf),
            gdf['area_km2'].sum() if 'area_km2' in gdf.columns else None,
            gdf['area_km2'].mean() if 'area_km2' in gdf.columns else None,
            gdf['area_km2'].min() if 'area_km2' in gdf.columns else None,
            gdf['area_km2'].max() if 'area_km2' in gdf.columns else None,
        ]
    })

    return summary