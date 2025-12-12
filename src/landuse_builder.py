# ============================================================================
# landuse_builder.py
# ============================================================================
# 模块职责：从OSM建筑/POI数据构建ActivitySim所需的land_use表
# 包括：
# - 定义建筑面积到就业岗位/人口的转换系数
# - 按TAZ聚合建筑和POI数据
# - 计算各类就业岗位数量
# - 估算人口和家庭数量
# - 生成ActivitySim格式的land_use.csv
# ============================================================================

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np
import geopandas as gpd

from .io_utils import logger, ensure_dir, write_csv, validate_dataframe
from .shapefile_utils import assign_points_to_zones, transform_to_wgs84

# ----------------------------------------------------------------------------
# 默认转换系数
# ----------------------------------------------------------------------------
# 这些系数定义了建筑面积到就业岗位/人口的转换关系
# 用户可以通过Streamlit界面修改这些系数

DEFAULT_CONVERSION_COEFFICIENTS = {
    # 办公建筑：每20平方米 = 1个办公岗位
    'office': {
        'area_per_job': 20.0,
        'job_type': 'emp_office'
    },

    # 零售/商业建筑：每30平方米 = 1个零售岗位
    'retail': {
        'area_per_job': 30.0,
        'job_type': 'emp_retail'
    },

    # 教育设施：每15平方米 = 0.1教职工 + 0.9学生（作为教育就业统计）
    'education': {
        'area_per_job': 15.0,
        'job_type': 'emp_edu',
        'job_multiplier': 0.1,  # 教职工比例
    },

    # 医疗设施：每25平方米 = 1个医疗岗位
    'healthcare': {
        'area_per_job': 25.0,
        'job_type': 'emp_health'
    },

    # 工业建筑：每40平方米 = 1个工业岗位
    'industrial': {
        'area_per_job': 40.0,
        'job_type': 'emp_industrial'
    },

    # 餐饮/住宿：每35平方米 = 1个服务业岗位
    'hospitality': {
        'area_per_job': 35.0,
        'job_type': 'emp_retail'  # 归入零售/服务类
    },

    # 交通设施：每50平方米 = 1个交通岗位
    'transport': {
        'area_per_job': 50.0,
        'job_type': 'emp_other'
    },

    # 住宅建筑：每50平方米 = 1人, 每120平方米 = 1户
    'residential': {
        'area_per_person': 50.0,
        'area_per_household': 120.0,
    },

    # 其他类型：每60平方米 = 1个其他岗位
    'other': {
        'area_per_job': 60.0,
        'job_type': 'emp_other'
    }
}


# ----------------------------------------------------------------------------
# 转换系数管理
# ----------------------------------------------------------------------------

class ConversionCoefficients:
    """
    转换系数管理类

    提供转换系数的存取、修改和验证功能
    """

    def __init__(self, coefficients: Optional[Dict] = None):
        """
        初始化转换系数

        参数:
            coefficients: 自定义系数字典；如为None则使用默认值
        """
        if coefficients is None:
            self.coefficients = DEFAULT_CONVERSION_COEFFICIENTS.copy()
        else:
            self.coefficients = coefficients

    def get_coefficient(
            self,
            building_type: str,
            key: str,
            default: float = 50.0
    ) -> float:
        """
        获取特定建筑类型的系数

        参数:
            building_type: 建筑类型
            key: 系数键名（如'area_per_job'）
            default: 默认值

        返回:
            系数值
        """
        if building_type in self.coefficients:
            return self.coefficients[building_type].get(key, default)
        else:
            return default

    def update_coefficient(
            self,
            building_type: str,
            key: str,
            value: float
    ) -> None:
        """
        更新系数

        参数:
            building_type: 建筑类型
            key: 系数键名
            value: 新的系数值
        """
        if building_type not in self.coefficients:
            self.coefficients[building_type] = {}

        self.coefficients[building_type][key] = value

    def to_dataframe(self) -> pd.DataFrame:
        """
        将系数转换为DataFrame（便于在Streamlit中编辑）

        返回:
            系数DataFrame
        """
        rows = []
        for btype, params in self.coefficients.items():
            for key, value in params.items():
                rows.append({
                    'building_type': btype,
                    'parameter': key,
                    'value': value
                })
        return pd.DataFrame(rows)

    def from_dataframe(self, df: pd.DataFrame) -> None:
        """
        从DataFrame更新系数

        参数:
            df: 包含building_type, parameter, value列的DataFrame
        """
        self.coefficients = {}
        for _, row in df.iterrows():
            btype = row['building_type']
            param = row['parameter']
            value = row['value']

            if btype not in self.coefficients:
                self.coefficients[btype] = {}

            self.coefficients[btype][param] = value


# ----------------------------------------------------------------------------
# 建筑数据到就业/人口的转换
# ----------------------------------------------------------------------------

def convert_buildings_to_employment(
        buildings: gpd.GeoDataFrame,
        coefficients: Optional[ConversionCoefficients] = None
) -> gpd.GeoDataFrame:
    """
    将建筑数据转换为就业岗位数量

    参数:
        buildings: 建筑GeoDataFrame，需包含building_type和area_m2或gross_floor_area_m2
        coefficients: 转换系数对象；如为None则使用默认系数

    返回:
        添加了就业岗位列的GeoDataFrame
    """
    logger.info("正在将建筑数据转换为就业岗位...")

    if coefficients is None:
        coefficients = ConversionCoefficients()

    buildings = buildings.copy()

    # ===== 新增：检查并修复缺失的列 =====
    # 确定使用哪个面积列
    area_col = None
    if 'gross_floor_area_m2' in buildings.columns:
        area_col = 'gross_floor_area_m2'
    elif 'area_m2' in buildings.columns:
        area_col = 'area_m2'
    else:
        # 如果两个面积列都不存在，尝试计算
        logger.warning("建筑数据缺少面积列，尝试计算...")
        try:
            from .osm_poi_utils import compute_building_areas
            buildings = compute_building_areas(buildings)
            area_col = 'area_m2'
        except Exception as e:
            logger.error(f"无法计算建筑面积: {e}")
            # 添加默认面积
            logger.warning("使用默认面积值 100 m²")
            buildings['area_m2'] = 100.0
            area_col = 'area_m2'

    # 确保有building_type列
    if 'building_type' not in buildings.columns:
        logger.warning("建筑数据缺少building_type列，使用默认值'other'")
        buildings['building_type'] = 'other'
    # ===== 修复结束 =====

    # 初始化就业列
    emp_columns = ['emp_office', 'emp_retail', 'emp_edu', 'emp_health',
                   'emp_industrial', 'emp_other']
    for col in emp_columns:
        buildings[col] = 0.0

    # ... 其余代码保持不变 ...

    # 按建筑类型转换
    for btype in buildings['building_type'].unique():
        mask = buildings['building_type'] == btype

        if btype == 'residential':
            # 住宅不产生就业
            continue

        # 获取转换系数
        area_per_job = coefficients.get_coefficient(btype, 'area_per_job', 50.0)
        job_type = coefficients.get_coefficient(btype, 'job_type', 'emp_other')
        job_multiplier = coefficients.get_coefficient(btype, 'job_multiplier', 1.0)

        # 计算就业数
        jobs = (buildings.loc[mask, area_col] / area_per_job) * job_multiplier

        # 分配到对应就业类型
        if job_type in emp_columns:
            buildings.loc[mask, job_type] += jobs
        else:
            buildings.loc[mask, 'emp_other'] += jobs

    # 计算总就业
    buildings['emp_total'] = buildings[emp_columns].sum(axis=1)

    logger.info(f"已转换就业岗位，总计: {buildings['emp_total'].sum():.0f}个")

    return buildings


def convert_buildings_to_population(
        buildings: gpd.GeoDataFrame,
        coefficients: Optional[ConversionCoefficients] = None
) -> gpd.GeoDataFrame:
    """
    将住宅建筑转换为人口和家庭数量

    参数:
        buildings: 建筑GeoDataFrame
        coefficients: 转换系数对象

    返回:
        添加了pop和hh列的GeoDataFrame
    """
    logger.info("正在将住宅建筑转换为人口和家庭...")

    if coefficients is None:
        coefficients = ConversionCoefficients()

    buildings = buildings.copy()

    # 确定面积列
    area_col = 'gross_floor_area_m2' if 'gross_floor_area_m2' in buildings.columns else 'area_m2'

    # 初始化
    buildings['pop'] = 0.0
    buildings['hh'] = 0.0

    # 仅对住宅建筑转换
    residential_mask = buildings['building_type'] == 'residential'

    area_per_person = coefficients.get_coefficient('residential', 'area_per_person', 50.0)
    area_per_household = coefficients.get_coefficient('residential', 'area_per_household', 120.0)

    buildings.loc[residential_mask, 'pop'] = \
        buildings.loc[residential_mask, area_col] / area_per_person

    buildings.loc[residential_mask, 'hh'] = \
        buildings.loc[residential_mask, area_col] / area_per_household

    total_pop = buildings['pop'].sum()
    total_hh = buildings['hh'].sum()

    logger.info(f"已估算人口: {total_pop:.0f}人, 家庭: {total_hh:.0f}户")

    return buildings


# ----------------------------------------------------------------------------
# 按TAZ聚合
# ----------------------------------------------------------------------------

def aggregate_buildings_by_zone(
        buildings: gpd.GeoDataFrame,
        zones: gpd.GeoDataFrame,
        zone_id_col: str = 'zone_id'
) -> pd.DataFrame:
    """
    按TAZ聚合建筑数据

    参数:
        buildings: 建筑GeoDataFrame，需包含就业和人口列
        zones: TAZ GeoDataFrame，包含zone_id
        zone_id_col: zone ID列名

    返回:
        按zone聚合的DataFrame
    """
    logger.info("正在按TAZ聚合建筑数据...")

    # 确保CRS一致
    buildings = transform_to_wgs84(buildings)
    zones = transform_to_wgs84(zones)

    # 空间连接：将建筑分配到TAZ
    buildings_with_zone = assign_points_to_zones(
        buildings,
        zones,
        zone_id_col=zone_id_col
    )

    # 定义聚合列
    agg_dict = {}

    # 就业列
    emp_cols = [col for col in buildings_with_zone.columns if col.startswith('emp_')]
    for col in emp_cols:
        agg_dict[col] = 'sum'

    # 人口和家庭
    if 'pop' in buildings_with_zone.columns:
        agg_dict['pop'] = 'sum'
    if 'hh' in buildings_with_zone.columns:
        agg_dict['hh'] = 'sum'

    # 建筑数量
    agg_dict['building_type'] = 'count'

    # 按zone聚合
    zone_data = buildings_with_zone.groupby(zone_id_col).agg(agg_dict).reset_index()

    # 重命名建筑数量列
    zone_data = zone_data.rename(columns={'building_type': 'num_buildings'})

    logger.info(f"已聚合到 {len(zone_data)} 个TAZ")

    return zone_data


# ----------------------------------------------------------------------------
# 构建land_use表
# ----------------------------------------------------------------------------

def build_landuse_table(
        zones: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        coefficients: Optional[ConversionCoefficients] = None,
        zone_id_col: str = 'zone_id'
) -> pd.DataFrame:
    """
    构建ActivitySim格式的land_use表

    参数:
        zones: TAZ GeoDataFrame，需包含zone_id, area_km2
        buildings: 建筑GeoDataFrame，已分类和计算面积
        coefficients: 转换系数
        zone_id_col: zone ID列名

    返回:
        land_use DataFrame，包含ActivitySim所需的所有字段

    输出列:
        - zone_id: TAZ ID
        - area: 面积（平方公里）
        - hh: 家庭数量
        - pop: 人口数量
        - emp_total: 总就业
        - emp_office: 办公就业
        - emp_retail: 零售就业
        - emp_edu: 教育就业
        - emp_health: 医疗就业
        - emp_industrial: 工业就业
        - emp_other: 其他就业
        - density: 人口密度（人/平方公里）
        - emp_density: 就业密度（岗位/平方公里）
        - is_cbd: 是否CBD（基于就业密度判断）
    """
    logger.info("正在构建land_use表...")

    # 1. 转换建筑数据
    buildings = convert_buildings_to_employment(buildings, coefficients)
    buildings = convert_buildings_to_population(buildings, coefficients)

    # 2. 按TAZ聚合
    zone_agg = aggregate_buildings_by_zone(buildings, zones, zone_id_col)

    # 3. 合并zone基础信息
    zone_base = zones[[zone_id_col, 'area_km2']].copy()
    zone_base = zone_base.rename(columns={zone_id_col: 'zone_id', 'area_km2': 'area'})

    # 去除geometry列（如果是GeoDataFrame）
    if isinstance(zone_base, gpd.GeoDataFrame):
        zone_base = pd.DataFrame(zone_base.drop(columns='geometry'))

    land_use = zone_base.merge(
        zone_agg,
        left_on='zone_id',
        right_on=zone_id_col,
        how='left'
    )

    # 4. 填充缺失值（没有建筑的zone）
    fill_columns = ['hh', 'pop', 'emp_total', 'emp_office', 'emp_retail',
                    'emp_edu', 'emp_health', 'emp_industrial', 'emp_other',
                    'num_buildings']
    for col in fill_columns:
        if col in land_use.columns:
            land_use[col] = land_use[col].fillna(0)

    # 5. 计算派生字段

    # 人口密度和就业密度
    land_use['density'] = land_use['pop'] / land_use['area'].clip(lower=0.01)
    land_use['emp_density'] = land_use['emp_total'] / land_use['area'].clip(lower=0.01)

    # CBD标识（就业密度前20%）
    emp_density_threshold = land_use['emp_density'].quantile(0.8)
    land_use['is_cbd'] = (land_use['emp_density'] >= emp_density_threshold).astype(int)

    # 6. 选择和排序列
    output_columns = [
        'zone_id', 'area', 'hh', 'pop',
        'emp_total', 'emp_office', 'emp_retail', 'emp_edu',
        'emp_health', 'emp_industrial', 'emp_other',
        'density', 'emp_density', 'is_cbd'
    ]

    land_use = land_use[[col for col in output_columns if col in land_use.columns]]

    # 7. 四舍五入
    round_cols = ['hh', 'pop', 'emp_total', 'emp_office', 'emp_retail',
                  'emp_edu', 'emp_health', 'emp_industrial', 'emp_other']
    for col in round_cols:
        if col in land_use.columns:
            land_use[col] = land_use[col].round(0).astype(int)

    land_use[['density', 'emp_density']] = land_use[['density', 'emp_density']].round(2)

    # 8. 统计汇总
    logger.info("land_use表统计汇总:")
    logger.info(f"  总TAZ数: {len(land_use)}")
    logger.info(f"  总人口: {land_use['pop'].sum():,.0f}")
    logger.info(f"  总家庭: {land_use['hh'].sum():,.0f}")
    logger.info(f"  总就业: {land_use['emp_total'].sum():,.0f}")
    logger.info(f"    - 办公: {land_use['emp_office'].sum():,.0f}")
    logger.info(f"    - 零售: {land_use['emp_retail'].sum():,.0f}")
    logger.info(f"    - 教育: {land_use['emp_edu'].sum():,.0f}")
    logger.info(f"    - 医疗: {land_use['emp_health'].sum():,.0f}")
    logger.info(f"    - 工业: {land_use['emp_industrial'].sum():,.0f}")
    logger.info(f"    - 其他: {land_use['emp_other'].sum():,.0f}")
    logger.info(f"  CBD区域数: {land_use['is_cbd'].sum()}")

    return land_use


# ----------------------------------------------------------------------------
# 保存land_use表
# ----------------------------------------------------------------------------

def save_landuse_table(
        land_use: pd.DataFrame,
        output_path: Union[str, Path]
) -> Path:
    """
    保存land_use表为CSV

    参数:
        land_use: land_use DataFrame
        output_path: 输出文件路径

    返回:
        输出文件路径
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    write_csv(land_use, output_path)
    logger.info(f"已保存land_use表: {output_path}")

    return output_path


# ----------------------------------------------------------------------------
# 数据验证
# ----------------------------------------------------------------------------

def validate_landuse_table(land_use: pd.DataFrame) -> List[str]:
    """
    验证land_use表的完整性

    参数:
        land_use: land_use DataFrame

    返回:
        错误和警告信息列表
    """
    issues = []

    # 检查必需列
    required_columns = ['zone_id', 'hh', 'pop', 'emp_total']
    missing = validate_dataframe(land_use, required_columns, "land_use")
    if missing:
        issues.append(f"缺少必需列: {missing}")

    # 检查负值
    numeric_cols = ['hh', 'pop', 'emp_total', 'emp_office', 'emp_retail']
    for col in numeric_cols:
        if col in land_use.columns:
            if (land_use[col] < 0).any():
                issues.append(f"列'{col}'包含负值")

    # 检查zone_id唯一性
    if land_use['zone_id'].duplicated().any():
        issues.append("zone_id存在重复值")

    # 检查合理性
    if 'hh' in land_use.columns and 'pop' in land_use.columns:
        # 平均家庭规模应在1-10之间
        avg_hh_size = land_use['pop'].sum() / land_use['hh'].sum() if land_use['hh'].sum() > 0 else 0
        if avg_hh_size < 1 or avg_hh_size > 10:
            issues.append(f"平均家庭规模异常: {avg_hh_size:.2f}")

    if issues:
        for issue in issues:
            logger.warning(f"land_use验证问题: {issue}")
    else:
        logger.info("land_use表验证通过")

    return issues


# ----------------------------------------------------------------------------
# 完整流程
# ----------------------------------------------------------------------------

def create_landuse_from_osm(
        zones: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        output_path: Union[str, Path],
        coefficients: Optional[ConversionCoefficients] = None
) -> pd.DataFrame:
    """
    从OSM数据创建land_use表的完整流程

    参数:
        zones: TAZ GeoDataFrame
        buildings: 建筑GeoDataFrame
        output_path: 输出文件路径
        coefficients: 转换系数

    返回:
        land_use DataFrame
    """
    # 构建land_use表
    land_use = build_landuse_table(zones, buildings, coefficients)

    # 验证
    validate_landuse_table(land_use)

    # 保存
    save_landuse_table(land_use, output_path)

    return land_use