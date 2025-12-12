# ============================================================================
# activitysim_runner.py
# ============================================================================
# 模块职责：封装ActivitySim的执行流程
# 仅使用真正的ActivitySim库，不提供简化版本
# ============================================================================

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from .io_utils import (
    logger, ensure_dir, read_csv, write_csv, read_yaml, write_yaml, ProgressCallback
)

# ----------------------------------------------------------------------------
# ActivitySim 导入（必须成功）
# ----------------------------------------------------------------------------

try:
    import activitysim.abm  # noqa: F401
    from activitysim.core import workflow

    ACTIVITYSIM_AVAILABLE = True
    logger.info("[OK] ActivitySim 已成功加载")

except ImportError as e:
    ACTIVITYSIM_AVAILABLE = False
    workflow = None
    logger.error(f"[ERR] ActivitySim 导入失败: {e}")
    logger.error("请安装: pip install activitysim")


# ----------------------------------------------------------------------------
# 数据格式转换
# ----------------------------------------------------------------------------

def convert_households_to_activitysim(synthetic_households: pd.DataFrame, land_use: pd.DataFrame) -> pd.DataFrame:
    """将合成家庭数据转换为ActivitySim格式"""
    logger.info("转换家庭数据为ActivitySim格式...")

    households = synthetic_households.copy()

    # 若household_id存在重复，重新编号并保留原ID映射
    if households['household_id'].duplicated().any():
        households['original_household_id'] = households['household_id']
        households['household_id'] = range(1, len(households) + 1)

    if 'home_zone_id' in synthetic_households.columns:
        households['home_zone_id'] = synthetic_households['home_zone_id']
    elif 'TAZ' in synthetic_households.columns:
        households['home_zone_id'] = synthetic_households['TAZ']
    else:
        zones = land_use['zone_id'].values if 'zone_id' in land_use.columns else land_use.index.values
        weights = land_use['hh'].values if 'hh' in land_use.columns else np.ones(len(zones))
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(zones)) / len(zones)
        households['home_zone_id'] = np.random.choice(zones, size=len(households), p=weights)

    households['income'] = synthetic_households.get('income', 75000)
    households['hhsize'] = synthetic_households.get('hhsize', synthetic_households.get('hh_size', 2))
    households['HHT'] = synthetic_households.get('HHT', np.where(households['hhsize'] == 1,
                                                                 np.random.choice([4, 5], len(households)), 1))
    households['auto_ownership'] = synthetic_households.get('auto_ownership', synthetic_households.get('num_auto', 1))
    households['num_workers'] = synthetic_households.get('num_workers', 1)
    households['income_in_thousands'] = households['income'] / 1000
    households['sample_weight'] = synthetic_households.get('sample_weight', 1)

    # 兼容 ActivitySim configs 中的常用别名
    households['hincp'] = households['income']
    households['hht'] = households['HHT']

    # 根据 land_use 推导 areatype / CBDFlag（如果 land_use 未提供则用默认值）
    area_type_source = None
    if 'areatype' in land_use.columns:
        area_type_source = 'areatype'
    elif 'area_type' in land_use.columns:
        area_type_source = 'area_type'

    if area_type_source is not None:
        if 'zone_id' in land_use.columns:
            area_type_map = land_use.set_index('zone_id')[area_type_source]
        else:
            area_type_map = land_use[area_type_source]
        households['areatype'] = households['home_zone_id'].map(area_type_map).fillna(2).astype(int)
    else:
        households['areatype'] = 2

    households['CBDFlag'] = (households['areatype'] == 1).astype(int)

    logger.info(f"已转换 {len(households)} 户家庭")
    return households


def convert_persons_to_activitysim(synthetic_persons: pd.DataFrame, households: pd.DataFrame) -> pd.DataFrame:
    """将合成人口数据转换为ActivitySim格式"""
    logger.info("转换人口数据为ActivitySim格式...")

    persons = pd.DataFrame()
    # PopulationSim输出可能没有person_id（仅有household_id/PNUM），缺失时顺序补齐
    if 'person_id' in synthetic_persons.columns:
        persons['person_id'] = synthetic_persons['person_id']
    else:
        persons['person_id'] = range(1, len(synthetic_persons) + 1)

    # 如果households做了重编码，映射household_id
    if 'original_household_id' in households.columns:
        mapping = dict(zip(households['original_household_id'], households['household_id']))
        persons['household_id'] = synthetic_persons['household_id'].map(mapping)
    else:
        persons['household_id'] = synthetic_persons['household_id']
    persons['age'] = synthetic_persons.get('age', 35)
    persons['PNUM'] = synthetic_persons.get('PNUM', synthetic_persons.groupby('household_id').cumcount() + 1)
    persons['sex'] = synthetic_persons.get('sex', np.random.choice([1, 2], len(synthetic_persons)))

    # 就业状态
    if 'pemploy' in synthetic_persons.columns:
        persons['pemploy'] = synthetic_persons['pemploy']
    elif 'worker' in synthetic_persons.columns:
        persons['pemploy'] = np.where(synthetic_persons['worker'] == 1, 1, 4)
    else:
        persons['pemploy'] = np.where(
            (persons['age'] >= 18) & (persons['age'] < 65),
            np.random.choice([1, 2, 4], len(persons), p=[0.6, 0.15, 0.25]),
            4
        )

    # 学生状态
    if 'pstudent' in synthetic_persons.columns:
        persons['pstudent'] = synthetic_persons['pstudent']
    else:
        persons['pstudent'] = np.where(
            persons['age'] < 5, 1,
            np.where(persons['age'] < 18, 2,
                     np.where((persons['age'] < 25) & (np.random.random(len(persons)) < 0.4), 3, 4))
        )

    # 人员类型
    if 'ptype' in synthetic_persons.columns:
        persons['ptype'] = synthetic_persons['ptype']
    else:
        persons['ptype'] = _compute_ptype(persons['age'], persons['pemploy'], persons['pstudent'])

    # 合并家庭层面字段，供后续模型使用
    merge_cols = [c for c in [
        'household_id', 'home_zone_id', 'income', 'income_in_thousands',
        'auto_ownership', 'num_workers', 'areatype'
    ] if c in households.columns]
    persons = persons.merge(households[merge_cols], on='household_id', how='left')

    # ActivitySim 一些 configs 仍会使用 agep 作为年龄字段
    persons['agep'] = persons['age']
    persons['workplace_zone_id'] = -1
    persons['school_zone_id'] = -1
    persons['has_license'] = np.where(persons['age'] >= 16, 1, 0)

    # 基础就学/就业/性别/年龄分组 dummy（用 0/1，避免 CSV 读入变成字符串布尔）
    persons['is_worker'] = persons['pemploy'].isin([1, 2]).astype(int)
    persons['is_fulltime_worker'] = (persons['pemploy'] == 1).astype(int)
    persons['is_parttime_worker'] = (persons['pemploy'] == 2).astype(int)

    persons['is_student'] = persons['pstudent'].isin([1, 2, 3]).astype(int)
    persons['is_k12'] = (persons['pstudent'] == 2).astype(int)
    persons['is_university'] = (persons['pstudent'] == 3).astype(int)

    persons['is_predrive'] = (persons['age'] <= 15).astype(int)
    persons['is_driving_age'] = (persons['age'] > 15).astype(int)
    persons['is_nondriving_age'] = (persons['age'] < 16).astype(int)

    persons['is_adult'] = (persons['age'] >= 18).astype(int)
    persons['adult'] = persons['is_adult']
    persons['is_female'] = (persons['sex'] == 2).astype(int)
    persons['is_male'] = (persons['sex'] == 1).astype(int)

    # 收入相关 dummy（供 school/work/location、mode choice、tour scheduling 使用）
    income = persons.get('income', pd.Series(0, index=persons.index)).fillna(0)
    if 'income_in_thousands' not in persons.columns:
        persons['income_in_thousands'] = income / 1000.0

    persons['is_low_income'] = (income < 20000).astype(int)
    persons['is_medium_income'] = ((income >= 20000) & (income < 50000)).astype(int)
    persons['is_high_income'] = ((income >= 50000) & (income < 100000)).astype(int)
    persons['is_very_high_income'] = (income >= 100000).astype(int)

    persons['is_income_less25K'] = (income < 25000).astype(int)
    persons['is_income_25K_to_60K'] = ((income >= 25000) & (income < 60000)).astype(int)
    persons['is_income_60K_to_120K'] = ((income >= 60000) & (income < 120000)).astype(int)
    persons['is_income_greater120K'] = (income >= 120000).astype(int)

    # school_segment / work_segment（缺失时按规则/随机生成，保证模型可运行）
    school_segment = pd.Series(0, index=persons.index, dtype=int)
    school_segment[(persons['is_k12'] == 1) & (persons['is_predrive'] == 1)] = 1   # k12_predrive
    school_segment[(persons['is_k12'] == 1) & (persons['is_driving_age'] == 1)] = 2  # k12_drive
    school_segment[persons['is_university'] == 1] = 3  # univ
    persons['school_segment'] = school_segment

    persons['work_segment'] = 0
    worker_mask = persons['is_worker'] == 1
    if worker_mask.any():
        persons.loc[worker_mask, 'work_segment'] = np.random.choice(
            [1, 2, 3, 4, 5],
            size=int(worker_mask.sum()),
            p=[0.30, 0.20, 0.10, 0.20, 0.20]
        )

    # household 组合 dummy（放在 persons 里供后续 tours/trips 继承）
    full_counts = persons.groupby('household_id')['is_fulltime_worker'].sum()
    part_counts = persons.groupby('household_id')['is_parttime_worker'].sum()
    predrive_child_counts = persons.groupby('household_id').apply(lambda df: (df['ptype'] == 7).sum())
    nonworker_adult_counts = persons.groupby('household_id').apply(lambda df: (df['ptype'] == 4).sum())
    adult_counts = persons.groupby('household_id')['is_adult'].sum()
    all_adults_full_time = (adult_counts > 0) & (full_counts == adult_counts)

    persons['num_full'] = persons['household_id'].map(full_counts).fillna(0).astype(int)
    persons['num_part'] = persons['household_id'].map(part_counts).fillna(0).astype(int)
    persons['is_pre_drive_child_in_HH'] = (persons['household_id'].map(predrive_child_counts).fillna(0) > 0).astype(int)
    persons['is_non_worker_in_HH'] = (persons['household_id'].map(nonworker_adult_counts).fillna(0) > 0).astype(int)
    persons['is_all_adults_full_time_workers'] = persons['household_id'].map(all_adults_full_time).fillna(False).astype(int)

    # joint tours 本项目未启用时给默认列，避免后续预处理报错
    if 'num_joint_tours' not in persons.columns:
        persons['num_joint_tours'] = 0
    if 'num_mand_tours' not in persons.columns:
        persons['num_mand_tours'] = 0

    # 免费停车 dummy（未建模时默认为无）
    if 'free_parking_at_work' not in persons.columns:
        persons['free_parking_at_work'] = 0

    # 注意：ActivitySim 的 initialize_households 会根据 pemploy/pstudent
    # 生成布尔型 is_worker/is_student，并在 location_choice 中作为布尔掩码使用。
    # 如果输入 CSV 里已有 0/1 的同名列，ActivitySim 可能不会覆盖，
    # 从而导致 school/work location 里把整数掩码当列名索引报错。
    persons = persons.drop(columns=['is_worker', 'is_student'], errors='ignore')

    logger.info(f"已转换 {len(persons)} 人")
    return persons


def _compute_ptype(ages: pd.Series, pemploy: pd.Series, pstudent: pd.Series) -> pd.Series:
    """计算人员类型 ptype"""
    ptype = pd.Series(4, index=ages.index, dtype=int)
    ptype[ages <= 5] = 8
    ptype[(ages > 5) & (ages < 16)] = 7
    ptype[(ages >= 16) & (ages < 18) & (pemploy != 1)] = 6
    ptype[pstudent == 3] = 3
    ptype[pemploy == 1] = 1
    ptype[pemploy == 2] = 2
    ptype[(ages >= 65) & (pemploy == 4)] = 5
    return ptype


def convert_landuse_to_activitysim(land_use: pd.DataFrame) -> pd.DataFrame:
    """将土地利用数据转换为ActivitySim格式"""
    logger.info("转换土地利用数据为ActivitySim格式...")

    lu = pd.DataFrame()
    if 'zone_id' in land_use.columns:
        lu['zone_id'] = land_use['zone_id']
    else:
        lu['zone_id'] = land_use.index
    lu['TAZ'] = lu['zone_id']
    lu['DISTRICT'] = land_use.get('DISTRICT', 1)
    lu['SD'] = land_use.get('SD', 1)
    lu['COUNTY'] = land_use.get('COUNTY', 1)

    # 基础人口、就业字段（保持原始命名以满足 configs 初始化要求）
    lu['area'] = land_use.get('area', land_use.get('area_km2', 1))
    lu['hh'] = land_use.get('hh', land_use.get('TOTHH', 100))
    lu['pop'] = land_use.get('pop', land_use.get('TOTPOP', 300))
    lu['emp_total'] = land_use.get('emp_total', land_use.get('TOTEMP', 500))
    lu['emp_office'] = land_use.get('emp_office', lu['emp_total'] * 0.25)
    lu['emp_retail'] = land_use.get('emp_retail', lu['emp_total'] * 0.2)
    lu['emp_edu'] = land_use.get('emp_edu', 0)
    lu['emp_health'] = land_use.get('emp_health', 0)
    lu['emp_industrial'] = land_use.get('emp_industrial', lu['emp_total'] * 0.18)
    lu['emp_other'] = land_use.get('emp_other', lu['emp_total'] * 0.2)
    lu['density'] = land_use.get('density', (lu['hh'] + lu['emp_total']) / lu['area'].clip(lower=0.01))
    lu['is_cbd'] = land_use.get('is_cbd', 0)

    # ActivitySim 需要的典型字段（与原模型表头兼容）
    lu['TOTHH'] = lu['hh']
    lu['TOTPOP'] = lu['pop']

    lu['TOTACRE'] = lu['area'] * 247.105
    lu['RESACRE'] = lu['TOTACRE'] * 0.4
    lu['CIACRE'] = lu['TOTACRE'] * 0.3

    lu['TOTEMP'] = lu['emp_total']
    lu['RETEMPN'] = lu['emp_retail']
    lu['FPSEMPN'] = lu['emp_office']
    lu['HEREMPN'] = lu['emp_health'] + lu['emp_edu'] + lu['TOTEMP'] * 0.15
    lu['OTHEMPN'] = lu['emp_other']
    lu['AGREMPN'] = lu['TOTEMP'] * 0.02
    lu['MWTEMPN'] = lu['emp_industrial']

    lu['AGE0519'] = lu['TOTPOP'] * 0.18
    lu['HSENROLL'] = lu['AGE0519'] * 0.25
    lu['COLLFTE'] = lu['TOTPOP'] * 0.05
    lu['COLLPTE'] = lu['TOTPOP'] * 0.02

    lu['PRKCST'] = land_use.get('PRKCST', 0)
    lu['OPRKCST'] = land_use.get('OPRKCST', 0)
    lu['TOPOLOGY'] = land_use.get('TOPOLOGY', 1)
    lu['TERMINAL'] = land_use.get('TERMINAL', 0)

    if 'area_type' in land_use.columns:
        lu['area_type'] = land_use['area_type']
    elif 'is_cbd' in land_use.columns:
        lu['area_type'] = np.where(land_use['is_cbd'] == 1, 1, 2)
    else:
        lu['area_type'] = 2

    # 为可达性模型提供简化字段
    lu['retail'] = lu['RETEMPN']
    lu['service'] = lu['OTHEMPN']
    lu['emp'] = lu['TOTEMP']

    # 兼容 configs 中常用字段（缺失时给默认值）
    lu['areatype'] = lu.get('area_type', land_use.get('areatype', 2)).fillna(2).astype(int)
    lu['CBDFlag'] = (lu['areatype'] == 1).astype(int)
    lu['acres'] = lu.get('TOTACRE', lu['area'] * 247.105)
    lu['hshld'] = lu['hh']

    # 学校/大学规模字段（destination_choice_size_terms 需要）
    pop_series = lu.get('pop', 0).astype(float)
    if pop_series.sum() <= 0:
        logger.warning("land_use 的 pop 全为 0，无法计算学校规模；使用 1 作为默认人口规模以保证模型可运行。")
        pop_series = pd.Series(1.0, index=lu.index)

    if 'univ' not in lu.columns or (lu.get('univ', 0) == 0).all():
        # 以教育就业和人口粗略推导大学容量
        lu['univ'] = (lu.get('emp_edu', 0) * 5 + pop_series * 0.05).round(0).astype(int)
    if 'EnrollDS' not in lu.columns or (lu.get('EnrollDS', 0) == 0).all():
        # K12 学生（约5-19岁）占比
        lu['EnrollDS'] = (pop_series * 0.18).round(0).astype(int)
    if 'EnrollPD' not in lu.columns or (lu.get('EnrollPD', 0) == 0).all():
        # 学前儿童（约0-5岁）占比
        lu['EnrollPD'] = (pop_series * 0.06).round(0).astype(int)

    # 工作地规模字段（按 NAICS 汇总的 N* 类别）；若无则按 emp_total 均分
    naics_cols = [
        'N11', 'N21', 'N22', 'N23', 'N313233', 'N42', 'N4445', 'N4849',
        'N51', 'N52', 'N53', 'N54', 'N55', 'N56', 'N61', 'N62',
        'N71', 'N72', 'N81', 'N92'
    ]
    total_emp = lu.get('emp_total', lu.get('TOTEMP', 0)).astype(float)
    if total_emp.sum() <= 0:
        logger.warning("land_use 的 emp_total 全为 0，无法计算工作地规模；使用 1 作为默认就业规模以保证模型可运行。")
        total_emp = pd.Series(1.0, index=lu.index)
    missing_naics = [c for c in naics_cols if c not in land_use.columns]
    if missing_naics:
        # 均匀分配到各类别，避免 size terms 全为 0
        share = 1.0 / len(naics_cols)
        for c in naics_cols:
            lu[c] = (total_emp * share).round(2)
    else:
        for c in naics_cols:
            lu[c] = land_use[c]

    # county/COUNTY 兼容
    if 'county' not in lu.columns:
        lu['county'] = lu['COUNTY']

    for col in [
        'PARKING_ZONE', 'PARKTOT', 'PARKLNG', 'PARKRATE', 'PROPFREE',
        'RetailEmp30', 'I_PCTLT10K', 'I_PCT10TO20', 'I_PCT20TO40', 'I_PCTGT40'
    ]:
        if col in land_use.columns:
            lu[col] = land_use[col]
        else:
            lu[col] = 0

    logger.info(f"已转换 {len(lu)} 个zone")
    return lu


# ----------------------------------------------------------------------------
# ActivitySim 配置生成
# ----------------------------------------------------------------------------

def generate_activitysim_configs(config_dir: Path, data_dir: Path, output_dir: Path, num_zones: int,
                                 sample_rate: float) -> None:
    """生成ActivitySim配置文件"""
    logger.info("生成ActivitySim配置...")
    ensure_dir(config_dir)

    settings = {
        'inherit_settings': True,
        'households_sample_size': 0,
        'chunk_size': 0,
        'num_processes': 1,
        'input_table_list': [
            {'tablename': 'households', 'filename': 'households.csv', 'index_col': 'household_id'},
            {'tablename': 'persons', 'filename': 'persons.csv', 'index_col': 'person_id'},
            {'tablename': 'land_use', 'filename': 'land_use.csv', 'index_col': 'zone_id'},
        ],
        'skims_file': 'skims.omx',
        'logging_config_file': 'logging.yaml',
        'output_tables': {'action': 'include', 'tables': ['households', 'persons', 'tours', 'trips']},
        'models': [
            'initialize_landuse',
            'initialize_households',
            'school_location',
            'workplace_location',
            'auto_ownership_simulate',
            'cdap_simulate',
            'mandatory_tour_frequency',
            'mandatory_tour_scheduling',
            'non_mandatory_tour_frequency',
            'non_mandatory_tour_destination',
            'non_mandatory_tour_scheduling',
            'tour_mode_choice_simulate',
            'stop_frequency',
            'trip_purpose',
            'trip_destination',
            'trip_scheduling',
            'trip_mode_choice',
            'write_tables',
        ],
    }
    write_yaml(settings, config_dir / 'settings.yaml')

    network_los = {
        'zone_system': 1,
        # 必需字段：时间划分
        'skim_time_periods': {
            # 总时间窗（分钟），这里用24小时
            'time_window': 24 * 60,
            # 分辨率（分钟）
            'period_minutes': 60,
            # 分段切点（小时制），与labels一一对应
            'periods': [0, 6, 10, 15, 19, 24],
            'labels': ['EA', 'AM', 'MD', 'PM', 'EV'],
        },
        # 简化：单个时段/全局 skim 使用 skims.omx
        'taz_skims': 'skims.omx',
    }
    write_yaml(network_los, config_dir / 'network_los.yaml')

    # 生成日志配置，写到当前输出目录，避免固定路径无法创建
    logging_config = {
        'logging': {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S',
                },
                'simple': {'format': '%(levelname)s - %(message)s'},
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'simple',
                    'stream': 'ext://sys.stdout',
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'level': 'DEBUG',
                    'formatter': 'standard',
                    'filename': str(output_dir / 'activitysim.log'),
                    'mode': 'w',
                    'encoding': 'utf-8',
                },
            },
            'loggers': {
                'activitysim': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file'],
                    'propagate': False,
                },
            },
            'root': {'level': 'WARNING', 'handlers': ['console']},
        }
    }
    write_yaml(logging_config, config_dir / 'logging.yaml')

    logger.info(f"ActivitySim配置已生成: {config_dir}")


# ----------------------------------------------------------------------------
# ActivitySim 运行
# ----------------------------------------------------------------------------

def run_activitysim(
        config_dir: Union[str, Path],
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        synthetic_households: pd.DataFrame,
        synthetic_persons: pd.DataFrame,
        land_use: pd.DataFrame,
        skims_path: Union[str, Path],
        random_seed: int = 1,
        sample_rate: float = 1.0,
        progress_callback: Optional[ProgressCallback] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """运行ActivitySim"""

    if not ACTIVITYSIM_AVAILABLE:
        raise RuntimeError(
            "ActivitySim 库不可用！\n"
            "请安装: pip install activitysim\n"
            "或检查安装是否正确"
        )

    # 统一使用绝对路径，避免工作目录变化或特殊字符导致路径失效
    config_dir = Path(config_dir).resolve()
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    ensure_dir(config_dir)
    ensure_dir(data_dir)
    ensure_dir(output_dir)

    logger.info("=" * 60)
    logger.info("★★★ ActivitySim 运行开始 ★★★")
    logger.info("=" * 60)

    if progress_callback:
        progress_callback.update(message="准备ActivitySim输入数据...")

    # 转换数据
    households = convert_households_to_activitysim(synthetic_households, land_use)
    persons = convert_persons_to_activitysim(synthetic_persons, households)
    # 只保留互相匹配的household_id，避免“households with no persons”
    valid_hh_ids = set(households['household_id'].unique())
    persons = persons[persons['household_id'].isin(valid_hh_ids)]
    hh_with_persons = set(persons['household_id'].unique())
    households = households[households['household_id'].isin(hh_with_persons)]
    households = households.reset_index(drop=True)
    persons = persons.reset_index(drop=True)

    lu = convert_landuse_to_activitysim(land_use)

    # 保存输入
    write_csv(households, data_dir / 'households.csv')
    write_csv(persons, data_dir / 'persons.csv')
    write_csv(lu, data_dir / 'land_use.csv')

    if Path(skims_path).exists():
        shutil.copy(skims_path, data_dir / 'skims.omx')
        # 将 TIME/DIST 简化 skim 扩展为 ActivitySim configs 所需的矩阵名称
        try:
            ensure_activitysim_compatible_skims(data_dir / 'skims.omx', config_dir / 'configs')
        except Exception as e:
            logger.warning(f"Skim 兼容化处理失败，可能导致后续模型缺少矩阵: {e}")

    # 生成配置
    generate_activitysim_configs(config_dir, data_dir, output_dir, len(lu), sample_rate)

    if progress_callback:
        progress_callback.update(message="运行 ActivitySim...")

    np.random.seed(random_seed)

    # 运行 ActivitySim
    logger.info("调用 ActivitySim workflow...")

    # 强制默认文本编码为 UTF-8，避免 Windows 默认 GBK 导致配置解析失败
    import builtins
    _orig_open = builtins.open
    def _open_utf8(file, mode='r', buffering=-1, encoding=None, *args, **kwargs):
        if encoding is None and 'b' not in mode:
            encoding = 'utf-8'
        return _orig_open(file, mode, buffering, encoding=encoding, *args, **kwargs)
    builtins.open = _open_utf8

    # Use project overrides in config_dir first, then fall back to configs/
    state = workflow.State.make_default(
        configs_dir=[str(config_dir), str(config_dir / 'configs')],
        data_dir=str(data_dir),
        output_dir=str(output_dir)
    )

    try:
        state.run.all()
    finally:
        builtins.open = _orig_open

    logger.info("★★★ ActivitySim 运行成功 ★★★")

    if progress_callback:
        progress_callback.update(message="读取ActivitySim输出...")

    # 读取输出
    tours = None
    trips = None

    for tours_file in [output_dir / 'final_tours.csv', output_dir / 'tours.csv']:
        if tours_file.exists():
            tours = read_csv(tours_file)
            logger.info(f"读取tours: {len(tours)}条")
            break

    for trips_file in [output_dir / 'final_trips.csv', output_dir / 'trips.csv']:
        if trips_file.exists():
            trips = read_csv(trips_file)
            logger.info(f"读取trips: {len(trips)}条")
            break

    if tours is None or trips is None:
        raise RuntimeError("未找到ActivitySim输出文件")

    return tours, trips


# ----------------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------------

def ensure_activitysim_compatible_skims(omx_path: Path, configs_dir: Path) -> None:
    """
    为简化 Skim 文件补齐 ActivitySim configs 中引用的矩阵名称。

    说明：
    - 本项目的 skim 可能只有 TIME/DIST 两个矩阵；
      但 configs 中会引用大量 SOV/HOV/TRANSIT/TOLL 矩阵。
    - 这里通过 HDF5 hard-link 的方式把缺失矩阵指向 TIME/DIST 或常量 0，
      以避免生成巨大重复文件。
    - Transit / Toll 相关矩阵默认指向 0（等价于不可用/无费用）。
    """
    import re
    import h5py

    omx_path = Path(omx_path)
    configs_dir = Path(configs_dir)
    if not omx_path.exists():
        raise FileNotFoundError(f"Skims 文件不存在: {omx_path}")

    # 收集 configs 中引用的 skim 名称
    pat_period = re.compile(r"\(\s*'([A-Z0-9_]+)'\s*,\s*'([A-Z]{2})'\s*\)")
    pat_simple = re.compile(r"\['([A-Z0-9_]+)'\]")
    required_names = set()
    required_periods = set()

    if configs_dir.exists():
        for csv_path in configs_dir.rglob("*.csv"):
            txt = csv_path.read_text(encoding="utf-8", errors="ignore")
            for m in pat_period.finditer(txt):
                required_names.add(m.group(1))
                required_periods.add(m.group(2))
            for m in pat_simple.finditer(txt):
                name = m.group(1)
                if name.startswith(("SOV_", "HOV2_", "HOV3_", "WLK_", "PNR_", "KNR_")):
                    required_names.add(name)

    # 默认时间分段（与 generate_activitysim_configs 的 network_los 一致）
    period_labels = {"EA", "AM", "MD", "PM", "EV"}
    if required_periods:
        period_labels |= required_periods

    with h5py.File(omx_path, "a") as f:
        data_group = f["data"] if "data" in f else f.require_group("data")

        def _get_ds(name: str):
            if name in data_group:
                return data_group[name]
            if name in f:
                return f[name]
            return None

        time_ds = _get_ds("TIME") or _get_ds("SOV_FREE_TIME") or _get_ds("SOV_FREE_TIME__MD")
        dist_ds = _get_ds("DIST") or _get_ds("SOV_FREE_DISTANCE") or _get_ds("SOV_FREE_DISTANCE__MD")
        if time_ds is None or dist_ds is None:
            raise ValueError("Skims 文件缺少 TIME/DIST（或 SOV_FREE_TIME/DISTANCE）基础矩阵")

        # 常量 0 矩阵（只存元数据，避免大文件）
        zero_name = "__CONST_ZERO__"
        if zero_name not in data_group:
            data_group.create_dataset(
                zero_name,
                shape=time_ds.shape,
                dtype="float32",
                fillvalue=0.0,
            )
        zero_ds = data_group[zero_name]

        def _link(target: str, source_ds) -> None:
            if target in data_group:
                return
            data_group[target] = source_ds  # hard-link，不复制数据

        for name in required_names:
            # 选择来源矩阵
            if name.startswith(("WLK_", "PNR_", "KNR_")):
                source = zero_ds  # 关闭公交/轨道等 transit
            elif "_TOLL" in name or name.endswith("FARE"):
                source = zero_ds  # 关闭收费/票价
            elif "DISTANCE" in name or "TOLLDISTANCE" in name:
                source = dist_ds
            else:
                source = time_ds

            _link(name, source)
            for label in period_labels:
                _link(f"{name}__{label}", source)

        logger.info(f"已在 skim 文件中补齐 {len(required_names)} 类矩阵名称（hard-link）")

def compute_od_matrix_from_trips(trips: pd.DataFrame, zones: List[int]) -> pd.DataFrame:
    """从trips表计算OD矩阵"""
    origin_col = 'origin' if 'origin' in trips.columns else 'origin_zone_id'
    dest_col = 'destination' if 'destination' in trips.columns else 'destination_zone_id'

    od_counts = trips.groupby([origin_col, dest_col]).size().reset_index(name='trips')
    od_matrix = od_counts.pivot(index=origin_col, columns=dest_col, values='trips').fillna(0)
    od_matrix = od_matrix.reindex(index=zones, columns=zones, fill_value=0)

    return od_matrix
