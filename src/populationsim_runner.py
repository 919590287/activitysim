# ============================================================================
# populationsim_runner.py
# ============================================================================
# 模块职责：封装PopulationSim的执行流程
# 仅使用真正的PopulationSim库
# ============================================================================

import os
import shutil
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from .io_utils import (
    logger, ensure_dir, read_csv, write_csv, read_yaml,
    write_yaml, ProgressCallback
)

# ----------------------------------------------------------------------------
# PopulationSim 导入
# ----------------------------------------------------------------------------

POPULATIONSIM_AVAILABLE = False
_popsim_run = None
_popsim_add_args = None

try:
    import populationsim

    if hasattr(populationsim, 'run') and callable(populationsim.run):
        _popsim_run = populationsim.run
        _popsim_add_args = getattr(populationsim, 'add_run_args', None)
        POPULATIONSIM_AVAILABLE = True
        logger.info("[OK] PopulationSim 已成功加载")
    else:
        raise ImportError("PopulationSim 缺少 run 函数")

except ImportError as e:
    logger.error(f"[ERR] PopulationSim 导入失败: {e}")
    logger.error("请安装: pip install populationsim")


# ----------------------------------------------------------------------------
# 清理函数
# ----------------------------------------------------------------------------

def cleanup_populationsim_state():
    """清理 PopulationSim 状态"""
    try:
        from populationsim.core import pipeline as popsim_pipeline
        if popsim_pipeline.is_open():
            popsim_pipeline.close_pipeline()
    except:
        pass

    try:
        import orca
        orca.clear_all()
    except:
        pass


# ----------------------------------------------------------------------------
# 默认概率分布表
# ----------------------------------------------------------------------------

def get_default_probability_tables() -> Dict[str, pd.DataFrame]:
    """获取默认的概率分布表"""
    prob_tables = {}

    prob_tables['hh_size_dist'] = pd.DataFrame({
        'hh_size': [1, 2, 3, 4, 5, 6, 7],
        'hhsize': [1, 2, 3, 4, 5, 6, 7],
        'probability': [0.28, 0.34, 0.15, 0.13, 0.06, 0.03, 0.01],
        'cumulative_prob': [0.28, 0.62, 0.77, 0.90, 0.96, 0.99, 1.00]
    })

    prob_tables['income_dist'] = pd.DataFrame({
        'income_cat': [1, 2, 3, 4, 5],
        'income_min': [0, 25000, 50000, 100000, 150000],
        'income_max': [25000, 50000, 100000, 150000, 500000],
        'probability': [0.20, 0.25, 0.30, 0.15, 0.10],
        'cumulative_prob': [0.20, 0.45, 0.75, 0.90, 1.00]
    })

    prob_tables['hht_dist'] = pd.DataFrame({
        'HHT': [1, 2, 3, 4, 5, 6, 7],
        'probability': [0.48, 0.05, 0.12, 0.12, 0.14, 0.05, 0.04],
        'cumulative_prob': [0.48, 0.53, 0.65, 0.77, 0.91, 0.96, 1.00]
    })

    auto_data = []
    for income_cat in [1, 2, 3, 4, 5]:
        for hh_size_group in [1, 2, 3]:
            if income_cat <= 2:
                probs = {1: [0.35, 0.50, 0.15, 0.00], 2: [0.15, 0.50, 0.30, 0.05], 3: [0.10, 0.40, 0.40, 0.10]}
            elif income_cat <= 3:
                probs = {1: [0.15, 0.55, 0.25, 0.05], 2: [0.05, 0.40, 0.40, 0.15], 3: [0.03, 0.27, 0.50, 0.20]}
            else:
                probs = {1: [0.05, 0.45, 0.40, 0.10], 2: [0.02, 0.25, 0.50, 0.23], 3: [0.01, 0.15, 0.49, 0.35]}
            for num_auto, prob in enumerate(probs[hh_size_group]):
                auto_data.append({
                    'income_cat': income_cat, 'hh_size_group': hh_size_group,
                    'auto_ownership': num_auto, 'num_auto': num_auto, 'probability': prob
                })
    prob_tables['auto_ownership'] = pd.DataFrame(auto_data)

    prob_tables['age_dist'] = pd.DataFrame({
        'age_cat': [1, 2, 3, 4, 5, 6],
        'age_min': [0, 5, 16, 18, 25, 65],
        'age_max': [4, 15, 17, 24, 64, 100],
        'age_group_name': ['preschool', 'child', 'teen', 'young_adult', 'adult', 'senior'],
        'probability': [0.06, 0.12, 0.04, 0.10, 0.52, 0.16],
        'cumulative_prob': [0.06, 0.18, 0.22, 0.32, 0.84, 1.00]
    })

    pemploy_data = [
        {'age_cat': 1, 'pemploy': 4, 'probability': 1.00},
        {'age_cat': 2, 'pemploy': 4, 'probability': 1.00},
        {'age_cat': 3, 'pemploy': 2, 'probability': 0.25},
        {'age_cat': 3, 'pemploy': 4, 'probability': 0.75},
        {'age_cat': 4, 'pemploy': 1, 'probability': 0.40},
        {'age_cat': 4, 'pemploy': 2, 'probability': 0.25},
        {'age_cat': 4, 'pemploy': 3, 'probability': 0.05},
        {'age_cat': 4, 'pemploy': 4, 'probability': 0.30},
        {'age_cat': 5, 'pemploy': 1, 'probability': 0.62},
        {'age_cat': 5, 'pemploy': 2, 'probability': 0.13},
        {'age_cat': 5, 'pemploy': 3, 'probability': 0.05},
        {'age_cat': 5, 'pemploy': 4, 'probability': 0.20},
        {'age_cat': 6, 'pemploy': 1, 'probability': 0.10},
        {'age_cat': 6, 'pemploy': 2, 'probability': 0.05},
        {'age_cat': 6, 'pemploy': 4, 'probability': 0.85},
    ]
    prob_tables['pemploy_dist'] = pd.DataFrame(pemploy_data)

    pstudent_data = [
        {'age_cat': 1, 'pstudent': 1, 'probability': 1.00},
        {'age_cat': 2, 'pstudent': 2, 'probability': 0.95},
        {'age_cat': 2, 'pstudent': 4, 'probability': 0.05},
        {'age_cat': 3, 'pstudent': 2, 'probability': 0.90},
        {'age_cat': 3, 'pstudent': 4, 'probability': 0.10},
        {'age_cat': 4, 'pstudent': 3, 'probability': 0.50},
        {'age_cat': 4, 'pstudent': 4, 'probability': 0.50},
        {'age_cat': 5, 'pstudent': 3, 'probability': 0.05},
        {'age_cat': 5, 'pstudent': 4, 'probability': 0.95},
        {'age_cat': 6, 'pstudent': 4, 'probability': 1.00},
    ]
    prob_tables['pstudent_dist'] = pd.DataFrame(pstudent_data)

    prob_tables['sex_dist'] = pd.DataFrame({
        'sex': [1, 2],
        'sex_name': ['male', 'female'],
        'probability': [0.49, 0.51],
        'cumulative_prob': [0.49, 1.00]
    })

    prob_tables['ptype_mapping'] = pd.DataFrame({
        'age_cat': [1, 2, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6],
        'pemploy': [4, 4, 4, 1, 2, 4, 1, 2, 4, 4, 4, 1],
        'pstudent': [1, 2, 2, 4, 4, 3, 4, 4, 4, 4, 4, 4],
        'ptype': [8, 7, 6, 1, 2, 3, 1, 2, 4, 4, 5, 1],
    })

    return prob_tables


# ----------------------------------------------------------------------------
# 虚拟种子生成器
# ----------------------------------------------------------------------------

class VirtualSeedGenerator:
    """从概率分布表生成虚拟种子数据"""

    def __init__(self, prob_tables: Optional[Dict[str, pd.DataFrame]] = None, random_seed: int = 1):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.prob_tables = self._normalize_prob_tables(prob_tables) if prob_tables else get_default_probability_tables()

    def _normalize_prob_tables(self, prob_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        defaults = get_default_probability_tables()
        normalized = {}
        key_mapping = {
            'hh_size_dist': ['hh_size_dist', 'hh_size_dist.csv'],
            'income_dist': ['income_dist', 'income_dist.csv'],
            'hht_dist': ['hht_dist', 'hht_dist.csv'],
            'auto_ownership': ['auto_ownership', 'auto_ownership.csv'],
            'age_dist': ['age_dist', 'age_dist.csv'],
            'pemploy_dist': ['pemploy_dist', 'worker_student'],
            'pstudent_dist': ['pstudent_dist', 'pstudent_dist.csv'],
            'sex_dist': ['sex_dist', 'sex_dist.csv'],
            'ptype_mapping': ['ptype_mapping', 'ptype_mapping.csv'],
        }
        for standard_key, possible_keys in key_mapping.items():
            found = False
            for key in possible_keys:
                if key in prob_tables:
                    normalized[standard_key] = prob_tables[key].copy()
                    found = True
                    break
            if not found:
                normalized[standard_key] = defaults[standard_key]
        return normalized

    def generate_virtual_seed(self, num_households: int = 1000, land_use: Optional[pd.DataFrame] = None) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        logger.info(f"正在生成 {num_households} 个虚拟种子家庭...")
        households = self._generate_households(num_households, land_use)
        persons = self._generate_persons(households)
        workers_count = persons[persons['pemploy'].isin([1, 2])].groupby('household_id').size()
        households['num_workers'] = households['household_id'].map(workers_count).fillna(0).astype(int)
        logger.info(f"虚拟种子生成完成: {len(households)}户, {len(persons)}人")
        return households, persons

    def _generate_households(self, num_households: int, land_use: Optional[pd.DataFrame]) -> pd.DataFrame:
        households = []
        hh_size_dist = self.prob_tables['hh_size_dist']
        income_dist = self.prob_tables['income_dist']
        hht_dist = self.prob_tables['hht_dist']
        auto_dist = self.prob_tables['auto_ownership']

        if land_use is not None and len(land_use) > 0:
            zones = land_use['zone_id'].values if 'zone_id' in land_use.columns else land_use.index.values
            zone_weights = land_use['hh'].values.astype(float) if 'hh' in land_use.columns else np.ones(len(zones))
            zone_weights = np.maximum(zone_weights, 0)
            if zone_weights.sum() == 0:
                zone_weights = np.ones(len(zones))
            zone_weights = zone_weights / zone_weights.sum()
        else:
            zones = np.array([1])
            zone_weights = np.array([1.0])

        for hh_id in range(1, num_households + 1):
            hhsize = self._sample_from_distribution(hh_size_dist, 'hh_size')
            hht = self._sample_from_distribution(hht_dist, 'HHT')
            if hhsize == 1 and hht not in [4, 5]:
                hht = np.random.choice([4, 5])
            elif hhsize > 1 and hht in [4, 5]:
                hht = np.random.choice([1, 2, 3, 6, 7], p=[0.6, 0.1, 0.15, 0.1, 0.05])

            income_cat = self._sample_from_distribution(income_dist, 'income_cat')
            income_row = income_dist[income_dist['income_cat'] == income_cat].iloc[0]
            income = np.random.uniform(income_row['income_min'], income_row['income_max'])

            hh_size_group = 1 if hhsize <= 2 else (2 if hhsize <= 4 else 3)
            auto_ownership = self._sample_auto_ownership(auto_dist, income_cat, hh_size_group)
            home_zone_id = np.random.choice(zones, p=zone_weights)

            households.append({
                'household_id': hh_id, 'home_zone_id': int(home_zone_id), 'hhsize': hhsize,
                'HHT': hht, 'income': int(income), 'income_in_thousands': income / 1000,
                'auto_ownership': auto_ownership, 'num_workers': 0, 'sample_weight': 1
            })
        return pd.DataFrame(households)

    def _generate_persons(self, households: pd.DataFrame) -> pd.DataFrame:
        persons = []
        person_id = 1
        age_dist = self.prob_tables['age_dist']
        sex_dist = self.prob_tables['sex_dist']
        pemploy_dist = self.prob_tables['pemploy_dist']
        pstudent_dist = self.prob_tables['pstudent_dist']

        for _, hh in households.iterrows():
            hh_id = hh['household_id']
            hhsize = int(hh['hhsize'])
            home_zone_id = hh['home_zone_id']

            for pnum in range(1, hhsize + 1):
                age_cat = self._sample_from_distribution(age_dist, 'age_cat')
                age_row = age_dist[age_dist['age_cat'] == age_cat].iloc[0]
                age = np.random.randint(int(age_row['age_min']), int(age_row['age_max']) + 1)
                sex = self._sample_from_distribution(sex_dist, 'sex')
                pemploy = self._sample_conditional(pemploy_dist, 'age_cat', age_cat, 'pemploy')
                pstudent = self._sample_conditional(pstudent_dist, 'age_cat', age_cat, 'pstudent')
                ptype = self._determine_ptype(age_cat, pemploy, pstudent)

                persons.append({
                    'person_id': person_id, 'household_id': hh_id, 'home_zone_id': int(home_zone_id),
                    'age': age, 'PNUM': pnum, 'sex': sex, 'pemploy': pemploy, 'pstudent': pstudent,
                    'ptype': ptype, 'has_license': 1 if age >= 16 else 0,
                    'workplace_zone_id': -1, 'school_zone_id': -1,
                })
                person_id += 1
        return pd.DataFrame(persons)

    def _sample_from_distribution(self, dist_df: pd.DataFrame, value_col: str) -> int:
        if 'cumulative_prob' in dist_df.columns:
            rand_val = np.random.random()
            selected = dist_df[dist_df['cumulative_prob'] >= rand_val]
            return int(selected.iloc[0][value_col]) if len(selected) > 0 else int(dist_df.iloc[-1][value_col])
        elif 'probability' in dist_df.columns:
            probs = dist_df['probability'].values.astype(float)
            probs = probs / probs.sum()
            idx = np.random.choice(len(dist_df), p=probs)
            return int(dist_df.iloc[idx][value_col])
        return int(dist_df.sample(1).iloc[0][value_col])

    def _sample_conditional(self, dist_df: pd.DataFrame, cond_col: str, cond_val: int, value_col: str) -> int:
        cond_dist = dist_df[dist_df[cond_col] == cond_val]
        if len(cond_dist) == 0:
            return 4
        probs = cond_dist['probability'].values.astype(float)
        probs = probs / probs.sum()
        idx = np.random.choice(len(cond_dist), p=probs)
        return int(cond_dist.iloc[idx][value_col])

    def _sample_auto_ownership(self, auto_dist: pd.DataFrame, income_cat: int, hh_size_group: int) -> int:
        cond_dist = auto_dist[(auto_dist['income_cat'] == income_cat) & (auto_dist['hh_size_group'] == hh_size_group)]
        if len(cond_dist) == 0:
            return 1
        probs = cond_dist['probability'].values.astype(float)
        probs = probs / probs.sum()
        values = cond_dist['auto_ownership'].values if 'auto_ownership' in cond_dist.columns else cond_dist[
            'num_auto'].values
        return int(np.random.choice(values, p=probs))

    def _determine_ptype(self, age_cat: int, pemploy: int, pstudent: int) -> int:
        if age_cat == 1:
            return 8
        elif age_cat == 2:
            return 7
        elif age_cat == 3:
            return 6 if pemploy not in [1, 2] else (1 if pemploy == 1 else 2)
        elif pstudent == 3:
            return 3
        elif pemploy == 1:
            return 1
        elif pemploy == 2:
            return 2
        elif age_cat == 6:
            return 5
        return 4


# ----------------------------------------------------------------------------
# PopulationSim 配置和输入准备
# ----------------------------------------------------------------------------

def generate_populationsim_configs(config_dir: Path, data_dir: Path, output_dir: Path) -> None:
    """生成PopulationSim配置文件"""
    logger.info("正在生成PopulationSim配置文件...")
    ensure_dir(config_dir)
    ensure_dir(output_dir)

    # 1. settings.yaml
    settings = {
        'geographies': ['REGION', 'TAZ'],
        'seed_geography': 'TAZ',
        # household总量控制字段，需与controls.csv中的target一致（使用最低层级的target）
        'total_hh_control': 'num_hh_taz',

        'input_table_list': [
            {'tablename': 'households', 'filename': 'seed_households.csv', 'index_col': 'household_id'},
            {'tablename': 'persons', 'filename': 'seed_persons.csv', 'index_col': 'person_id'},
            {'tablename': 'geo_cross_walk', 'filename': 'geo_cross_walk.csv'},
            {'tablename': 'TAZ_control_data', 'filename': 'control_totals_taz.csv'},
            {'tablename': 'REGION_control_data', 'filename': 'control_totals_region.csv'},
        ],

        'household_weight_col': 'sample_weight',
        'household_id_col': 'household_id',

        'output_tables': {
            'action': 'include',
            'tables': ['summary', 'expanded_household_ids']
        },

        'output_synthetic_population': {
            'households': {
                'filename': 'synthetic_households.csv',
                'columns': [
                    'household_id', 'home_zone_id', 'hhsize', 'HHT', 'income',
                    'auto_ownership', 'num_workers', 'sample_weight', 'REGION', 'TAZ'
                ],
            },
            'persons': {
                'filename': 'synthetic_persons.csv',
                'columns': [
                    'household_id', 'home_zone_id', 'age', 'PNUM',
                    'sex', 'pemploy', 'pstudent', 'ptype', 'has_license',
                    'workplace_zone_id', 'school_zone_id'
                ],
            },
        },

        'models': [
            'input_pre_processor',
            'setup_data_structures',
            'initial_seed_balancing',
            'meta_control_factoring',
            'final_seed_balancing',
            'integerize_final_seed_weights',
            'expand_households',
            'write_tables',
            'write_synthetic_population',
        ],
    }
    write_yaml(settings, config_dir / 'settings.yaml')

    # 2. controls.csv - 完整格式
    controls_data = [
        {
            'target': 'num_hh_region',
            'control_field': 'num_hh_region',  # 唯一列名，避免与下级重复
            'geography': 'REGION',
            'seed_table': 'households',
            'expression': '1*1',  # household 数控制：加入符号避免被读成数值
            'importance': 10000,
            'control_type': 'simple',
        },
        {
            'target': 'num_hh_taz',
            'control_field': 'num_hh_taz',  # 唯一列名，避免与上级重复
            'geography': 'TAZ',
            'seed_table': 'households',
            'expression': '1*1',  # household 数控制：加入符号避免被读成数值
            'importance': 1000,
            'control_type': 'simple',
        },
    ]
    controls_df = pd.DataFrame(controls_data)
    write_csv(controls_df, config_dir / 'controls.csv')

    # 3. logging.yaml
    logging_config = {
        'logging': {
            'version': 1,
            'formatters': {'simple': {'format': '%(levelname)s - %(message)s'}},
            'handlers': {'console': {'class': 'logging.StreamHandler', 'level': 'INFO', 'formatter': 'simple'}},
            'root': {'level': 'INFO', 'handlers': ['console']}
        }
    }
    write_yaml(logging_config, config_dir / 'logging.yaml')

    logger.info(f"PopulationSim配置已生成: {config_dir}")


def prepare_populationsim_inputs(households: pd.DataFrame, persons: pd.DataFrame, land_use: pd.DataFrame,
                                 data_dir: Path) -> None:
    """准备PopulationSim输入文件"""
    logger.info("正在准备PopulationSim输入文件...")
    ensure_dir(data_dir)

    seed_hh = households.copy()
    if 'sample_weight' not in seed_hh.columns:
        seed_hh['sample_weight'] = 1
    if 'REGION' not in seed_hh.columns:
        seed_hh['REGION'] = 1
    if 'TAZ' not in seed_hh.columns:
        seed_hh['TAZ'] = seed_hh.get('home_zone_id', 1)

    write_csv(seed_hh, data_dir / 'seed_households.csv')
    write_csv(persons, data_dir / 'seed_persons.csv')

    zones = land_use['zone_id'].unique() if 'zone_id' in land_use.columns else land_use.index.unique()

    # 地理交叉表（简化：只有 REGION 和 TAZ）
    geo_cross_walk = pd.DataFrame({
        'TAZ': zones,
        'REGION': 1
    })
    write_csv(geo_cross_walk, data_dir / 'geo_cross_walk.csv')

    # TAZ 控制总量
    hh_col = 'hh' if 'hh' in land_use.columns else 'TOTHH' if 'TOTHH' in land_use.columns else None
    num_hh = land_use[hh_col].values if hh_col else np.full(len(zones), 100)
    taz_controls = pd.DataFrame({
        'TAZ': zones,
        'num_hh': num_hh
    })
    taz_controls['num_hh'] = taz_controls['num_hh'].fillna(0).astype(int)
    # 为控制文件提供唯一列名，避免多层地理的列名重复
    taz_controls['num_hh_taz'] = taz_controls['num_hh']
    write_csv(taz_controls, data_dir / 'control_totals_taz.csv')

    # REGION 控制总量
    region_controls = pd.DataFrame({
        'REGION': [1],
        'num_hh': [int(taz_controls['num_hh'].sum())]
    })
    region_controls['num_hh_region'] = region_controls['num_hh']
    write_csv(region_controls, data_dir / 'control_totals_region.csv')

    logger.info(f"PopulationSim输入准备完成:")
    logger.info(f"  种子: {len(seed_hh)}户, {len(persons)}人")
    logger.info(f"  TAZ: {len(zones)}个")
    logger.info(f"  目标家庭总数: {taz_controls['num_hh'].sum()}")


# ----------------------------------------------------------------------------
# PopulationSim 运行器
# ----------------------------------------------------------------------------

class PopulationSimRunner:
    """PopulationSim运行器"""

    def __init__(self, config_dir: Union[str, Path], data_dir: Union[str, Path], output_dir: Union[str, Path]):
        if not POPULATIONSIM_AVAILABLE:
            raise RuntimeError("PopulationSim 库不可用！请安装: pip install populationsim")

        # 使用绝对路径，避免在切换工作目录后找不到配置/数据/输出目录
        self.config_dir = Path(config_dir).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir).resolve()

        ensure_dir(self.config_dir)
        ensure_dir(self.data_dir)
        ensure_dir(self.output_dir)

    def run(
            self,
            mode: str = 'with_seed',
            random_seed: int = 1,
            num_virtual_households: int = 1000,
            prob_tables: Optional[Dict[str, pd.DataFrame]] = None,
            land_use: Optional[pd.DataFrame] = None,
            progress_callback: Optional[ProgressCallback] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """运行PopulationSim"""
        logger.info("=" * 60)
        logger.info("★★★ PopulationSim 运行开始 ★★★")
        logger.info("=" * 60)
        logger.info(f"运行模式: {mode}")

        np.random.seed(random_seed)

        if mode == 'with_seed':
            households, persons = self._load_seed_data()
        else:
            generator = VirtualSeedGenerator(prob_tables=prob_tables, random_seed=random_seed)
            households, persons = generator.generate_virtual_seed(num_virtual_households, land_use)

        if land_use is None:
            raise ValueError("land_use 数据是必需的")

        if progress_callback:
            progress_callback.update(message="运行 PopulationSim 库...")

        return self._execute_populationsim(households, persons, land_use, progress_callback)

    def _load_seed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载种子数据"""
        seed_dir = self.data_dir / 'seed'
        hh_file = seed_dir / 'households_seed.csv'
        per_file = seed_dir / 'persons_seed.csv'

        if not hh_file.exists():
            raise FileNotFoundError(f"种子家庭文件不存在: {hh_file}")
        if not per_file.exists():
            raise FileNotFoundError(f"种子人口文件不存在: {per_file}")

        households = read_csv(hh_file)
        persons = read_csv(per_file)
        logger.info(f"已加载种子数据: {len(households)}户, {len(persons)}人")
        return households, persons

    def _execute_populationsim(
            self,
            households: pd.DataFrame,
            persons: pd.DataFrame,
            land_use: pd.DataFrame,
            progress_callback: Optional[ProgressCallback]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """执行PopulationSim"""

        ensure_dir(self.config_dir)
        ensure_dir(self.data_dir)
        ensure_dir(self.output_dir)

        prepare_populationsim_inputs(households, persons, land_use, self.data_dir)
        generate_populationsim_configs(self.config_dir, self.data_dir, self.output_dir)

        if progress_callback:
            progress_callback.update(message="执行 PopulationSim...")

        # 清理之前的状态
        cleanup_populationsim_state()

        # 使用命令行方式运行 PopulationSim
        import subprocess
        import sys

        cmd = [
            sys.executable,
            '-m', 'populationsim',
            '-c', str(self.config_dir),
            '-d', str(self.data_dir),
            '-o', str(self.output_dir),
        ]

        logger.info(f"运行命令: {' '.join(cmd)}")
        logger.info(f"工作目录: {self.output_dir.parent}")

        # 检查配置文件是否存在
        logger.info("检查配置文件:")
        for config_file in ['settings.yaml', 'controls.yaml', 'logging.yaml']:
            config_path = self.config_dir / config_file
            logger.info(f"  {config_file}: {'存在' if config_path.exists() else '缺失'}")

        # 检查输入文件是否存在
        logger.info("检查输入文件:")
        for data_file in ['seed_households.csv', 'seed_persons.csv', 'geo_cross_walk.csv',
                          'control_totals_taz.csv', 'control_totals_region.csv']:
            data_path = self.data_dir / data_file
            if data_path.exists():
                df = read_csv(data_path)
                logger.info(f"  {data_file}: {len(df)} 行")
            else:
                logger.error(f"  {data_file}: 缺失！")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30分钟超时
                cwd=str(self.output_dir.parent)
            )

            # 始终输出完整日志
            logger.info("=" * 60)
            logger.info("PopulationSim 标准输出:")
            logger.info("=" * 60)
            if result.stdout:
                for line in result.stdout.split('\n'):
                    logger.info(line)
            else:
                logger.info("(无输出)")

            logger.info("=" * 60)
            logger.info("PopulationSim 错误输出:")
            logger.info("=" * 60)
            if result.stderr:
                for line in result.stderr.split('\n'):
                    logger.error(line)
            else:
                logger.info("(无错误输出)")
            logger.info("=" * 60)

            if result.returncode != 0:
                # 保存日志到文件
                error_log_file = self.output_dir / 'populationsim_error.log'
                with open(error_log_file, 'w', encoding='utf-8') as f:
                    f.write("=== STDOUT ===\n")
                    f.write(result.stdout or "(无输出)")
                    f.write("\n\n=== STDERR ===\n")
                    f.write(result.stderr or "(无错误输出)")

                error_msg = (
                    f"PopulationSim 运行失败，返回码: {result.returncode}\n"
                    f"错误日志已保存到: {error_log_file}\n"
                    f"请检查上方的详细输出"
                )
                raise RuntimeError(error_msg)

            logger.info("★★★ PopulationSim 运行成功 ★★★")

        except subprocess.TimeoutExpired:
            raise RuntimeError("PopulationSim 运行超时（30分钟）")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"PopulationSim 运行异常: {e}")
        finally:
            cleanup_populationsim_state()

        if progress_callback:
            progress_callback.update(message="读取PopulationSim输出...")

        return self._read_output(households, persons)

    def _read_output(self, seed_hh: pd.DataFrame, seed_per: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """读取PopulationSim输出"""
        synthetic_hh = None
        synthetic_per = None

        for hh_file in [self.output_dir / 'synthetic_households.csv', self.output_dir / 'final_households.csv']:
            if hh_file.exists():
                synthetic_hh = read_csv(hh_file)
                logger.info(f"读取合成家庭: {hh_file} ({len(synthetic_hh)}户)")
                break

        for per_file in [self.output_dir / 'synthetic_persons.csv', self.output_dir / 'final_persons.csv']:
            if per_file.exists():
                synthetic_per = read_csv(per_file)
                logger.info(f"读取合成人口: {per_file} ({len(synthetic_per)}人)")
                break

        if synthetic_hh is not None and synthetic_per is not None:
            return synthetic_hh, synthetic_per

        expanded_file = self.output_dir / 'expanded_household_ids.csv'
        if expanded_file.exists():
            logger.info("从 expanded_household_ids.csv 重建合成人口...")
            expanded = read_csv(expanded_file)
            synthetic_hh = expanded.merge(seed_hh, on='household_id', how='left', suffixes=('', '_seed'))
            synthetic_hh['original_household_id'] = synthetic_hh['household_id']
            synthetic_hh['household_id'] = range(1, len(synthetic_hh) + 1)

            synthetic_per = seed_per.merge(
                synthetic_hh[['original_household_id', 'household_id']],
                left_on='household_id', right_on='original_household_id', how='inner'
            )
            synthetic_per['household_id'] = synthetic_per['household_id_y']
            synthetic_per = synthetic_per.drop(columns=['household_id_x', 'household_id_y', 'original_household_id'],
                                               errors='ignore')
            synthetic_per['person_id'] = range(1, len(synthetic_per) + 1)

            logger.info(f"重建完成: {len(synthetic_hh)}户, {len(synthetic_per)}人")
            return synthetic_hh, synthetic_per

        raise RuntimeError("未找到PopulationSim输出文件")


# ----------------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------------

def run_populationsim(
        config_dir: Union[str, Path],
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        mode: str = 'with_seed',
        random_seed: int = 1,
        num_virtual_households: int = 1000,
        prob_tables: Optional[Dict[str, pd.DataFrame]] = None,
        land_use: Optional[pd.DataFrame] = None,
        progress_callback: Optional[ProgressCallback] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """运行PopulationSim"""
    cleanup_populationsim_state()
    runner = PopulationSimRunner(config_dir, data_dir, output_dir)
    try:
        return runner.run(mode, random_seed, num_virtual_households, prob_tables, land_use, progress_callback)
    finally:
        cleanup_populationsim_state()
