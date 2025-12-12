# ============================================================================
# pipeline.py
# ============================================================================
# 模块职责：全流程编排和调度
# 包括：
# - 协调各模块按正确顺序执行
# - 管理中间数据的传递
# - 提供进度回调和错误处理
# - 生成流程运行报告
# ============================================================================

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

import pandas as pd
import numpy as np
import geopandas as gpd

from .io_utils import (
    logger, ensure_dir, write_csv, write_json,
    ProgressCallback, get_timestamp
)
from .shapefile_utils import (
    read_shapefile, extract_taz_info, compute_zone_centroids,
    save_study_area, get_zone_summary
)
from .osm_poi_utils import (
    process_osm_buildings, process_osm_pois, save_osm_data,
    get_building_statistics, get_poi_statistics
)
from .network_skim_utils import (
    build_skims_from_matsim_network, build_simple_skims
)
from .landuse_builder import (
    create_landuse_from_osm, ConversionCoefficients,
    validate_landuse_table
)
from .populationsim_runner import (
    run_populationsim, get_default_probability_tables,
    POPULATIONSIM_AVAILABLE
)
from .activitysim_runner import (
    run_activitysim, compute_od_matrix_from_trips,
    ACTIVITYSIM_AVAILABLE
)
# 在文件开头的导入部分后添加
from .populationsim_runner import POPULATIONSIM_AVAILABLE
from .activitysim_runner import ACTIVITYSIM_AVAILABLE

# 检查库可用性
if not POPULATIONSIM_AVAILABLE:
    raise ImportError(
        "PopulationSim 库未正确安装！\n"
        "请运行: pip install populationsim"
    )

if not ACTIVITYSIM_AVAILABLE:
    raise ImportError(
        "ActivitySim 库未正确安装！\n"
        "请运行: pip install activitysim"
    )

logger.info("✓ PopulationSim 和 ActivitySim 库均已加载")

# ----------------------------------------------------------------------------
# 流程配置
# ----------------------------------------------------------------------------

class PipelineConfig:
    """
    流程配置类

    封装全流程的配置参数
    """

    def __init__(self):
        """初始化默认配置"""

        # 输入文件路径
        self.shapefile_path: Optional[Path] = None
        self.network_file_path: Optional[Path] = None

        # PopulationSim模式
        self.populationsim_mode: str = 'with_seed'  # 'with_seed' or 'from_prob_tables'
        self.num_virtual_households: int = 1000

        # 概率分布表（无种子模式使用）
        self.prob_tables: Optional[Dict[str, pd.DataFrame]] = None

        # OSM下载设置
        self.osm_buffer_km: float = 1.0
        self.osm_timeout: int = 180

        # 转换系数
        self.conversion_coefficients: Optional[ConversionCoefficients] = None

        # Skim构建设置
        self.use_matsim_network: bool = True  # False则使用简化欧氏距离
        self.average_speed_kmh: float = 40.0  # 简化skim的平均速度

        # ActivitySim设置
        self.activitysim_sample_rate: float = 1.0  # 1.0=全样本
        self.use_full_activitysim: bool = True  # 是否尝试使用完整ActivitySim

        # 随机种子
        self.random_seed: int = 1

        # 目录路径
        self.project_root: Optional[Path] = None
        self.config_dir: Optional[Path] = None
        self.data_dir: Optional[Path] = None
        self.output_dir: Optional[Path] = None

    def set_paths(
            self,
            project_root: Union[str, Path],
            create_timestamped_output: bool = False,
            use_temp_for_data: bool = False
    ) -> None:
        """
        设置项目路径

        参数:
            project_root: 项目根目录
            create_timestamped_output: 是否创建带时间戳的输出目录
            use_temp_for_data: 是否使用临时目录存放数据
        """
        self.project_root = Path(project_root).resolve()
        self.config_dir = self.project_root / 'config'

        if use_temp_for_data:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp(prefix="tdm_")).resolve()
            self.data_dir = temp_dir / 'data'
            base_output = temp_dir / 'output'
        else:
            self.data_dir = (self.project_root / 'data').resolve()
            base_output = self.data_dir / 'output'

        if create_timestamped_output:
            timestamp = get_timestamp()
            self.output_dir = (base_output / f'run_{timestamp}').resolve()
        else:
            self.output_dir = base_output.resolve()

        ensure_dir(self.data_dir)
        ensure_dir(self.output_dir)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'shapefile_path': str(self.shapefile_path) if self.shapefile_path else None,
            'network_file_path': str(self.network_file_path) if self.network_file_path else None,
            'populationsim_mode': self.populationsim_mode,
            'num_virtual_households': self.num_virtual_households,
            'osm_buffer_km': self.osm_buffer_km,
            'osm_timeout': self.osm_timeout,
            'use_matsim_network': self.use_matsim_network,
            'average_speed_kmh': self.average_speed_kmh,
            'activitysim_sample_rate': self.activitysim_sample_rate,
            'use_full_activitysim': self.use_full_activitysim,
            'random_seed': self.random_seed,
            'populationsim_available': POPULATIONSIM_AVAILABLE,
            'activitysim_available': ACTIVITYSIM_AVAILABLE,
        }

    def validate(self) -> List[str]:
        """
        验证配置完整性

        返回:
            错误信息列表（空列表表示验证通过）
        """
        errors = []

        if self.project_root is None:
            errors.append("必须设置project_root路径")

        if self.shapefile_path is None:
            errors.append("必须提供shapefile_path")
        elif not Path(self.shapefile_path).exists():
            errors.append(f"Shapefile不存在: {self.shapefile_path}")

        if self.populationsim_mode == 'with_seed':
            seed_dir = self.data_dir / 'input' / 'seed'
            if seed_dir.exists():
                hh_seed = seed_dir / 'households_seed.csv'
                per_seed = seed_dir / 'persons_seed.csv'
                if not hh_seed.exists():
                    errors.append(f"种子家庭文件不存在: {hh_seed}")
                if not per_seed.exists():
                    errors.append(f"种子人口文件不存在: {per_seed}")

        return errors


# ----------------------------------------------------------------------------
# 流程状态管理
# ----------------------------------------------------------------------------

class PipelineState:
    """
    流程状态管理类

    跟踪流程执行状态和中间结果
    """

    def __init__(self):
        """初始化状态"""
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # 各步骤完成状态
        self.steps_completed: Dict[str, bool] = {
            'load_shapefile': False,
            'download_osm': False,
            'build_landuse': False,
            'build_skims': False,
            'run_populationsim': False,
            'run_activitysim': False,
        }

        # 中间结果
        self.study_area: Optional[gpd.GeoDataFrame] = None
        self.zone_centroids: Optional[gpd.GeoDataFrame] = None
        self.buildings: Optional[gpd.GeoDataFrame] = None
        self.pois: Optional[gpd.GeoDataFrame] = None
        self.land_use: Optional[pd.DataFrame] = None
        self.skims_path: Optional[Path] = None
        self.synthetic_households: Optional[pd.DataFrame] = None
        self.synthetic_persons: Optional[pd.DataFrame] = None
        self.tours: Optional[pd.DataFrame] = None
        self.trips: Optional[pd.DataFrame] = None

        # 统计信息
        self.statistics: Dict = {}

        # 错误和警告
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def mark_step_complete(self, step_name: str) -> None:
        """标记步骤完成"""
        self.steps_completed[step_name] = True
        logger.info(f"步骤 '{step_name}' 已完成")

    def is_step_complete(self, step_name: str) -> bool:
        """检查步骤是否完成"""
        return self.steps_completed.get(step_name, False)

    def get_progress(self) -> float:
        """获取整体进度百分比"""
        total_steps = len(self.steps_completed)
        completed = sum(self.steps_completed.values())
        return completed / total_steps if total_steps > 0 else 0.0

    def get_elapsed_time(self) -> float:
        """获取已用时间（秒）"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else datetime.now()
        return (end - self.start_time).total_seconds()

    def add_warning(self, message: str) -> None:
        """添加警告信息"""
        self.warnings.append(message)
        logger.warning(message)

    def add_error(self, message: str) -> None:
        """添加错误信息"""
        self.errors.append(message)
        logger.error(message)


# ----------------------------------------------------------------------------
# 主流程类
# ----------------------------------------------------------------------------

class TravelDemandModelPipeline:
    """
    出行需求建模流程主类

    编排和执行完整的建模流程
    """

    def __init__(
            self,
            config: PipelineConfig,
            progress_callback: Optional[ProgressCallback] = None
    ):
        """
        初始化流程

        参数:
            config: 流程配置
            progress_callback: 进度回调
        """
        self.config = config
        self.state = PipelineState()
        self.progress_callback = progress_callback or ProgressCallback(total_steps=100)

        # 记录库可用性
        self._log_library_availability()

    def _log_library_availability(self) -> None:
        """记录库的可用性状态"""
        if POPULATIONSIM_AVAILABLE:
            logger.info("✓ PopulationSim库可用")
        else:
            logger.warning("✗ PopulationSim库不可用，将使用简化扩样")

        if ACTIVITYSIM_AVAILABLE:
            logger.info("✓ ActivitySim库可用")
        else:
            logger.warning("✗ ActivitySim库不可用，将使用简化出行生成")

    def run(self) -> PipelineState:
        """
        运行完整流程

        返回:
            流程状态对象，包含所有中间和最终结果
        """
        logger.info("=" * 80)
        logger.info("开始执行出行需求建模流程")
        logger.info("=" * 80)

        self.state.start_time = datetime.now()

        # 验证配置
        config_errors = self.config.validate()
        if config_errors:
            for error in config_errors:
                self.state.add_error(error)
            raise ValueError(f"配置验证失败: {config_errors}")

        try:
            # 步骤1: 加载研究区域
            self._step_load_shapefile()

            # 步骤2: 下载OSM数据
            self._step_download_osm()

            # 步骤3: 构建土地利用
            self._step_build_landuse()

            # 步骤4: 构建Skim矩阵
            self._step_build_skims()

            # 步骤5: 运行PopulationSim
            self._step_run_populationsim()

            # 步骤6: 运行ActivitySim
            self._step_run_activitysim()

            # 步骤7: 生成报告
            self._step_generate_report()

            self.state.end_time = datetime.now()

            logger.info("=" * 80)
            logger.info(f"流程执行完成，耗时: {self.state.get_elapsed_time():.1f}秒")
            logger.info("=" * 80)

        except Exception as e:
            self.state.add_error(str(e))
            logger.error(f"流程执行失败: {e}", exc_info=True)
            self.state.end_time = datetime.now()
            raise

        return self.state

    def _step_load_shapefile(self) -> None:
        """步骤1: 加载研究区域Shapefile"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤1: 加载研究区域Shapefile")
        logger.info("=" * 80)

        self.progress_callback.update(10, "正在加载研究区域Shapefile...")

        # 读取shapefile
        study_area = read_shapefile(
            self.config.shapefile_path,
            target_crs='EPSG:4326'
        )

        # 提取TAZ信息
        study_area = extract_taz_info(study_area)

        # 计算质心
        zone_centroids = compute_zone_centroids(study_area)

        # 保存
        preprocessed_dir = self.config.output_dir / 'preprocessed'
        ensure_dir(preprocessed_dir)

        save_study_area(
            study_area,
            preprocessed_dir / 'study_area.geojson'
        )

        # 保存到状态
        self.state.study_area = study_area
        self.state.zone_centroids = zone_centroids

        # 统计
        zone_summary = get_zone_summary(study_area)
        self.state.statistics['zone_summary'] = zone_summary.to_dict('records')
        self.state.statistics['num_zones'] = len(study_area)

        logger.info(f"研究区域包含 {len(study_area)} 个TAZ")
        logger.info(f"\n{zone_summary.to_string(index=False)}")

        self.state.mark_step_complete('load_shapefile')

    def _step_download_osm(self) -> None:
        """步骤2: 下载OSM数据"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤2: 下载OSM建筑和POI数据")
        logger.info("=" * 80)

        self.progress_callback.update(20, "正在下载OSM数据...")

        try:
            # 下载建筑
            buildings = process_osm_buildings(
                self.state.study_area,
                buffer_km=self.config.osm_buffer_km,
                timeout=self.config.osm_timeout
            )

            # 下载POI
            pois = process_osm_pois(
                self.state.study_area,
                buffer_km=self.config.osm_buffer_km,
                timeout=self.config.osm_timeout
            )

            # 保存
            preprocessed_dir = self.config.output_dir / 'preprocessed'
            save_osm_data(buildings, pois, preprocessed_dir)

            # 保存到状态
            self.state.buildings = buildings
            self.state.pois = pois

            # 统计
            if len(buildings) > 0:
                building_stats = get_building_statistics(buildings)
                self.state.statistics['building_stats'] = building_stats.to_dict('records')
                self.state.statistics['num_buildings'] = len(buildings)
                logger.info(f"\n建筑统计:\n{building_stats.to_string(index=False)}")

            if len(pois) > 0:
                poi_stats = get_poi_statistics(pois)
                self.state.statistics['poi_stats'] = poi_stats.to_dict('records')
                self.state.statistics['num_pois'] = len(pois)
                logger.info(f"\nPOI统计:\n{poi_stats.to_string(index=False)}")

        except Exception as e:
            self.state.add_warning(f"OSM数据下载失败: {e}")
            logger.warning("将使用空的建筑和POI数据继续")
            self.state.buildings = gpd.GeoDataFrame()
            self.state.pois = gpd.GeoDataFrame()

        self.state.mark_step_complete('download_osm')

    def _step_build_landuse(self) -> None:
        """步骤3: 构建土地利用表"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤3: 构建土地利用表")
        logger.info("=" * 80)

        self.progress_callback.update(35, "正在构建土地利用表...")

        landuse_dir = self.config.output_dir / 'landuse'
        ensure_dir(landuse_dir)

        if self.state.buildings is not None and len(self.state.buildings) > 0:
            # 从OSM数据构建
            land_use = create_landuse_from_osm(
                self.state.study_area,
                self.state.buildings,
                landuse_dir / 'land_use.csv',
                coefficients=self.config.conversion_coefficients
            )
        else:
            # 生成默认土地利用
            logger.warning("无OSM建筑数据，生成默认土地利用表")
            land_use = self._generate_default_landuse()
            write_csv(land_use, landuse_dir / 'land_use.csv')

        # 保存到状态
        self.state.land_use = land_use

        # 验证
        issues = validate_landuse_table(land_use)
        if issues:
            self.state.statistics['landuse_issues'] = issues
            for issue in issues:
                self.state.add_warning(f"土地利用验证: {issue}")

        # 统计
        self.state.statistics['landuse_summary'] = {
            'total_zones': len(land_use),
            'total_population': int(land_use['pop'].sum()) if 'pop' in land_use.columns else 0,
            'total_households': int(land_use['hh'].sum()) if 'hh' in land_use.columns else 0,
            'total_employment': int(land_use['emp_total'].sum()) if 'emp_total' in land_use.columns else 0,
        }

        logger.info(f"land_use表已生成: {len(land_use)} 个zone")

        self.state.mark_step_complete('build_landuse')

    def _generate_default_landuse(self) -> pd.DataFrame:
        """生成默认的土地利用表"""
        zones = self.state.study_area['zone_id'].values
        n_zones = len(zones)

        land_use = pd.DataFrame({
            'zone_id': zones,
            'area': self.state.study_area['area_km2'].values if 'area_km2' in self.state.study_area.columns else 1.0,
            'hh': np.random.randint(50, 500, size=n_zones),
            'pop': np.random.randint(150, 1500, size=n_zones),
            'emp_total': np.random.randint(100, 1000, size=n_zones),
            'emp_office': np.random.randint(20, 200, size=n_zones),
            'emp_retail': np.random.randint(20, 200, size=n_zones),
            'emp_edu': np.random.randint(10, 100, size=n_zones),
            'emp_health': np.random.randint(10, 100, size=n_zones),
            'emp_industrial': np.random.randint(20, 200, size=n_zones),
            'emp_other': np.random.randint(20, 200, size=n_zones),
        })

        # 计算密度
        land_use['density'] = land_use['pop'] / land_use['area'].clip(lower=0.01)
        land_use['emp_density'] = land_use['emp_total'] / land_use['area'].clip(lower=0.01)

        # CBD标识
        emp_density_threshold = land_use['emp_density'].quantile(0.8)
        land_use['is_cbd'] = (land_use['emp_density'] >= emp_density_threshold).astype(int)

        return land_use

    def _step_build_skims(self) -> None:
        """步骤4: 构建Skim矩阵"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤4: 构建Skim矩阵")
        logger.info("=" * 80)

        self.progress_callback.update(50, "正在构建Skim矩阵...")

        skims_dir = self.config.output_dir / 'skims'
        ensure_dir(skims_dir)
        skims_file = skims_dir / 'skims.omx'

        try:
            if self.config.use_matsim_network and self.config.network_file_path:
                # 使用MATSim路网
                logger.info("使用MATSim路网构建Skim矩阵...")
                skims_path = build_skims_from_matsim_network(
                    self.config.network_file_path,
                    self.state.zone_centroids,
                    skims_file,
                    progress_callback=self.progress_callback
                )
            else:
                # 使用简化方法
                logger.info("使用简化方法（欧氏距离）构建Skim矩阵...")
                skims_path = build_simple_skims(
                    self.state.zone_centroids,
                    skims_file,
                    average_speed_kmh=self.config.average_speed_kmh
                )

            self.state.skims_path = skims_path

        except Exception as e:
            self.state.add_warning(f"Skim矩阵构建失败: {e}")
            logger.warning("使用简化方法重试...")

            skims_path = build_simple_skims(
                self.state.zone_centroids,
                skims_file,
                average_speed_kmh=self.config.average_speed_kmh
            )
            self.state.skims_path = skims_path

        self.state.mark_step_complete('build_skims')

    def _step_run_populationsim(self) -> None:
        """步骤5: 运行PopulationSim"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤5: 运行PopulationSim生成合成人口")
        logger.info("=" * 80)

        self.progress_callback.update(60, "正在运行PopulationSim...")

        # 记录模式
        mode = self.config.populationsim_mode
        logger.info(f"PopulationSim模式: {mode}")

        if mode == 'with_seed':
            logger.info("使用种子数据进行扩样")
            if POPULATIONSIM_AVAILABLE:
                logger.info("将调用PopulationSim库")
            else:
                logger.info("PopulationSim不可用，将使用简化扩样")
        else:
            logger.info(f"从概率表生成虚拟种子 ({self.config.num_virtual_households}户)")

        # 准备概率表
        prob_tables = self.config.prob_tables
        if prob_tables is None and mode == 'from_prob_tables':
            prob_tables = get_default_probability_tables()
            logger.info("使用默认概率分布表")

        # 运行PopulationSim
        try:
            households, persons = run_populationsim(
                config_dir=self.config.config_dir / 'populationsim',
                data_dir=self.config.data_dir / 'input',
                output_dir=self.config.output_dir / 'populationsim',
                mode=mode,
                random_seed=self.config.random_seed,
                num_virtual_households=self.config.num_virtual_households,
                prob_tables=prob_tables,
                land_use=self.state.land_use,
                progress_callback=self.progress_callback
            )

            # 保存到状态
            self.state.synthetic_households = households
            self.state.synthetic_persons = persons

            # 统计
            self.state.statistics['synthetic_population'] = {
                'num_households': len(households),
                'num_persons': len(persons),
                'avg_hh_size': len(persons) / len(households) if len(households) > 0 else 0,
            }

            # 详细统计
            if 'ptype' in persons.columns:
                ptype_dist = persons['ptype'].value_counts().to_dict()
                self.state.statistics['ptype_distribution'] = ptype_dist

            if 'pemploy' in persons.columns:
                pemploy_dist = persons['pemploy'].value_counts().to_dict()
                self.state.statistics['pemploy_distribution'] = pemploy_dist

            logger.info(f"合成人口生成完成: {len(households)}户, {len(persons)}人")

        except Exception as e:
            self.state.add_error(f"PopulationSim失败: {e}")
            raise

        self.state.mark_step_complete('run_populationsim')

    def _step_run_activitysim(self) -> None:
        """步骤6: 运行ActivitySim"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤6: 运行ActivitySim生成活动链")
        logger.info("=" * 80)

        self.progress_callback.update(75, "正在运行ActivitySim...")

        if ACTIVITYSIM_AVAILABLE and self.config.use_full_activitysim:
            logger.info("将尝试调用ActivitySim库")
        else:
            logger.info("将使用简化版出行生成")

        try:
            # 运行ActivitySim
            tours, trips = run_activitysim(
                config_dir=self.config.config_dir / 'activitysim',
                data_dir=self.config.data_dir / 'input',
                output_dir=self.config.output_dir / 'activitysim',
                synthetic_households=self.state.synthetic_households,
                synthetic_persons=self.state.synthetic_persons,
                land_use=self.state.land_use,
                skims_path=self.state.skims_path,
                random_seed=self.config.random_seed,
                sample_rate=self.config.activitysim_sample_rate,
                progress_callback=self.progress_callback
            )

            # 保存到状态
            self.state.tours = tours
            self.state.trips = trips

            # 统计
            if tours is not None and len(tours) > 0:
                self.state.statistics['tours_summary'] = {
                    'num_tours': len(tours),
                }
                if 'tour_type' in tours.columns:
                    self.state.statistics['tours_by_type'] = \
                        tours['tour_type'].value_counts().to_dict()
                elif 'tour_purpose' in tours.columns:
                    self.state.statistics['tours_by_purpose'] = \
                        tours['tour_purpose'].value_counts().to_dict()

            if trips is not None and len(trips) > 0:
                self.state.statistics['trips_summary'] = {
                    'num_trips': len(trips),
                }
                if 'trip_mode' in trips.columns:
                    self.state.statistics['trips_by_mode'] = \
                        trips['trip_mode'].value_counts().to_dict()

            logger.info(
                f"活动链生成完成: "
                f"{len(tours) if tours is not None else 0} tours, "
                f"{len(trips) if trips is not None else 0} trips"
            )

        except Exception as e:
            self.state.add_error(f"ActivitySim失败: {e}")
            raise

        self.state.mark_step_complete('run_activitysim')

    def _step_generate_report(self) -> None:
        """步骤7: 生成流程报告"""
        logger.info("\n" + "=" * 80)
        logger.info("步骤7: 生成流程报告")
        logger.info("=" * 80)

        self.progress_callback.update(95, "正在生成流程报告...")

        # 生成报告
        report = self._create_report()

        # 保存报告
        report_file = self.config.output_dir / 'pipeline_report.json'
        write_json(report, report_file)

        logger.info(f"流程报告已保存: {report_file}")

        self.progress_callback.update(100, "流程全部完成！")

    def _create_report(self) -> Dict:
        """
        创建流程运行报告
        """
        report = {
            'run_info': {
                'start_time': self.state.start_time.isoformat() if self.state.start_time else None,
                'end_time': self.state.end_time.isoformat() if self.state.end_time else None,
                'elapsed_seconds': self.state.get_elapsed_time(),
                'progress': self.state.get_progress(),
            },
            'library_status': {
                'populationsim_available': POPULATIONSIM_AVAILABLE,
                'activitysim_available': ACTIVITYSIM_AVAILABLE,
            },
            'config': self.config.to_dict(),
            'steps_completed': self.state.steps_completed,
            'statistics': self.state.statistics,
            'warnings': self.state.warnings,
            'errors': self.state.errors,
            'output_files': {
                'study_area': str(self.config.output_dir / 'preprocessed' / 'study_area.geojson'),
                'land_use': str(self.config.output_dir / 'landuse' / 'land_use.csv'),
                'skims': str(self.state.skims_path) if self.state.skims_path else None,
                'synthetic_households': str(
                    self.config.output_dir / 'populationsim' / 'synthetic_households.csv'),
                'synthetic_persons': str(self.config.output_dir / 'populationsim' / 'synthetic_persons.csv'),
                'tours': str(self.config.output_dir / 'activitysim' / 'tours.csv'),
                'trips': str(self.config.output_dir / 'activitysim' / 'trips.csv'),
            }
        }

        return report


# ----------------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------------

def run_full_pipeline(
        project_root: Union[str, Path],
        shapefile_path: Union[str, Path],
        network_file_path: Optional[Union[str, Path]] = None,
        populationsim_mode: str = 'with_seed',
        num_virtual_households: int = 1000,
        prob_tables: Optional[Dict[str, pd.DataFrame]] = None,
        random_seed: int = 1,
        progress_callback: Optional[ProgressCallback] = None
) -> PipelineState:
    """
    运行完整流程的便捷函数

    参数:
        project_root: 项目根目录
        shapefile_path: 研究区域shapefile路径
        network_file_path: MATSim路网文件路径（可选）
        populationsim_mode: PopulationSim模式
        num_virtual_households: 虚拟家庭数（无种子模式）
        prob_tables: 概率分布表
        random_seed: 随机种子
        progress_callback: 进度回调

    返回:
        流程状态对象
    """
    # 创建配置
    config = PipelineConfig()
    config.set_paths(project_root, create_timestamped_output=True)
    config.shapefile_path = Path(shapefile_path)
    config.network_file_path = Path(network_file_path) if network_file_path else None
    config.populationsim_mode = populationsim_mode
    config.num_virtual_households = num_virtual_households
    config.prob_tables = prob_tables
    config.random_seed = random_seed

    # 创建并运行流程
    pipeline = TravelDemandModelPipeline(config, progress_callback)
    state = pipeline.run()

    return state


def check_dependencies() -> Dict[str, bool]:
    """
    检查依赖库的可用性

    返回:
        依赖库可用性字典
    """
    return {
        'populationsim': POPULATIONSIM_AVAILABLE,
        'activitysim': ACTIVITYSIM_AVAILABLE,
        'pandas': True,
        'geopandas': True,
        'numpy': True,
    }
