# ============================================================================
# app.py
# ============================================================================
# Streamlit主应用：出行需求建模系统Web界面
# 功能：
# - 文件上传（Shapefile、MATSim路网、种子数据）
# - 参数配置（转换系数、概率分布表等）
# - 一键运行完整流程
# - 进度实时显示
# - 结果可视化（地图、图表、统计）
# - 结果文件下载
# ============================================================================

import streamlit as st
import sys
from pathlib import Path
import tempfile
import shutil
from typing import Optional, Dict, List

import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加src目录到Python路径（使用绝对路径避免后续切换工作目录导致路径出错）
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / 'src'))

from src.io_utils import (
    logger, setup_logger, ensure_dir, write_csv,
    ProgressCallback
)
from src.shapefile_utils import read_shapefile, extract_taz_info
from src.landuse_builder import ConversionCoefficients, DEFAULT_CONVERSION_COEFFICIENTS
from src.pipeline import (
    PipelineConfig, TravelDemandModelPipeline, PipelineState,
    check_dependencies
)
from src.populationsim_runner import (
    get_default_probability_tables, POPULATIONSIM_AVAILABLE
)
from src.activitysim_runner import ACTIVITYSIM_AVAILABLE


# ============================================================================
# 页面配置
# ============================================================================

st.set_page_config(
    page_title="出行需求建模系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 全局样式
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
    }
    .lib-status {
        font-size: 0.9rem;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
    }
    .lib-available {
        background-color: #c8e6c9;
        color: #2e7d32;
    }
    .lib-unavailable {
        background-color: #ffcdd2;
        color: #c62828;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State初始化
# ============================================================================

def init_session_state():
    """初始化session state"""
    if 'pipeline_state' not in st.session_state:
        st.session_state.pipeline_state = None

    if 'pipeline_running' not in st.session_state:
        st.session_state.pipeline_running = False

    if 'progress_messages' not in st.session_state:
        st.session_state.progress_messages = []

    if 'current_progress' not in st.session_state:
        st.session_state.current_progress = 0.0

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}

    if 'conversion_coefficients' not in st.session_state:
        st.session_state.conversion_coefficients = ConversionCoefficients()

    # 初始化ActivitySim兼容的概率表
    if 'prob_tables' not in st.session_state:
        st.session_state.prob_tables = get_default_probability_tables()


init_session_state()


# ============================================================================
# 辅助函数
# ============================================================================

def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    """保存上传的文件"""
    ensure_dir(target_dir)
    file_path = target_dir / uploaded_file.name

    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    return file_path


class StreamlitProgressCallback(ProgressCallback):
    """Streamlit进度回调"""

    def __init__(self, total_steps: int = 100):
        super().__init__(total_steps)
        self.progress_bar = st.progress(0.0)
        self.status_text = st.empty()
        self.log_container = st.expander("详细日志", expanded=False)

    def update(
            self,
            step: Optional[int] = None,
            message: str = "",
            progress: Optional[float] = None
    ) -> None:
        """更新进度"""
        super().update(step, message, progress)

        current_progress = self.get_progress()
        self.progress_bar.progress(min(current_progress, 1.0))
        self.status_text.text(f"📊 {message}")

        with self.log_container:
            st.text(f"[{self.current_step}/{self.total_steps}] {message}")

        st.session_state.current_progress = current_progress
        st.session_state.progress_messages.append(message)


def display_library_status():
    """显示库可用性状态"""
    col1, col2 = st.columns(2)

    with col1:
        if POPULATIONSIM_AVAILABLE:
            st.markdown(
                '<div class="lib-status lib-available">✓ PopulationSim 可用</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="lib-status lib-unavailable">✗ PopulationSim 不可用（使用简化扩样）</div>',
                unsafe_allow_html=True
            )

    with col2:
        if ACTIVITYSIM_AVAILABLE:
            st.markdown(
                '<div class="lib-status lib-available">✓ ActivitySim 可用</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="lib-status lib-unavailable">✗ ActivitySim 不可用（使用简化出行生成）</div>',
                unsafe_allow_html=True
            )


# ============================================================================
# 页面标题
# ============================================================================

st.markdown('<div class="main-header">🚗 出行需求建模系统</div>', unsafe_allow_html=True)

# 显示库状态
display_library_status()

st.markdown("""
<div class="info-box">
    <strong>系统功能：</strong>从研究区域定义到个体出行链生成的端到端建模流程
    <ul>
        <li>📍 研究区域处理：支持Shapefile/ZIP上传</li>
        <li>🏢 OSM数据下载：自动获取建筑、POI数据</li>
        <li>🗺️ 土地利用构建：建筑面积→就业/人口转换</li>
        <li>🛣️ 路网与Skim：支持MATSim路网或简化距离矩阵</li>
        <li>👨‍👩‍👧‍👦 合成人口生成：PopulationSim（有种子/无种子模式）</li>
        <li>🚌 活动链生成：ActivitySim生成出行链和出行段</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# 侧边栏：输入配置
# ============================================================================

st.sidebar.markdown("## 📂 输入数据")

# --- 1. 研究区域Shapefile ---
st.sidebar.markdown("### 1️⃣ 研究区域")
shapefile_upload = st.sidebar.file_uploader(
    "上传Shapefile或ZIP",
    type=['shp', 'zip'],
    help="上传.shp文件（需同目录有.dbf/.shx/.prj）或包含完整shapefile的.zip压缩包"
)

if shapefile_upload:
    st.session_state.uploaded_files['shapefile'] = shapefile_upload
    st.sidebar.success(f"✅ 已上传: {shapefile_upload.name}")

# --- 2. MATSim路网（可选）---
st.sidebar.markdown("### 2️⃣ 路网文件（可选）")
use_matsim_network = st.sidebar.checkbox(
    "使用MATSim路网构建Skim",
    value=False,
    help="如不勾选，则使用简化的欧氏距离方法"
)

network_upload = None
if use_matsim_network:
    network_upload = st.sidebar.file_uploader(
        "上传MATSim network.xml",
        type=['xml'],
        help="MATSim格式的路网文件"
    )

    if network_upload:
        st.session_state.uploaded_files['network'] = network_upload
        st.sidebar.success(f"✅ 已上传: {network_upload.name}")
else:
    avg_speed = st.sidebar.slider(
        "平均出行速度 (km/h)",
        min_value=20,
        max_value=80,
        value=40,
        step=5,
        help="用于简化Skim矩阵的时间计算"
    )

# --- 3. PopulationSim模式 ---
st.sidebar.markdown("### 3️⃣ 合成人口模式")
popsim_mode = st.sidebar.radio(
    "选择运行模式",
    options=['with_seed', 'from_prob_tables'],
    format_func=lambda x: "有种子数据" if x == 'with_seed' else "无种子（概率表）",
    help="有种子模式需上传样本数据；无种子模式从概率分布生成虚拟样本"
)

if popsim_mode == 'with_seed':
    st.sidebar.markdown("#### 上传种子数据")

    # 显示种子数据格式要求
    with st.sidebar.expander("📋 种子数据格式要求"):
        st.markdown("""
        **households_seed.csv 必需列：**
        - `household_id`: 家庭唯一标识
        - `hhsize`: 家庭规模
        - `income`: 家庭收入
        - `HHT`: 家庭类型 (1-7)
        - `auto_ownership`: 车辆数量

        **persons_seed.csv 必需列：**
        - `person_id`: 人员唯一标识
        - `household_id`: 所属家庭ID
        - `age`: 年龄
        - `sex`: 性别 (1=男, 2=女)
        - `pemploy`: 就业状态 (1-4)
        - `pstudent`: 学生状态 (1-4)
        """)

    hh_seed_upload = st.sidebar.file_uploader(
        "households_seed.csv",
        type=['csv'],
        key='hh_seed'
    )
    per_seed_upload = st.sidebar.file_uploader(
        "persons_seed.csv",
        type=['csv'],
        key='per_seed'
    )

    if hh_seed_upload and per_seed_upload:
        st.session_state.uploaded_files['hh_seed'] = hh_seed_upload
        st.session_state.uploaded_files['per_seed'] = per_seed_upload
        st.sidebar.success("✅ 种子数据已上传")
else:
    num_virtual_hh = st.sidebar.number_input(
        "虚拟种子家庭数量",
        min_value=100,
        max_value=50000,
        value=1000,
        step=100,
        help="从概率表生成的虚拟种子样本数量"
    )

    st.sidebar.info("💡 概率表可在主界面Tab 1中编辑")

# --- 4. 其他参数 ---
st.sidebar.markdown("### 4️⃣ 其他参数")

random_seed = st.sidebar.number_input(
    "随机种子",
    min_value=1,
    max_value=9999,
    value=1,
    help="确保结果可复现"
)

osm_buffer_km = st.sidebar.slider(
    "OSM下载缓冲区 (km)",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.5,
    help="研究区域边界向外扩展距离，减轻边界效应"
)

activitysim_sample_rate = st.sidebar.slider(
    "ActivitySim采样率",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
    step=0.1,
    help="1.0=全样本，0.1=10%样本（用于快速测试）"
)

# ============================================================================
# 主区域：Tab布局
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 参数配置",
    "🚀 运行流程",
    "📈 结果可视化",
    "📥 结果下载"
])

# ============================================================================
# Tab 1: 参数配置（转换系数 + 概率表）
# ============================================================================

with tab1:
    subtab1, subtab2 = st.tabs(["🏢 建筑转换系数", "📊 概率分布表（ActivitySim格式）"])

    # ===== 子Tab 1: 建筑转换系数 =====
    with subtab1:
        st.markdown('<div class="sub-header">建筑面积 → 就业/人口 转换系数</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            以下系数用于将OSM建筑面积转换为就业岗位数和人口数。
            可根据实际情况调整。
        </div>
        """, unsafe_allow_html=True)

        coef_df = st.session_state.conversion_coefficients.to_dataframe()

        st.markdown("#### 就业类建筑")
        employment_types = ['office', 'retail', 'education', 'healthcare',
                            'industrial', 'hospitality', 'transport', 'other']
        emp_coef = coef_df[coef_df['building_type'].isin(employment_types)]

        edited_emp = st.data_editor(
            emp_coef,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "building_type": st.column_config.TextColumn("建筑类型", disabled=True),
                "parameter": st.column_config.TextColumn("参数", disabled=True),
                "value": st.column_config.NumberColumn("系数值", format="%.2f")
            }
        )

        st.markdown("#### 住宅类建筑")
        res_coef = coef_df[coef_df['building_type'] == 'residential']

        edited_res = st.data_editor(
            res_coef,
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "building_type": st.column_config.TextColumn("建筑类型", disabled=True),
                "parameter": st.column_config.TextColumn("参数", disabled=True),
                "value": st.column_config.NumberColumn("系数值", format="%.2f")
            }
        )

        edited_coef = pd.concat([edited_emp, edited_res], ignore_index=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("📝 应用系数修改", type="primary", key="apply_coef"):
                st.session_state.conversion_coefficients.from_dataframe(edited_coef)
                st.success("✅ 系数已更新！")

        with col2:
            if st.button("🔄 恢复默认系数", key="reset_coef"):
                st.session_state.conversion_coefficients = ConversionCoefficients()
                st.success("✅ 已恢复默认系数！")
                st.rerun()

    # ===== 子Tab 2: 概率分布表（ActivitySim格式）=====
    with subtab2:
        st.markdown('<div class="sub-header">概率分布表（ActivitySim兼容格式）</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            以下概率表用于在"无种子模式"下生成虚拟家庭和人口样本。
            <strong>字段定义遵循ActivitySim标准</strong>，确保生成的数据可被ActivitySim直接使用。
        </div>
        """, unsafe_allow_html=True)

        # 1. 家庭规模分布
        st.markdown("#### 1️⃣ 家庭规模分布 (hhsize)")
        st.caption("定义不同家庭规模的概率分布")

        edited_hh_size = st.data_editor(
            st.session_state.prob_tables['hh_size_dist'],
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "hh_size": st.column_config.NumberColumn("家庭规模", min_value=1, max_value=10, step=1),
                "hhsize": st.column_config.NumberColumn("hhsize（ActivitySim列名）", min_value=1, max_value=10,
                                                        step=1),
                "probability": st.column_config.NumberColumn("概率", min_value=0.0, max_value=1.0, format="%.3f"),
                "cumulative_prob": st.column_config.NumberColumn("累积概率", min_value=0.0, max_value=1.0,
                                                                 format="%.3f")
            },
            key="edit_hh_size"
        )

        prob_sum = edited_hh_size['probability'].sum()
        if abs(prob_sum - 1.0) > 0.01:
            st.warning(f"⚠️ 概率总和应为1.0，当前为{prob_sum:.3f}")

        # 2. 家庭类型分布 HHT
        st.markdown("#### 2️⃣ 家庭类型分布 (HHT)")
        st.caption("""
        Census定义的家庭类型:
        1=已婚夫妇家庭, 2=男户主其他家庭, 3=女户主其他家庭,
        4=单人男性, 5=单人女性, 6=非家庭男户主, 7=非家庭女户主
        """)

        edited_hht = st.data_editor(
            st.session_state.prob_tables['hht_dist'],
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "HHT": st.column_config.NumberColumn("家庭类型HHT", min_value=1, max_value=7, step=1),
                "probability": st.column_config.NumberColumn("概率", min_value=0.0, max_value=1.0, format="%.3f"),
                "cumulative_prob": st.column_config.NumberColumn("累积概率", format="%.3f")
            },
            key="edit_hht"
        )

        # 3. 收入分布
        st.markdown("#### 3️⃣ 收入类别分布")
        st.caption("定义收入类别的概率分布和收入范围（元/年）")

        edited_income = st.data_editor(
            st.session_state.prob_tables['income_dist'],
            hide_index=True,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "income_cat": st.column_config.NumberColumn("收入类别", min_value=1, step=1),
                "income_min": st.column_config.NumberColumn("最低收入", format="%d"),
                "income_max": st.column_config.NumberColumn("最高收入", format="%d"),
                "probability": st.column_config.NumberColumn("概率", min_value=0.0, max_value=1.0, format="%.3f"),
                "cumulative_prob": st.column_config.NumberColumn("累积概率", format="%.3f")
            },
            key="edit_income"
        )

        # 4. 车辆拥有条件分布
        st.markdown("#### 4️⃣ 车辆拥有条件分布")
        st.caption("P(auto_ownership | income_cat, hh_size_group)")
        st.info("家庭规模组: 1=1-2人, 2=3-4人, 3=5人及以上")

        edited_auto = st.data_editor(
            st.session_state.prob_tables['auto_ownership'],
            hide_index=True,
            use_container_width=True,
            height=300,
            column_config={
                "income_cat": st.column_config.NumberColumn("收入类别", min_value=1, max_value=5, step=1),
                "hh_size_group": st.column_config.NumberColumn("家庭规模组", min_value=1, max_value=3, step=1),
                "auto_ownership": st.column_config.NumberColumn("车辆数", min_value=0, max_value=5, step=1),
                "num_auto": st.column_config.NumberColumn("num_auto", min_value=0, max_value=5, step=1),
                "probability": st.column_config.NumberColumn("概率", min_value=0.0, max_value=1.0, format="%.3f")
            },
            key="edit_auto"
        )

        # 5. 年龄分布
        st.markdown("#### 5️⃣ 年龄组分布")
        st.caption("与ActivitySim ptype对应的年龄组")

        edited_age = st.data_editor(
            st.session_state.prob_tables['age_dist'],
            hide_index=True,
            use_container_width=True,
            column_config={
                "age_cat": st.column_config.NumberColumn("年龄组", min_value=1, step=1),
                "age_min": st.column_config.NumberColumn("最小年龄", min_value=0, max_value=100, step=1),
                "age_max": st.column_config.NumberColumn("最大年龄", min_value=0, max_value=100, step=1),
                "age_group_name": st.column_config.TextColumn("年龄组名称"),
                "probability": st.column_config.NumberColumn("概率", min_value=0.0, max_value=1.0, format="%.3f"),
                "cumulative_prob": st.column_config.NumberColumn("累积概率", format="%.3f")
            },
            key="edit_age"
        )

        # 6. 就业状态条件分布 pemploy
        st.markdown("#### 6️⃣ 就业状态条件分布 (pemploy)")
        st.caption("""
        ActivitySim就业状态定义:
        1=全职就业, 2=兼职就业, 3=失业但有工作经历, 4=非劳动力
        """)

        edited_pemploy = st.data_editor(
            st.session_state.prob_tables['pemploy_dist'],
            hide_index=True,
            use_container_width=True,
            height=300,
            column_config={
                "age_cat": st.column_config.NumberColumn("年龄组", min_value=1, max_value=6, step=1),
                "pemploy": st.column_config.NumberColumn("就业状态", min_value=1, max_value=4, step=1),
                "probability": st.column_config.NumberColumn("概率", min_value=0.0, max_value=1.0, format="%.3f")
            },
            key="edit_pemploy"
        )

        # 7. 学生状态条件分布 pstudent
        st.markdown("#### 7️⃣ 学生状态条件分布 (pstudent)")
        st.caption("""
        ActivitySim学生状态定义:
        1=学龄前, 2=K-12学生, 3=大学生, 4=非学生
        """)

        edited_pstudent = st.data_editor(
            st.session_state.prob_tables['pstudent_dist'],
            hide_index=True,
            use_container_width=True,
            height=250,
            column_config={
                "age_cat": st.column_config.NumberColumn("年龄组", min_value=1, max_value=6, step=1),
                "pstudent": st.column_config.NumberColumn("学生状态", min_value=1, max_value=4, step=1),
                "probability": st.column_config.NumberColumn("概率", min_value=0.0, max_value=1.0, format="%.3f")
            },
            key="edit_pstudent"
        )

        # 8. 性别分布
        st.markdown("#### 8️⃣ 性别分布")
        st.caption("1=男性, 2=女性")

        edited_sex = st.data_editor(
            st.session_state.prob_tables['sex_dist'],
            hide_index=True,
            use_container_width=True,
            column_config={
                "sex": st.column_config.NumberColumn("性别代码", min_value=1, max_value=2, step=1),
                "sex_name": st.column_config.TextColumn("性别名称"),
                "probability": st.column_config.NumberColumn("概率", min_value=0.0, max_value=1.0, format="%.3f"),
                "cumulative_prob": st.column_config.NumberColumn("累积概率", format="%.3f")
            },
            key="edit_sex"
        )

        # 保存按钮
        st.markdown("---")

        # 显示ptype映射参考
        with st.expander("📖 ActivitySim人员类型(ptype)参考"):
            st.markdown("""
            **ptype 定义：**
            | ptype | 描述 | 确定条件 |
            |-------|------|----------|
            | 1 | 全职工作者 | pemploy=1 |
            | 2 | 兼职工作者 | pemploy=2 |
            | 3 | 大学生 | pstudent=3 |
            | 4 | 非工作成人 | 18-64岁, pemploy=4, pstudent=4 |
            | 5 | 退休人员 | 65+岁, pemploy=4 |
            | 6 | 驾龄儿童 | 16-17岁, 非全职工作 |
            | 7 | 非驾龄儿童 | 6-15岁 |
            | 8 | 学龄前儿童 | 0-5岁 |

            ptype由系统根据age、pemploy、pstudent自动计算。
            """)

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("💾 保存所有概率表修改", type="primary", key="save_prob"):
                st.session_state.prob_tables = {
                    'hh_size_dist': edited_hh_size,
                    'hht_dist': edited_hht,
                    'income_dist': edited_income,
                    'auto_ownership': edited_auto,
                    'age_dist': edited_age,
                    'pemploy_dist': edited_pemploy,
                    'pstudent_dist': edited_pstudent,
                    'sex_dist': edited_sex,
                    'ptype_mapping': st.session_state.prob_tables.get('ptype_mapping',
                                                                       get_default_probability_tables()['ptype_mapping'])
                }
                st.success("✅ 概率表已更新！")

        with col2:
            if st.button("🔄 恢复默认概率表", key="reset_prob"):
                st.session_state.prob_tables = get_default_probability_tables()
                st.success("✅ 已恢复默认概率表！")
                st.rerun()

# ============================================================================
# Tab 2: 运行流程
# ============================================================================

with tab2:
    st.markdown('<div class="sub-header">流程运行控制</div>', unsafe_allow_html=True)

    # 显示库状态
    st.markdown("#### 📦 依赖库状态")
    display_library_status()

    st.markdown("---")

    # 检查必需输入
    can_run = False
    missing_inputs = []

    if 'shapefile' not in st.session_state.uploaded_files:
        missing_inputs.append("研究区域Shapefile")

    if popsim_mode == 'with_seed':
        if 'hh_seed' not in st.session_state.uploaded_files:
            missing_inputs.append("households_seed.csv")
        if 'per_seed' not in st.session_state.uploaded_files:
            missing_inputs.append("persons_seed.csv")

    if use_matsim_network and 'network' not in st.session_state.uploaded_files:
        missing_inputs.append("MATSim network.xml")

    if missing_inputs:
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ <strong>缺少必需输入：</strong><br>
            {'<br>'.join([f"• {item}" for item in missing_inputs])}
        </div>
        """, unsafe_allow_html=True)
    else:
        can_run = True
        st.markdown("""
        <div class="success-box">
            ✅ <strong>所有必需输入已就绪，可以开始运行！</strong>
        </div>
        """, unsafe_allow_html=True)

    # 运行模式信息
    st.markdown("#### ⚙️ 当前配置")
    config_info = f"""
    - **PopulationSim模式**: {'有种子' if popsim_mode == 'with_seed' else f'无种子（{num_virtual_hh}户）'}
    - **Skim构建**: {'MATSim路网' if use_matsim_network else f'简化欧氏距离（{avg_speed}km/h）'}
    - **ActivitySim采样率**: {activitysim_sample_rate * 100:.0f}%
    - **随机种子**: {random_seed}
    """
    st.markdown(config_info)

    # 运行按钮
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        run_button = st.button(
            "🚀 运行完整流程",
            type="primary",
            disabled=not can_run or st.session_state.pipeline_running,
            use_container_width=True
        )

    with col2:
        if st.button("🗑️ 清除结果", use_container_width=True):
            st.session_state.pipeline_state = None
            st.session_state.progress_messages = []
            st.session_state.current_progress = 0.0
            st.success("已清除结果")
            st.rerun()

    # 运行流程
    if run_button:
        st.session_state.pipeline_running = True
        st.session_state.progress_messages = []

        try:
            project_root = Path(__file__).parent
            temp_dir = Path(tempfile.mkdtemp(prefix="tdm_"))

            st.info(f"📁 项目目录: {project_root}")
            st.info(f"📁 临时工作目录: {temp_dir}")

            # 保存上传的文件
            st.write("正在保存上传的文件...")

            input_dir = temp_dir / 'data' / 'input'
            ensure_dir(input_dir)

            shapefile_path = save_uploaded_file(
                st.session_state.uploaded_files['shapefile'],
                input_dir / 'shapefiles'
            )

            network_path = None
            if use_matsim_network and 'network' in st.session_state.uploaded_files:
                network_path = save_uploaded_file(
                    st.session_state.uploaded_files['network'],
                    input_dir / 'network'
                )

            if popsim_mode == 'with_seed':
                seed_dir = input_dir / 'seed'
                ensure_dir(seed_dir)
                save_uploaded_file(st.session_state.uploaded_files['hh_seed'], seed_dir)
                save_uploaded_file(st.session_state.uploaded_files['per_seed'], seed_dir)

            # 创建配置
            config = PipelineConfig()
            config.project_root = project_root
            config.config_dir = project_root / 'config'
            config.data_dir = temp_dir / 'data'
            config.output_dir = temp_dir / 'output'

            ensure_dir(config.data_dir)
            ensure_dir(config.output_dir)

            config.shapefile_path = shapefile_path
            config.network_file_path = network_path
            config.populationsim_mode = popsim_mode
            config.num_virtual_households = num_virtual_hh if popsim_mode == 'from_prob_tables' else 1000
            config.random_seed = random_seed
            config.osm_buffer_km = osm_buffer_km
            config.use_matsim_network = use_matsim_network
            config.average_speed_kmh = avg_speed if not use_matsim_network else 40.0
            config.activitysim_sample_rate = activitysim_sample_rate
            config.conversion_coefficients = st.session_state.conversion_coefficients

            # 概率表
            if popsim_mode == 'from_prob_tables':
                config.prob_tables = st.session_state.prob_tables
            else:
                config.prob_tables = None

            # 进度回调
            progress_callback = StreamlitProgressCallback(total_steps=100)

            # 运行流程
            st.write("---")
            st.markdown("### 🔄 流程执行中...")

            pipeline = TravelDemandModelPipeline(config, progress_callback)
            state = pipeline.run()

            st.session_state.pipeline_state = state

            # 显示警告
            if state.warnings:
                st.markdown("#### ⚠️ 警告信息")
                for warning in state.warnings:
                    st.warning(warning)

            st.markdown("""
            <div class="success-box">
                🎉 <strong>流程执行完成！</strong><br>
                请切换到"结果可视化"和"结果下载"标签页查看结果。
            </div>
            """, unsafe_allow_html=True)

            st.balloons()

        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
                ❌ <strong>流程执行失败:</strong><br>
                {str(e)}
            </div>
            """, unsafe_allow_html=True)
            st.exception(e)

        finally:
            st.session_state.pipeline_running = False

    # 显示历史运行信息
    if st.session_state.pipeline_state is not None:
        st.write("---")
        st.markdown("### 📊 上次运行摘要")

        state = st.session_state.pipeline_state

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("执行时间", f"{state.get_elapsed_time():.1f}秒")

        with col2:
            if state.synthetic_households is not None:
                st.metric("合成家庭", f"{len(state.synthetic_households):,}")

        with col3:
            if state.synthetic_persons is not None:
                st.metric("合成人口", f"{len(state.synthetic_persons):,}")

        with col4:
            if state.trips is not None:
                st.metric("生成出行", f"{len(state.trips):,}")

# ============================================================================
# Tab 3: 结果可视化
# ============================================================================

with tab3:
    st.markdown('<div class="sub-header">结果可视化</div>', unsafe_allow_html=True)

    if st.session_state.pipeline_state is None:
        st.info("ℹ️ 请先运行流程以生成结果")
    else:
        state = st.session_state.pipeline_state

        # --- 1. 研究区域地图 ---
        if state.study_area is not None:
            st.markdown("#### 🗺️ 研究区域与TAZ")

            try:
                # 准备数据
                plot_data = state.study_area.copy()
                if 'land_use' in dir(state) and state.land_use is not None:
                    plot_data = plot_data.merge(
                        state.land_use[['zone_id', 'pop', 'emp_total']],
                        on='zone_id',
                        how='left'
                    )

                color_col = 'pop' if 'pop' in plot_data.columns else 'area_km2'

                fig = px.choropleth_mapbox(
                    plot_data,
                    geojson=plot_data.geometry.__geo_interface__,
                    locations=plot_data.index,
                    color=color_col,
                    hover_name='zone_id',
                    mapbox_style="carto-positron",
                    zoom=10,
                    opacity=0.6,
                    color_continuous_scale="YlOrRd"
                )

                fig.update_layout(height=500, margin={"r": 0, "t": 0, "l": 0, "b": 0})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"地图渲染失败: {e}")

        # --- 2. 土地利用统计 ---
        if state.land_use is not None:
            st.markdown("#### 📊 土地利用统计")

            col1, col2 = st.columns(2)

            with col1:
                emp_cols = [c for c in state.land_use.columns if c.startswith('emp_') and c != 'emp_total']
                if emp_cols:
                    emp_data = state.land_use[emp_cols].sum()

                    fig = px.pie(
                        values=emp_data.values,
                        names=[c.replace('emp_', '').title() for c in emp_data.index],
                        title="就业岗位类型分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'density' in state.land_use.columns:
                    fig = px.histogram(
                        state.land_use,
                        x='density',
                        nbins=30,
                        title="人口密度分布 (人/km²)",
                        labels={'density': '人口密度'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # --- 3. 合成人口统计 ---
        if state.synthetic_persons is not None:
            st.markdown("#### 👥 合成人口统计")

            col1, col2, col3 = st.columns(3)

            with col1:
                if 'age' in state.synthetic_persons.columns:
                    fig = px.histogram(
                        state.synthetic_persons,
                        x='age',
                        nbins=20,
                        title="年龄分布",
                        labels={'age': '年龄'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'ptype' in state.synthetic_persons.columns:
                    ptype_names = {
                        1: '全职工作', 2: '兼职工作', 3: '大学生', 4: '非工作成人',
                        5: '退休', 6: '驾龄儿童', 7: '非驾龄儿童', 8: '学龄前'
                    }
                    ptype_counts = state.synthetic_persons['ptype'].value_counts().sort_index()
                    ptype_labels = [ptype_names.get(i, f'类型{i}') for i in ptype_counts.index]

                    fig = px.bar(
                        x=ptype_labels,
                        y=ptype_counts.values,
                        title="人员类型分布 (ptype)",
                        labels={'x': '人员类型', 'y': '人数'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col3:
                if 'pemploy' in state.synthetic_persons.columns:
                    pemploy_names = {1: '全职', 2: '兼职', 3: '失业', 4: '非劳动力'}
                    pemploy_counts = state.synthetic_persons['pemploy'].value_counts().sort_index()
                    pemploy_labels = [pemploy_names.get(i, f'{i}') for i in pemploy_counts.index]

                    fig = px.pie(
                        values=pemploy_counts.values,
                        names=pemploy_labels,
                        title="就业状态分布 (pemploy)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # 家庭规模分布
            if state.synthetic_households is not None and 'hhsize' in state.synthetic_households.columns:
                col1, col2 = st.columns(2)

                with col1:
                    hh_size_counts = state.synthetic_households['hhsize'].value_counts().sort_index()

                    fig = px.bar(
                        x=hh_size_counts.index,
                        y=hh_size_counts.values,
                        title="家庭规模分布",
                        labels={'x': '家庭规模', 'y': '家庭数量'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    if 'auto_ownership' in state.synthetic_households.columns:
                        auto_counts = state.synthetic_households['auto_ownership'].value_counts().sort_index()

                        fig = px.bar(
                            x=auto_counts.index,
                            y=auto_counts.values,
                            title="车辆拥有分布",
                            labels={'x': '车辆数', 'y': '家庭数量'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # --- 4. 出行统计 ---
        if state.trips is not None and len(state.trips) > 0:
            st.markdown("#### 🚌 出行统计")

            col1, col2 = st.columns(2)

            with col1:
                if 'trip_mode' in state.trips.columns:
                    mode_counts = state.trips['trip_mode'].value_counts()

                    fig = px.pie(
                        values=mode_counts.values,
                        names=mode_counts.index,
                        title="出行方式分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'purpose' in state.trips.columns:
                    purpose_counts = state.trips['purpose'].value_counts()

                    fig = px.bar(
                        x=purpose_counts.index,
                        y=purpose_counts.values,
                        title="出行目的分布",
                        labels={'x': '出行目的', 'y': '出行次数'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # OD矩阵热力图
            origin_col = 'origin' if 'origin' in state.trips.columns else None
            dest_col = 'destination' if 'destination' in state.trips.columns else None

            if origin_col and dest_col:
                st.markdown("#### 🔥 OD矩阵热力图")

                od_matrix = state.trips.groupby([origin_col, dest_col]).size().reset_index(name='trips')
                od_pivot = od_matrix.pivot(index=origin_col, columns=dest_col, values='trips').fillna(0)

                if len(od_pivot) > 50:
                    st.warning("⚠️ TAZ数量过多，仅显示前50个zone的OD矩阵")
                    od_pivot = od_pivot.iloc[:50, :50]

                fig = px.imshow(
                    od_pivot,
                    labels=dict(x="目的地Zone", y="起点Zone", color="出行次数"),
                    title="OD出行矩阵",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Tab 4: 结果下载
# ============================================================================

with tab4:
    st.markdown('<div class="sub-header">结果文件下载</div>', unsafe_allow_html=True)

    if st.session_state.pipeline_state is None:
        st.info("ℹ️ 请先运行流程以生成结果")
    else:
        state = st.session_state.pipeline_state

        st.markdown("""
        <div class="info-box">
            点击下方按钮下载各类结果文件。所有CSV文件使用UTF-8编码，可直接用于ActivitySim。
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 📍 土地利用与研究区域")

            if state.land_use is not None:
                csv = state.land_use.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载 land_use.csv",
                    data=csv,
                    file_name="land_use.csv",
                    mime="text/csv"
                )

            if state.study_area is not None:
                geojson = state.study_area.to_json()
                st.download_button(
                    label="📥 下载 study_area.geojson",
                    data=geojson,
                    file_name="study_area.geojson",
                    mime="application/json"
                )

        with col2:
            st.markdown("##### 👥 合成人口（ActivitySim格式）")

            if state.synthetic_households is not None:
                csv = state.synthetic_households.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载 households.csv",
                    data=csv,
                    file_name="households.csv",
                    mime="text/csv"
                )

            if state.synthetic_persons is not None:
                csv = state.synthetic_persons.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载 persons.csv",
                    data=csv,
                    file_name="persons.csv",
                    mime="text/csv"
                )

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("##### 🚌 出行结果")

            if state.tours is not None and len(state.tours) > 0:
                csv = state.tours.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载 tours.csv",
                    data=csv,
                    file_name="tours.csv",
                    mime="text/csv"
                )

            if state.trips is not None and len(state.trips) > 0:
                csv = state.trips.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载 trips.csv",
                    data=csv,
                    file_name="trips.csv",
                    mime="text/csv"
                )

        with col4:
            st.markdown("##### 📊 统计报告")

            if state.statistics:
                import json
                stats_json = json.dumps(state.statistics, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 下载 statistics.json",
                    data=stats_json,
                    file_name="statistics.json",
                    mime="application/json"
                )

        # 显示数据格式说明
        st.markdown("---")
        with st.expander("📖 ActivitySim数据格式说明"):
            st.markdown("""
            ### households.csv 列说明
            | 列名 | 描述 |
            |------|------|
            | household_id | 家庭唯一标识 |
            | home_zone_id | 居住地TAZ ID |
            | income | 家庭年收入 |
            | hhsize | 家庭规模 |
            | HHT | 家庭类型 (1-7) |
            | auto_ownership | 车辆数量 |
            | num_workers | 就业人数 |

            ### persons.csv 列说明
            | 列名 | 描述 |
            |------|------|
            | person_id | 人员唯一标识 |
            | household_id | 所属家庭ID |
            | age | 年龄 |
            | sex | 性别 (1=男, 2=女) |
            | pemploy | 就业状态 (1=全职, 2=兼职, 3=失业, 4=非劳动力) |
            | pstudent | 学生状态 (1=学龄前, 2=K-12, 3=大学生, 4=非学生) |
            | ptype | 人员类型 (1-8) |
            | PNUM | 家庭内人员编号 |
            """)

# ============================================================================
# 页脚
# ============================================================================

st.markdown("---")

# 显示依赖状态摘要
deps = check_dependencies()
dep_status = " | ".join([
    f"{'✓' if v else '✗'} {k}"
    for k, v in deps.items()
])

st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>出行需求建模系统 v2.0 | 集成 PopulationSim & ActivitySim</p>
    <p>依赖状态: {dep_status}</p>
    <p>💡 提示：首次运行可能需要较长时间下载OSM数据，请耐心等待</p>
</div>
""", unsafe_allow_html=True)
