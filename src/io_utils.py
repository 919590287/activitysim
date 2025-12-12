# ============================================================================
# io_utils.py
# ============================================================================
# 模块职责：提供项目通用的文件I/O操作和工具函数
# 包括：
# - 路径管理与目录创建
# - ZIP文件解压
# - CSV/YAML/JSON文件读写
# - 日志工具
# - 配置加载与验证
# ============================================================================

import os
import sys
import json
import yaml
import shutil
import zipfile
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import pandas as pd
import numpy as np


# ----------------------------------------------------------------------------
# 日志配置
# ----------------------------------------------------------------------------

def setup_logger(
        name: str = "travel_demand_model",
        log_level: int = logging.INFO,
        log_file: Optional[str] = None
) -> logging.Logger:
    """
    配置并返回日志记录器

    参数:
        name: 日志记录器名称
        log_level: 日志级别（DEBUG/INFO/WARNING/ERROR）
        log_file: 日志文件路径，如为None则仅输出到控制台

    返回:
        配置好的Logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出（可选）
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 默认全局日志记录器
logger = setup_logger()


# ----------------------------------------------------------------------------
# 路径与目录管理
# ----------------------------------------------------------------------------

def get_project_root() -> Path:
    """
    获取项目根目录路径
    假设此文件位于 project_root/src/ 目录下

    返回:
        项目根目录的Path对象
    """
    current_file = Path(__file__).resolve()
    # 向上两级：src -> project_root
    project_root = current_file.parent.parent
    return project_root


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    确保目录存在，如不存在则创建

    参数:
        dir_path: 目录路径

    返回:
        目录的Path对象
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_dir(dir_path: Union[str, Path], keep_dir: bool = True) -> None:
    """
    清空目录内容

    参数:
        dir_path: 要清空的目录路径
        keep_dir: 是否保留空目录，默认True
    """
    path = Path(dir_path)
    if path.exists():
        shutil.rmtree(path)
    if keep_dir:
        path.mkdir(parents=True, exist_ok=True)


def get_temp_dir(prefix: str = "tdm_") -> Path:
    """
    创建临时目录

    参数:
        prefix: 临时目录名称前缀

    返回:
        临时目录的Path对象
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return Path(temp_dir)


def list_files(
        dir_path: Union[str, Path],
        extensions: Optional[List[str]] = None,
        recursive: bool = False
) -> List[Path]:
    """
    列出目录中的文件

    参数:
        dir_path: 目录路径
        extensions: 文件扩展名过滤列表，如['.csv', '.shp']
        recursive: 是否递归搜索子目录

    返回:
        文件路径列表
    """
    path = Path(dir_path)
    if not path.exists():
        return []

    if recursive:
        files = list(path.rglob("*"))
    else:
        files = list(path.iterdir())

    # 只保留文件（排除目录）
    files = [f for f in files if f.is_file()]

    # 按扩展名过滤
    if extensions:
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                      for ext in extensions]
        files = [f for f in files if f.suffix.lower() in extensions]

    return sorted(files)


# ----------------------------------------------------------------------------
# ZIP文件处理
# ----------------------------------------------------------------------------

def extract_zip(
        zip_path: Union[str, Path],
        extract_to: Optional[Union[str, Path]] = None
) -> Path:
    """
    解压ZIP文件

    参数:
        zip_path: ZIP文件路径
        extract_to: 解压目标目录，如为None则解压到ZIP同目录下的同名文件夹

    返回:
        解压后的目录路径

    异常:
        ValueError: 如果文件不是有效的ZIP格式
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP文件不存在: {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"文件不是有效的ZIP格式: {zip_path}")

    if extract_to is None:
        extract_to = zip_path.parent / zip_path.stem

    extract_to = Path(extract_to)
    ensure_dir(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)

    logger.info(f"已解压ZIP文件到: {extract_to}")
    return extract_to


def create_zip(
        source_paths: List[Union[str, Path]],
        zip_path: Union[str, Path],
        base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    创建ZIP压缩文件

    参数:
        source_paths: 要压缩的文件或目录列表
        zip_path: 输出ZIP文件路径
        base_dir: 压缩时的基准目录（用于计算相对路径）

    返回:
        创建的ZIP文件路径
    """
    zip_path = Path(zip_path)
    ensure_dir(zip_path.parent)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for source in source_paths:
            source = Path(source)
            if source.is_file():
                if base_dir:
                    arcname = source.relative_to(base_dir)
                else:
                    arcname = source.name
                zf.write(source, arcname)
            elif source.is_dir():
                for file in source.rglob("*"):
                    if file.is_file():
                        if base_dir:
                            arcname = file.relative_to(base_dir)
                        else:
                            arcname = file.relative_to(source.parent)
                        zf.write(file, arcname)

    logger.info(f"已创建ZIP文件: {zip_path}")
    return zip_path


# ----------------------------------------------------------------------------
# CSV文件读写
# ----------------------------------------------------------------------------

def read_csv(
        file_path: Union[str, Path],
        encoding: str = 'utf-8',
        **kwargs
) -> pd.DataFrame:
    """
    读取CSV文件为DataFrame

    参数:
        file_path: CSV文件路径
        encoding: 文件编码，默认utf-8
        **kwargs: 传递给pd.read_csv的额外参数

    返回:
        DataFrame对象

    说明:
        - 自动跳过以#开头的注释行
        - 自动处理常见编码问题
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV文件不存在: {file_path}")

    # 尝试多种编码
    encodings_to_try = [encoding, 'utf-8', 'gbk', 'gb2312', 'latin1']

    for enc in encodings_to_try:
        try:
            # 先读取文件内容，过滤注释行
            with open(file_path, 'r', encoding=enc) as f:
                lines = [line for line in f if not line.strip().startswith('#')]

            # 使用StringIO读取过滤后的内容
            from io import StringIO
            content = ''.join(lines)
            df = pd.read_csv(StringIO(content), **kwargs)

            logger.debug(f"成功读取CSV文件: {file_path} (编码: {enc})")
            return df

        except UnicodeDecodeError:
            continue
        except Exception as e:
            if enc == encodings_to_try[-1]:
                raise
            continue

    raise ValueError(f"无法以任何尝试的编码读取文件: {file_path}")


def write_csv(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        encoding: str = 'utf-8',
        index: bool = False,
        **kwargs
) -> Path:
    """
    将DataFrame写入CSV文件

    参数:
        df: 要写入的DataFrame
        file_path: 输出文件路径
        encoding: 文件编码
        index: 是否写入索引列
        **kwargs: 传递给df.to_csv的额外参数

    返回:
        写入的文件路径
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    df.to_csv(file_path, encoding=encoding, index=index, **kwargs)
    logger.debug(f"已写入CSV文件: {file_path} ({len(df)}行)")

    return file_path


# ----------------------------------------------------------------------------
# YAML文件读写
# ----------------------------------------------------------------------------

def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取YAML配置文件

    参数:
        file_path: YAML文件路径

    返回:
        解析后的字典
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    logger.debug(f"已读取YAML文件: {file_path}")
    return data or {}


def write_yaml(
        data: Dict[str, Any],
        file_path: Union[str, Path],
        default_flow_style: bool = False
) -> Path:
    """
    将字典写入YAML文件

    参数:
        data: 要写入的字典
        file_path: 输出文件路径
        default_flow_style: YAML格式风格

    返回:
        写入的文件路径
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=default_flow_style,
                  allow_unicode=True, sort_keys=False)

    logger.debug(f"已写入YAML文件: {file_path}")
    return file_path


# ----------------------------------------------------------------------------
# JSON文件读写
# ----------------------------------------------------------------------------

def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    读取JSON文件

    参数:
        file_path: JSON文件路径

    返回:
        解析后的字典或列表
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.debug(f"已读取JSON文件: {file_path}")
    return data


def write_json(
        data: Any,
        file_path: Union[str, Path],
        indent: int = 2
) -> Path:
    """
    将数据写入JSON文件

    参数:
        data: 要写入的数据
        file_path: 输出文件路径
        indent: 缩进空格数

    返回:
        写入的文件路径
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.debug(f"已写入JSON文件: {file_path}")
    return file_path


# ----------------------------------------------------------------------------
# 配置管理
# ----------------------------------------------------------------------------

class ConfigManager:
    """
    配置管理器：统一管理项目配置文件的加载和访问

    使用示例:
        config = ConfigManager("config/populationsim")
        settings = config.get("settings.yaml")
        value = config.get_value("settings.yaml", "max_iterations", default=100)
    """

    def __init__(self, config_dir: Union[str, Path]):
        """
        初始化配置管理器

        参数:
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Any] = {}

    def get(self, filename: str, reload: bool = False) -> Dict[str, Any]:
        """
        获取配置文件内容

        参数:
            filename: 配置文件名
            reload: 是否强制重新加载（忽略缓存）

        返回:
            配置字典
        """
        if filename not in self._cache or reload:
            file_path = self.config_dir / filename

            if filename.endswith('.yaml') or filename.endswith('.yml'):
                self._cache[filename] = read_yaml(file_path)
            elif filename.endswith('.json'):
                self._cache[filename] = read_json(file_path)
            elif filename.endswith('.csv'):
                self._cache[filename] = read_csv(file_path).to_dict('records')
            else:
                raise ValueError(f"不支持的配置文件格式: {filename}")

        return self._cache[filename]

    def get_value(
            self,
            filename: str,
            key: str,
            default: Any = None
    ) -> Any:
        """
        获取配置文件中的特定值

        参数:
            filename: 配置文件名
            key: 配置键名，支持点号分隔的嵌套键如"a.b.c"
            default: 键不存在时的默认值

        返回:
            配置值
        """
        config = self.get(filename)

        keys = key.split('.')
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set_value(
            self,
            filename: str,
            key: str,
            value: Any,
            save: bool = True
    ) -> None:
        """
        设置配置值

        参数:
            filename: 配置文件名
            key: 配置键名
            value: 要设置的值
            save: 是否立即保存到文件
        """
        config = self.get(filename)

        keys = key.split('.')
        target = config

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

        if save:
            file_path = self.config_dir / filename
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                write_yaml(config, file_path)
            elif filename.endswith('.json'):
                write_json(config, file_path)


# ----------------------------------------------------------------------------
# 数据验证工具
# ----------------------------------------------------------------------------

def validate_dataframe(
        df: pd.DataFrame,
        required_columns: List[str],
        df_name: str = "DataFrame"
) -> List[str]:
    """
    验证DataFrame是否包含必需的列

    参数:
        df: 要验证的DataFrame
        required_columns: 必需的列名列表
        df_name: DataFrame名称（用于错误信息）

    返回:
        缺失的列名列表（空列表表示验证通过）
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        logger.warning(f"{df_name} 缺少必需列: {missing}")

    return missing


def validate_file_exists(
        file_path: Union[str, Path],
        file_description: str = "文件"
) -> bool:
    """
    验证文件是否存在

    参数:
        file_path: 文件路径
        file_description: 文件描述（用于错误信息）

    返回:
        文件是否存在
    """
    exists = Path(file_path).exists()

    if not exists:
        logger.warning(f"{file_description}不存在: {file_path}")

    return exists


# ----------------------------------------------------------------------------
# 进度回调工具
# ----------------------------------------------------------------------------

class ProgressCallback:
    """
    进度回调类：用于向Streamlit等UI报告执行进度

    使用示例:
        callback = ProgressCallback(total_steps=5)
        callback.update(1, "正在处理数据...")
        callback.update(2, "正在计算...", progress=0.5)
    """

    def __init__(
            self,
            total_steps: int = 100,
            callback_func: Optional[callable] = None
    ):
        """
        初始化进度回调

        参数:
            total_steps: 总步骤数
            callback_func: 回调函数，签名为 func(step, message, progress)
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.callback_func = callback_func
        self.messages: List[str] = []

    def update(
            self,
            step: Optional[int] = None,
            message: str = "",
            progress: Optional[float] = None
    ) -> None:
        """
        更新进度

        参数:
            step: 当前步骤编号
            message: 进度消息
            progress: 进度百分比（0-1），如为None则根据step计算
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        if progress is None:
            progress = self.current_step / self.total_steps

        self.messages.append(f"[{self.current_step}/{self.total_steps}] {message}")
        logger.info(message)

        if self.callback_func:
            self.callback_func(self.current_step, message, progress)

    def get_progress(self) -> float:
        """获取当前进度百分比"""
        return self.current_step / self.total_steps

    def get_messages(self) -> List[str]:
        """获取所有进度消息"""
        return self.messages.copy()


# ----------------------------------------------------------------------------
# 时间戳工具
# ----------------------------------------------------------------------------

def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    获取当前时间戳字符串

    参数:
        format_str: 时间格式字符串

    返回:
        格式化的时间戳
    """
    return datetime.now().strftime(format_str)


def create_output_dir_with_timestamp(
        base_dir: Union[str, Path],
        prefix: str = "run"
) -> Path:
    """
    创建带时间戳的输出目录

    参数:
        base_dir: 基础目录
        prefix: 目录名前缀

    返回:
        创建的目录路径
    """
    timestamp = get_timestamp()
    output_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    ensure_dir(output_dir)
    return output_dir