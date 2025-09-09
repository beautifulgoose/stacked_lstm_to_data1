import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import List, Dict, Tuple, Any

from config import csv_path, ACCEL_GYRO_RATE
from data_processing import lowpass_filter, window_process

REQUIRED_COLS = ["AccX","AccY","AccZ","GyroX","GyroY","GyroZ","Class","Gesture","User"]
# 约定：CSV 中包含列：
# ['AccX','AccY','AccZ','GyroX','GyroY','GyroZ','Class','Gesture','User']
SENSOR_COLS = ['AccX','AccY','AccZ','GyroX','GyroY','GyroZ']

# ------------------------- 加载和数据处理模块 -------------------------
def load_csv(path: str) -> pd.DataFrame:
    """
    读取 CSV，做基础校验与清洗：
    - 确保必须列存在
    - 去除全空行
    - 保留原始顺序（按原始行号排序）
    """
    df = pd.read_csv(path)
    # 统一去掉列名两端空白
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}（需要列：{REQUIRED_COLS}）")

    # 删除全空行
    df = df.dropna(how="all").copy()

    # 保留原始顺序（如果原 CSV 没有时间戳，就按出现顺序视为时间）
    df["_order"] = range(len(df))
    df = df.sort_values("_order").reset_index(drop=True)
    return df

def extract_load_actions(df: pd.DataFrame) -> Dict[Tuple[int,int,int], pd.DataFrame]:
    """
    将 DataFrame 按 (User, Class, Gesture) 分组，返回每个动作的完整序列。
    返回值是字典：key=(User, Class, Gesture)，value=该段序列的 DataFrame。
    """
    # 只保留关心的列的顺序
    keep_cols = REQUIRED_COLS
    df = df[keep_cols + (["_order"] if "_order" in df.columns else [])].copy()

    actions: Dict[Tuple[int,int,int], pd.DataFrame] = {}
    for (user, klass, gesture), g in df.groupby(["User","Class","Gesture"], sort=False):
        # 保证时间顺序
        if "_order" in g.columns:
            g = g.sort_values("_order").drop(columns=["_order"])
        actions[(int(user), int(klass), int(gesture))] = g.reset_index(drop=True)
        print("now load user:"+str(user)+" class:"+str(klass)+" gesture:"+str(gesture))
    return actions

# 数据处理process
def process_data(aligned_imu):
    """
        aligned_imu: List[np.ndarray]，每个元素形状 (T, 6)，列顺序为
                     [AccX, AccY, AccZ, GyroX, GyroY, GyroZ]
        返回：windows_imu（与你现有 window_process 的返回保持一致）
        """
    windows_imu = []
    for i, sample in enumerate(aligned_imu):
        # 基本校验
        if not isinstance(sample, np.ndarray) or sample.ndim != 2 or sample.shape[1] != 6:
            raise ValueError(
                f"aligned_imu[{i}] 期望为 shape=(T,6) 的 ndarray，但实际是 {type(sample)}，shape={getattr(sample, 'shape', None)}")

        # 确保浮点类型
        sample = sample.astype(np.float32, copy=False)

        # 拆分为加速度和陀螺仪三轴
        accel = sample[:, :3]  # (T,3)
        gyro = sample[:, 3:]  # (T,3)

        # ==================== ★ 3. 低通滤波 (2-25 Hz) ★ ====================
        accel_f = lowpass_filter(accel)
        gyro_f = lowpass_filter(gyro)

        # 仍按你原先 window_process 的入参格式组织
        sample_dict = {
            'accel': accel_f,  # (T,3)
            'gyro': gyro_f  # (T,3)
        }

        window_imu= window_process(sample_dict)
        windows_imu.append(window_imu)

    return windows_imu
# -------------------------------------------------------------------

# ----------------------建立五折交叉验证训练测试集------------------------
def build_kfold_datasets(n_splits=5):
    """
    返回 datasets, selected_categories
    datasets: List[dict]，长度 = n_splits
              每个元素包含 'train' 和 'val' 两个子字典
    """
    # 读取与分割单个动作
    # df为完整的数据
    df = load_csv(csv_path)
    # actions为字典，每个元素代表一个独立的动作。
    # key=Tuple(User, Class, Gesture)，value=该段序列的 DataFrame （大小为500行，9列）
    actions: Dict[Tuple[Any, Any, Any], Any] = extract_load_actions(df)

    # 统计集合
    unique_users = sorted({k[0] for k in actions.keys()})
    unique_gestures = sorted({k[2] for k in actions.keys()})  # Gesture 更贴近“类别”概念
    selected_categories = list(unique_gestures)

    # ============= 按“用户标签”构建 K 折 =============
    if len(unique_users) < n_splits:
        # 用户数少于折数时，降级为留一法或按用户数设置折数
        n_splits = max(2, len(unique_users))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    datasets = []

    def pack_from_users(user_set: set):
        """从给定用户集合中提取序列与标签，直接由 actions 聚合"""
        X_list, y_list, y_user_list, len_list = [], [], [], []
        for (user, cls, gest), seg_df in actions.items():
            if user in user_set:
                # X：仅取 6 维 IMU 传感器特征
                X_seg = seg_df[SENSOR_COLS].to_numpy(copy=True)  # (T, 6)，通常 T=500
                T = X_seg.shape[0]

                X_list.append(X_seg)
                # 这里 y 作为主任务（类别）标签：你也可以替换为 gest（若你的主任务用 Gesture）
                y_list.append(int(seg_df['Class'].iloc[0]))
                y_user_list.append(str(seg_df['User'].iloc[0]))
                len_list.append(int(T))
        return X_list, y_list, y_user_list, len_list

    # 逐折划分
    user_idxs = np.arange(len(unique_users))
    for fold, (train_idx, val_idx) in enumerate(kf.split(user_idxs), 1):
        print(f"\n===== Fold {fold}/{n_splits} =====")
        print(f"\n loading......")
        train_users = {unique_users[i] for i in train_idx}
        val_users = {unique_users[i] for i in val_idx}

        # 组装训练/验证集
        train_X, train_y, train_y_user, train_len = pack_from_users(train_users)
        val_X, val_y, val_y_user, val_len = pack_from_users(val_users)

        #数据处理(针对train_X和val_X)
        train_X_imu = process_data(train_X)
        val_X_imu = process_data(val_X)

        datasets.append({
            'train': {
                'X_imu': train_X_imu,
                'y': train_y,
            },
            'val': {
                'X_imu': val_X_imu,
                'y': val_y,
            }
        })
    return datasets, selected_categories
# -------------------------------------------------------------------