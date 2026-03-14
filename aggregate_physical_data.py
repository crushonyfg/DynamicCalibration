"""
aggregate_physical_data.py

将 PhysicalData_v3 目录下的多个 Excel 文件整理成单个 CSV 文件。

用法:
    python aggregate_physical_data.py --data_dir <PhysicalData_v3路径> --output <输出CSV路径>

示例:
    python aggregate_physical_data.py \
        --data_dir "C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3" \
        --output "C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/physical_data.csv"
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_mmss_to_seconds(val) -> float:
    """
    Parse CustomerLbd formatted as "min:sec(.fraction)" into SECONDS (float).
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)

    s = str(val).strip()
    if ":" not in s:
        try:
            return float(s)
        except Exception:
            return np.nan

    m_str, sec_str = s.split(":", 1)
    try:
        m = int(float(m_str))
        sec = float(sec_str)
        return float(m * 60.0 + sec)
    except Exception:
        return np.nan


def aggregate_physical_data(data_dir: str, output_path: str):
    """
    聚合 PhysicalData_v3 目录下的所有 factory_Mode{mode}t{t}.xlsx 文件。
    
    输出 CSV 包含列: t, mode, W, R, M1, M2, Q, NetRevenue, CustomerLbd_sec, CustomerLbd_min
    """
    all_rows = []
    
    for mode in [0, 1]:
        pattern = os.path.join(data_dir, f"factory_Mode{mode}t*.xlsx")
        files = glob.glob(pattern)
        
        if not files:
            print(f"Warning: No files matched pattern: {pattern}")
            continue
        
        rx = re.compile(rf"factory_Mode{mode}t(\d+)\.xlsx$", re.IGNORECASE)
        
        for fp in tqdm(files, desc=f"Processing mode={mode}"):
            m = rx.search(os.path.basename(fp))
            if not m:
                continue
            
            t = int(m.group(1))
            
            try:
                df = pd.read_excel(fp)
            except Exception as e:
                print(f"Error reading {fp}: {e}")
                continue
            
            if df.shape[0] < 1:
                continue
            
            df.columns = [str(c).strip() for c in df.columns]
            required = ["CustomerLbd", "NetRevenue", "W", "R", "M1", "M2", "Q"]
            miss = [c for c in required if c not in df.columns]
            if miss:
                print(f"Missing columns {miss} in {fp}")
                continue
            
            r0 = df.iloc[0]
            theta_sec = parse_mmss_to_seconds(r0["CustomerLbd"])
            y = float(pd.to_numeric(r0["NetRevenue"], errors="coerce"))
            
            W  = int(pd.to_numeric(r0["W"],  errors="coerce"))
            R  = int(pd.to_numeric(r0["R"],  errors="coerce"))
            M1 = int(pd.to_numeric(r0["M1"], errors="coerce"))
            M2 = int(pd.to_numeric(r0["M2"], errors="coerce"))
            Q  = int(pd.to_numeric(r0["Q"],  errors="coerce"))
            
            if np.isnan(theta_sec) or np.isnan(y):
                continue
            
            all_rows.append({
                "t": t,
                "mode": mode,
                "W": W,
                "R": R,
                "M1": M1,
                "M2": M2,
                "Q": Q,
                "NetRevenue": y,
                "CustomerLbd_sec": theta_sec,
                "CustomerLbd_min": theta_sec / 60.0,
            })
    
    if not all_rows:
        raise ValueError("No valid data found!")
    
    result_df = pd.DataFrame(all_rows)
    result_df = result_df.sort_values(by=["mode", "t"]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    print(f"\nDone!")
    print(f"Total rows: {len(result_df)}")
    print(f"Mode 0 rows: {(result_df['mode'] == 0).sum()}")
    print(f"Mode 1 rows: {(result_df['mode'] == 1).sum()}")
    print(f"Saved to: {output_path}")
    
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate PhysicalData_v3 Excel files into a single CSV")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="C:/Users/yxu59/files/winter2026/park/simulation/PhysicalData_v3",
        help="Path to PhysicalData_v3 directory"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="C:/Users/yxu59/files/autumn2025/park/DynamicCalibration/physical_data.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()
    
    aggregate_physical_data(args.data_dir, args.output)
