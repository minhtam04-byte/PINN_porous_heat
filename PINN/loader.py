import pandas as pd
import numpy as np
import os

def process_pressure(p_raw):
    # Các tham số nên để bên trong hàm hoặc truyền vào làm tham số
    U_MAX = 0.10
    CS2 = 1.0 / 3.0
    return (p_raw - CS2) / (U_MAX ** 2)

def load_dataset(Path="", coord_file="coordinates.csv", file_prefix='field40_', start_step=260, end_step=360):
    DT = 0.01
    print(f"Read coordinates data from: {os.path.join(Path, coord_file)}...")
    df_coords = pd.read_csv(os.path.join(Path, coord_file), skipinitialspace=True)

    data_list = []
    for step in range(start_step, end_step + 1):
        file_name = f"{file_prefix}{step:05d}.csv"
        full_path = os.path.join(Path, file_name)

        if not os.path.exists(full_path):
            continue

        df_field = pd.read_csv(full_path, skipinitialspace=True)
        df_step = pd.DataFrame()
        df_step['x'] = df_coords['x']
        df_step['y'] = df_coords['y']
        df_step['t'] = step * DT
        df_step['u'] = df_field['velocity-x']
        df_step['v'] = df_field['velocity-y']
        df_step['p'] = process_pressure(df_field['pressure'])
        data_list.append(df_step)

    if len(data_list) > 0:
        full_df = pd.concat(data_list, ignore_index=True)
        print(f"Loaded {len(full_df)} points.")
        return full_df
    return None