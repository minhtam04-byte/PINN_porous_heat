import jax
import jax.numpy as jnp
import jaxopt
import pandas as pd
import numpy as np
import scipy.io as sio
import sys
import os

# 1. Thêm đường dẫn dự án
project_path = '/content/drive/MyDrive/Colab_Project/D00040'
if project_path not in sys.path:
    sys.path.append(project_path)

from loader import load_dataset
from utils import DotDict
from models import PorousInversePINN 

def run_training():
    # --- 1. CẤU HÌNH ---
    raw_config = {
        'seed': 42, 
        'input_dim': 3, # t, x, y
        'arch': {
            'arch_name': 'ModifiedMlp', 
            'num_layers': 8, 
            'hidden_dim': 50, 
            'out_dim': 3
        },
        'optim': {
            'optimizer': 'Adam', 
            'learning_rate': 1e-3, 
            'decay_steps': 10000, 
            'decay_rate': 0.9,
            'beta1': 0.9,      
            'beta2': 0.999,   
            'eps': 1e-8,       
            'grad_accum_steps': 1 
        },
        'weighting': {
            'scheme': 'ntk', 
            'init_weights': {'data': 1.0, 'phys': 1.0}, 
            'momentum': 0.9
        }
    }
    config = DotDict(raw_config)

    # --- 2. LOAD VÀ CHUẨN BỊ DỮ LIỆU ---
    PATH = f"{project_path}/CSVdata/"
    df_pinn = load_dataset(Path=PATH, start_step=260, end_step=360)
    
    if df_pinn is None:
        print("Không tìm thấy dữ liệu tại đường dẫn cung cấp!")
        return

    # Lấy mẫu huấn luyện
    df_train = df_pinn.sample(20000).sort_values('t')

    # --- 3. CHUẨN BỊ BATCH & DOMAIN ---
    batch = {
        'data': (jnp.array(df_train['t'].values)[:,None], 
                 jnp.array(df_train['x'].values)[:,None], 
                 jnp.array(df_train['y'].values)[:,None], 
                 jnp.array(df_train['u'].values)[:,None], 
                 jnp.array(df_train['v'].values)[:,None]),
        'eqn': (jnp.array(df_train['t'].values)[:,None], 
                jnp.array(df_train['x'].values)[:,None], 
                jnp.array(df_train['y'].values)[:,None])
    }
    
    lb = jnp.array([df_train['t'].min(), -2.5, -2.5])
    ub = jnp.array([df_train['t'].max(), 2.5, 7.5])
    
    # --- 4. KHỞI TẠO MODEL ---
    spatial_coords = jnp.array(df_train[['x', 'y']].drop_duplicates().values)
    model = PorousInversePINN(config, lb, ub, spatial_coords)

    # --- 5. HUẤN LUYỆN ADAM ---
    n_dev = jax.local_device_count()
    shard = lambda x: x[:(x.shape[0]//n_dev)*n_dev].reshape((n_dev, -1) + x.shape[1:])
    sharded_batch = jax.tree_util.tree_map(shard, batch)

    print(f"Running Adam on {n_dev} device(s)...")
    for step in range(50001):
        model.state = model.step(model.state, sharded_batch)
        if step % 1000 == 0:
            p_curr = jax.tree_util.tree_map(lambda x: x[0], model.state.params)
            ls = model.losses(p_curr, batch)
            print(f"Step {step:5d} | Loss: {ls['data'] + ls['phys']:.6e}")

    # --- 6. HUẤN LUYỆN L-BFGS ---
    print("\n Running L-BFGS...")
    params_adam = jax.tree_util.tree_map(lambda x: x[0], model.state.params)
    
    def obj_fn(p, b):
        l = model.losses(p, b)
        w = jax.tree_util.tree_map(lambda x: x[0], model.state.weights)
        return l['data']*w['data'] + l['phys']*w['phys']

    lbfgs = jaxopt.LBFGS(fun=obj_fn, maxiter=50000, tol=1e-12)
    lbfgs_res = lbfgs.run(params_adam, batch=batch)

    # --- 7. DỰ ĐOÁN & LƯU KẾT QUẢ ---
    print("\n Saving results...")
    t_all = jnp.array(df_pinn['t'].values)[:, None]
    x_all = jnp.array(df_pinn['x'].values)[:, None]
    y_all = jnp.array(df_pinn['y'].values)[:, None]
    
    u_p, v_p, p_p = model.u_net(lbfgs_res.params, x_all, y_all, t_all)
    
    result_path = f"{project_path}/Porous_Result.mat"
    sio.savemat(result_path, {
        'x': np.array(x_all), 
        'y': np.array(y_all), 
        't': np.array(t_all),
        'u_pred': np.array(u_p), 
        'v_pred': np.array(v_p), 
        'p_pred': np.array(p_p - jnp.mean(p_p))
    })
    print(f"Xong! Kết quả lưu tại: {result_path}")

if __name__ == "__main__":
    run_training()