import jax
import jax.numpy as jnp
import jaxopt
import pandas as pd
import numpy as np
import scipy.io as sio
import sys
import os

# 1. THIẾT LẬP ĐƯỜNG DẪN 
PROJECT_PATH = '/content/drive/MyDrive/Colab_Project/D00040'
if PROJECT_PATH not in sys.path:
    sys.path.append(PROJECT_PATH)
    sys.path.append(os.path.join(PROJECT_PATH, 'PINN'))

from loader import load_dataset
from utils import DotDict
from models import PorousInversePINN 

def run_training():
    # --- 1. CẤU HÌNH THAM SỐ ---
    raw_config = {
        'seed': 42, 
        'input_dim': 3, # (t, x, y)
        'arch': {'arch_name': 'ModifiedMlp', 'num_layers': 8, 'hidden_dim': 50, 'out_dim': 3},
        'optim': {'optimizer': 'Adam', 'learning_rate': 1e-3, 'decay_steps': 10000, 'decay_rate': 0.9,
                  'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8, 'grad_accum_steps': 1},
        'weighting': {'scheme': 'ntk', 'init_weights': {'data': 1.0, 'phys': 1.0}, 'momentum': 0.9}
    }
    config = DotDict(raw_config)

    # --- 2. LOAD VÀ CHUẨN BỊ DỮ LIỆU ---
    DATA_PATH = os.path.join(PROJECT_PATH, "CSVdata/")
    print(f"Đang tải dữ liệu từ: {DATA_PATH}...")
    df_pinn = load_dataset(Path=DATA_PATH, start_step=260, end_step=360)
    
    if df_pinn is None:
        print("Lỗi: Không tìm thấy dữ liệu CSV!")
        return

    # Lấy mẫu ngẫu nhiên 20,000 điểm để huấn luyện
    df_train = df_pinn.sample(20000).sort_values('t')

    # Chuẩn bị Batch 
    t_d = jnp.array(df_train['t'].values)[:, None]
    x_d = jnp.array(df_train['x'].values)[:, None]
    y_d = jnp.array(df_train['y'].values)[:, None]
    u_d = jnp.array(df_train['u'].values)[:, None]
    v_d = jnp.array(df_train['v'].values)[:, None]

    batch = {
        'data': (t_d, x_d, y_d, u_d, v_d),
        'eqn': (t_d, x_d, y_d) 
    }

    # Domain
    lb = jnp.array([df_train['t'].min(), -2.5, -2.5])
    ub = jnp.array([df_train['t'].max(), 2.5, 7.5])
    
    print("Khởi tạo mô hình PorousInversePINN...")
    spatial_coords = jnp.array(df_train[['x', 'y']].drop_duplicates().values)
    model = PorousInversePINN(config, lb, ub, spatial_coords)

    # --- 3. GIAI ĐOẠN 1: HUẤN LUYỆN ADAM ---
    n_dev = jax.local_device_count()
    shard = lambda x: x[:(x.shape[0]//n_dev)*n_dev].reshape((n_dev, -1) + x.shape[1:])
    sharded_batch = jax.tree_util.tree_map(shard, batch)

    print(f"Đang chạy Adam trên {n_dev} thiết bị...")
    for step in range(50001):
        # Cập nhật trọng số NTK định kỳ (ví dụ mỗi 1000 bước) để cân bằng Data và Physics loss
        if step % 1000 == 0:
            model.state = model.update_weights(model.state, sharded_batch)
            
        # Bước huấn luyện Adam
        model.state = model.step(model.state, sharded_batch)
        
        if step % 1000 == 0:
            # Lấy tham số và trọng số 
            params_curr = jax.tree_util.tree_map(lambda x: x[0], model.state.params)
            weights_curr = jax.tree_util.tree_map(lambda x: x[0], model.state.weights)
            ls = model.losses(params_curr, batch)
            
            total_loss = ls['data'] * weights_curr['data'] + ls['phys'] * weights_curr['phys']
            print(f"Step {step:5d} | Total Loss: {total_loss:.6e} | Data: {ls['data']:.4e} | Phys: {ls['phys']:.4e}")

    # --- 4. GIAI ĐOẠN 2: TỐI ƯU HÓA L-BFGS ---
    print("\nĐang chạy L-BFGS để hội tụ sâu...")
    params_adam = jax.tree_util.tree_map(lambda x: x[0], model.state.params)
    weights_final = jax.tree_util.tree_map(lambda x: x[0], model.state.weights)
    
    # Định nghĩa hàm 
    def objective_fn(p, b):
        l = model.losses(p, b)
        return l['data'] * weights_final['data'] + l['phys'] * weights_final['phys']

    lbfgs = jaxopt.LBFGS(fun=objective_fn, maxiter=50000, tol=1e-12)
    lbfgs_result = lbfgs.run(params_adam, b=batch)
    print(f"Final Loss: {objective_fn(lbfgs_result.params, batch):.6e}")

    # --- 5. DỰ ĐOÁN VÀ LƯU KẾT QUẢ ---
    print("\nĐang dự đoán toàn bộ trường dòng chảy...")
    t_all = jnp.array(df_pinn['t'].values)[:, None]
    x_all = jnp.array(df_pinn['x'].values)[:, None]
    y_all = jnp.array(df_pinn['y'].values)[:, None]

    # Dự đoán bằng tham số tốt nhất từ L-BFGS
    u_pred, v_pred, p_pred = model.u_net(lbfgs_result.params, x_all, y_all, t_all)
    
    # Khử nhiễu áp suất
    p_pred = p_pred - jnp.mean(p_pred)

    result_file = os.path.join(PROJECT_PATH, 'Porous_Result.mat')
    sio.savemat(result_file, {
        'x': np.array(x_all),
        'y': np.array(y_all),
        't': np.array(t_all),
        'u_pred': np.array(u_pred),
        'v_pred': np.array(v_pred),
        'p_pred': np.array(p_pred),
        'Re': model.Re,
        'Da': model.Da
    })

    print(f"Đã lưu kết quả tại: {result_file}")
    return model, lbfgs_result.params

if __name__ == "__main__":
    run_training()

