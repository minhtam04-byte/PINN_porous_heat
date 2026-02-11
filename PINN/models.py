import jax
import jax.numpy as jnp
from jax import grad, vmap, jacrev
from jaxpi.models import PINN
from jaxpi.utils import flatten_pytree

class PorousInversePINN(PINN):
    def __init__(self, config, lb, ub, spatial_coords):
        super().__init__(config)
        self.lb = lb
        self.ub = ub
        self.spatial_coords = spatial_coords
        self.Re = 100.0
        self.Da = 5e-3
        self.R = 0.5

    def u_net(self, params, x, y, t):
        """
        Hàm dự đoán chính. 
        Input: x, y, t có shape (N, 1)
        Output: u, v, p có shape (N, 1)
        """
        X = jnp.concatenate([t, x, y], axis=-1)
        X_norm = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        outputs = self.state.apply_fn(params, X_norm)
        u = outputs[..., 0:1]
        v = outputs[..., 1:2]
        p = outputs[..., 2:3]
        return u, v, p

    def r_net(self, params, x, y, t):
        def get_u(x, y, t):
            u, _, _ = self.u_net(params, x[None], y[None], t[None])
            return u[0, 0] # Trả về scalar

        def get_v(x, y, t):
            _, v, _ = self.u_net(params, x[None], y[None], t[None])
            return v[0, 0] # Trả về scalar

        def get_p(x, y, t):
            _, _, p = self.u_net(params, x[None], y[None], t[None])
            return p[0, 0] # Trả về scalar

        # --- (AD) ---
        u_t = grad(get_u, argnums=2)(x, y, t)
        u_x = grad(get_u, argnums=0)(x, y, t)
        u_y = grad(get_u, argnums=1)(x, y, t)
        
        v_t = grad(get_v, argnums=2)(x, y, t)
        v_x = grad(get_v, argnums=0)(x, y, t)
        v_y = grad(get_v, argnums=1)(x, y, t)
        
        p_x = grad(get_p, argnums=0)(x, y, t)
        p_y = grad(get_p, argnums=1)(x, y, t)

        u_xx = grad(lambda x,y,t: grad(get_u, argnums=0)(x,y,t))(x, y, t)
        u_yy = grad(lambda x,y,t: grad(get_u, argnums=1)(x,y,t))(x, y, t)
        
        v_xx = grad(lambda x,y,t: grad(get_v, argnums=0)(x,y,t))(x, y, t)
        v_yy = grad(lambda x,y,t: grad(get_v, argnums=1)(x,y,t))(x, y, t)

        # Lấy giá trị u, v 
        u_val = get_u(x, y, t)
        v_val = get_v(x, y, t)

        # Định nghĩa vùng xốp 
        is_porous = jnp.where(x**2 + y**2 <= self.R**2, 1.0, 0.0)
        
        # --- Phương trình Navier-Stokes + Darcy (cho vùng xốp) ---
        # Momentum u
        f_u = u_t + (u_val*u_x + v_val*u_y) + p_x - (1.0/self.Re)*(u_xx + u_yy) + (1.0/self.Da)*u_val*is_porous
        # Momentum v
        f_v = v_t + (u_val*v_x + v_val*v_y) + p_y - (1.0/self.Re)*(v_xx + v_yy) + (1.0/self.Da)*v_val*is_porous
        # Continuity
        f_e = u_x + v_y
        
        return f_u, f_v, f_e

    def losses(self, params, batch):
        t_d, x_d, y_d, u_d, v_d = batch['data']
        t_e, x_e, y_e = batch['eqn']
        
        # 1. Loss dữ liệu 
        u_p, v_p, _ = self.u_net(params, x_d, y_d, t_d)
        
        #  MSE 
        loss_data = jnp.mean(jnp.square(u_p - u_d) + jnp.square(v_p - v_d))
        
        # 2. Loss vật lý 
        f_u, f_v, f_e = vmap(self.r_net, (None, 0, 0, 0))(params, 
                                                          x_e.reshape(-1), 
                                                          y_e.reshape(-1), 
                                                          t_e.reshape(-1))
        
        loss_phys = jnp.mean(jnp.square(f_u) + jnp.square(f_v) + jnp.square(f_e))
        return {'data': loss_data, 'phys': loss_phys}

    def compute_diag_ntk(self, params, batch):
        def loss_data(p): return self.losses(p, batch)['data']
        def loss_phys(p): return self.losses(p, batch)['phys']
        
        jac_data = jacrev(loss_data)(params)
        jac_phys = jacrev(loss_phys)(params)
        
        ntk_data = jnp.sum(jnp.square(flatten_pytree(jac_data)))
        ntk_phys = jnp.sum(jnp.square(flatten_pytree(jac_phys)))
        
        return {'data': ntk_data, 'phys': ntk_phys}
