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
        X = jnp.concatenate([t, x, y], axis=-1)
        X_norm = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        outputs = self.state.apply_fn(params, X_norm)
        return outputs[..., 0], outputs[..., 1], outputs[..., 2]

    def r_net(self, params, x, y, t):
        def get_u(x, y, t): return self.u_net(params, x, y, t)[0].reshape(())
        def get_v(x, y, t): return self.u_net(params, x, y, t)[1].reshape(())
        def get_p(x, y, t): return self.u_net(params, x, y, t)[2].reshape(())

        # Đạo hàm bậc 1
        u_t = grad(get_u, argnums=2)(x, y, t)
        u_x = grad(get_u, argnums=0)(x, y, t)
        u_y = grad(get_u, argnums=1)(x, y, t)
        v_t = grad(get_v, argnums=2)(x, y, t)
        v_x = grad(get_v, argnums=0)(x, y, t)
        v_y = grad(get_v, argnums=1)(x, y, t)
        p_x = grad(get_p, argnums=0)(x, y, t)
        p_y = grad(get_p, argnums=1)(x, y, t)

        # Đạo hàm bậc 2
        u_xx = grad(lambda x,y,t: grad(get_u, argnums=0)(x,y,t))(x, y, t)
        u_yy = grad(lambda x,y,t: grad(get_u, argnums=1)(x,y,t))(x, y, t)
        v_xx = grad(lambda x,y,t: grad(get_v, argnums=0)(x,y,t))(x, y, t)
        v_yy = grad(lambda x,y,t: grad(get_v, argnums=1)(x,y,t))(x, y, t)

        u_val, v_val, _ = self.u_net(params, x, y, t)
        is_porous = jnp.where(x**2 + y**2 <= self.R**2, 1.0, 0.0)
        
        # Phương trình Navier-Stokes + Darcy:
        # f_u = u_t + (u \cdot \nabla) u + p_x - \frac{1}{Re} \nabla^2 u + \frac{1}{Da} u \cdot \mathbb{I}_{porous}
        f_u = u_t + (u_val*u_x + v_val*u_y) + p_x - (1.0/self.Re)*(u_xx + u_yy) + (1.0/self.Da)*u_val*is_porous
        f_v = v_t + (u_val*v_x + v_val*v_y) + p_y - (1.0/self.Re)*(v_xx + v_yy) + (1.0/self.Da)*v_val*is_porous
        f_e = u_x + v_y
        
        return f_u, f_v, f_e

    def losses(self, params, batch):
        t_d, x_d, y_d, u_d, v_d = batch['data']
        t_e, x_e, y_e = batch['eqn']
        
        u_p, v_p, _ = self.u_net(params, x_d, y_d, t_d)
        
        # SỬA LỖI: Squeeze và reshape để đảm bảo kết quả là () thay vì (1,)
        loss_data = jnp.mean(jnp.square(u_p.squeeze() - u_d.squeeze()) + 
                             jnp.square(v_p.squeeze() - v_d.squeeze()))
        
        f_u, f_v, f_e = vmap(self.r_net, (None, 0, 0, 0))(params, 
                                                         x_e.squeeze(), 
                                                         y_e.squeeze(), 
                                                         t_e.squeeze())
        loss_phys = jnp.mean(jnp.square(f_u) + jnp.square(f_v) + jnp.square(f_e))
        return {'data': loss_data.reshape(()), 'phys': loss_phys.reshape(())}

    def compute_diag_ntk(self, params, batch):
        def loss_data(p): return self.losses(p, batch)['data']
        def loss_phys(p): return self.losses(p, batch)['phys']
        
        jac_data = jacrev(loss_data)(params)
        jac_phys = jacrev(loss_phys)(params)
        
        ntk_data = jnp.sum(jnp.square(flatten_pytree(jac_data)))
        ntk_phys = jnp.sum(jnp.square(flatten_pytree(jac_phys)))
        
        return {'data': ntk_data, 'phys': ntk_phys}
