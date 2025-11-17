use std::{f32, fmt::Debug, time::Instant};

use ipm::{
    alg::barrier::{BarrierProblem, barrier_method},
    *,
};
use nalgebra::{
    Const, DMatrix, DVector, Dyn, OMatrix, OVector, RawStorage, Scalar, Vector, Vector2, dvector,
};
use num_traits::{Num, One};

struct LinearDiscrimination {
    xs: Vec<Vector2<f32>>,
    ys: Vec<Vector2<f32>>,
    gamma: f32,
}

impl CostFunction for LinearDiscrimination {
    type F = f32;

    fn cost<S>(&mut self, param: &Vector<Self::F, Dyn, S>, out: &mut Self::F)
    where
        Self::F: Debug + Num + Scalar,
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let a = param.rows(0, 2);
        // let b = param.rows(2, 1);
        let u = param.rows(3, self.xs.len());
        let v = param.rows(3 + self.xs.len(), self.ys.len());

        *out = a.norm() + self.gamma * (u.sum() + v.sum())
    }

    fn dims(&self) -> usize {
        self.xs.len() + self.ys.len() + 3
    }
}

impl Gradient for LinearDiscrimination {
    fn gradient<S>(&mut self, param: &Vector<Self::F, Dyn, S>, out: &mut OVector<Self::F, Dyn>)
    where
        Self::F: Debug + Num + Scalar,
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        out.fill(0.0);

        // layout: a (2) | b (1) | u (nx) | v (ny)
        let a = param.rows(0, 2).into_owned();
        let nx = self.xs.len();
        let ny = self.ys.len();

        // gradient wrt a: a / ||a||  (choose 0 when ||a|| == 0)
        let norm_a = a.norm();
        if norm_a != 0.0 {
            let grad_a = &a / norm_a;
            out.rows_mut(0, 2).copy_from(&grad_a);
        } // else leave zeros (subgradient choice)

        // b has no contribution -> gradient 0 (already zero)

        // u and v: gamma * ones
        let start_u = 3;
        let start_v = 3 + nx;
        if nx > 0 {
            let ones_u = OVector::<Self::F, Dyn>::from_element(nx, self.gamma);
            out.rows_mut(start_u, nx).copy_from(&ones_u);
        }
        if ny > 0 {
            let ones_v = OVector::<Self::F, Dyn>::from_element(ny, self.gamma);
            out.rows_mut(start_v, ny).copy_from(&ones_v);
        }
    }
}

impl Hessian for LinearDiscrimination {
    fn hessian<S>(&mut self, _param: &Vector<Self::F, Dyn, S>, out: &mut OMatrix<Self::F, Dyn, Dyn>)
    where
        Self::F: Debug + Num + Scalar,
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        out.fill(0.0);

        // Only a-block (top-left 2x2) is nonzero:
        // H_a = (I / ||a||) - (a a^T / ||a||^3)
        let a = _param.rows(0, 2).into_owned();
        let norm_a = a.norm();

        if norm_a != 0.0 {
            // build 2x2 identity and aa^T
            let i2 = OMatrix::<Self::F, Dyn, Dyn>::identity(2, 2);
            let aa_t = &a * a.transpose(); // 2x2

            let ha = (&i2) / norm_a - (&aa_t) / (norm_a * norm_a * norm_a);

            // copy ha into top-left block of h
            let mut block = out.view_mut((0, 0), (2, 2));
            block.copy_from(&ha);
        }
        // all other second derivatives are zero because cost is linear in b,u,v and uses only ||a|| for quadratic part.
    }
}

impl ConvexConstraints for LinearDiscrimination {
    fn number_of_constraints(&self) -> usize {
        2 * (self.xs.len() + self.ys.len())
    }

    fn convex_constraints<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [Self::F])
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let n = self.xs.len();
        let m = self.ys.len();

        let a = param.rows_generic(0, Const::<2>);
        let b = param.rows_generic(2, Const::<1>);
        let u = param.rows(3, n);
        let v = param.rows(3 + n, m);

        for i in 0..n {
            out[i] = b[0] - a.dot(&self.xs[i]) - u[i] + Self::F::one();
        }
        for i in 0..m {
            out[n + i] = a.dot(&self.ys[i]) - b[0] - v[i] + Self::F::one();
        }
        for i in 0..n {
            out[n + m + i] = -u[i];
        }
        for i in 0..m {
            out[n + m + n + i] = -v[i];
        }
    }

    fn convex_gradients<S>(&self, _param: &Vector<Self::F, Dyn, S>, out: &mut [DVector<Self::F>])
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let n = self.xs.len();
        let m = self.ys.len();

        // b[0] - a.dot(x_i) - u_i + 1
        for i in 0..n {
            let g = &mut out[i];
            g.rows_generic_mut(0, Const::<2>).copy_from(&-self.xs[i]);
            g[2] = Self::F::one();
            g[3 + i] = -Self::F::one();
        }

        // a.dot(y_i) - b[0] - v_i + 1
        for i in 0..m {
            let g = &mut out[n + i];
            g.rows_generic_mut(0, Const::<2>).copy_from(&self.ys[i]);
            g[2] = -Self::F::one();
            g[3 + n + i] = -Self::F::one();
        }

        // -u_i
        for i in 0..n {
            let g = &mut out[n + m + i];
            g.fill(0.0);
            g[3 + i] = -Self::F::one();
        }

        // -v_i
        for i in 0..m {
            let g = &mut out[n + m + n + i];
            g.fill(0.0);
            g[3 + n + i] = -Self::F::one();
        }
    }

    fn convex_hessians<S>(&self, _param: &Vector<Self::F, Dyn, S>, out: &mut [DMatrix<Self::F>])
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        out.iter_mut().for_each(|x| x.fill(0.0));
    }
}

impl LinearConstraints for LinearDiscrimination {
    fn mat_a(&self) -> DMatrix<Self::F>
    where
        Self::F: std::fmt::Debug + nalgebra::Scalar,
    {
        DMatrix::zeros(0, self.dims())
    }

    fn vec_b(&self) -> DVector<Self::F>
    where
        Self::F: std::fmt::Debug + nalgebra::Scalar,
    {
        DVector::zeros(0)
    }
}

fn main() {
    let x = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(1.0, 0.0),
        Vector2::new(0.5, 0.5),
        Vector2::new(0.0, 0.5),
        Vector2::new(0.1, 0.5),
        Vector2::new(0.2, 0.5),
        Vector2::new(0.3, 0.5),
        Vector2::new(0.4, 0.5),
    ];
    let y = vec![
        Vector2::new(0.0, 1.0),
        Vector2::new(1.0, 1.0),
        Vector2::new(0.1, 1.5),
        Vector2::new(0.2, 1.5),
        Vector2::new(0.3, 1.5),
        Vector2::new(0.4, 1.5),
    ];

    let mut x0 = DVector::zeros(3 + x.len() + y.len());
    x0.rows_mut(0, 2).fill(1.0);
    x0.rows_range_mut(3..).fill(4.0);

    println!("{x0}");

    let start = Instant::now();

    let mut disc = LinearDiscrimination {
        xs: x,
        ys: y,
        gamma: 1.0,
    };

    let mut constraints = vec![0.0; disc.number_of_constraints()];
    disc.convex_constraints(&x0, &mut constraints);

    println!("{constraints:?}");

    // let sol = newtons_method(&disc, &x0, 1e-20, 0.3, 0.8);
    let sol = barrier_method(&mut disc, &x0, 0.1, 5.0, 1e-5, 1e-3, 0.3, 0.8);

    let dur = start.elapsed();

    println!("Elapsed: {dur:?}");

    // println!("{sol:#?}");
}
