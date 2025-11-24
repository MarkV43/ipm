use std::{fmt::Debug, time::Instant};

use ipm::{
    ConvexConstraints, CostFunction, Gradient, Hessian, LinearConstraints,
    alg::{
        barrier::{BarrierParams, barrier_method_infeasible},
        line_search::LineSearchParams,
        newton::NewtonParams,
    },
};
use nalgebra::{DVector, Dyn, Matrix, RawStorage, StorageMut, Vector};

struct Question6;

impl CostFunction for Question6 {
    type F = f64;

    fn cost<S>(&mut self, param: &Vector<Self::F, Dyn, S>, out: &mut Self::F)
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        *out = -3.0 * param[0] - 2.0 * param[1];
    }

    #[inline(always)]
    fn dims(&self) -> usize {
        2
    }
}

impl Gradient for Question6 {
    fn gradient<S1, S2>(
        &mut self,
        _param: &Vector<Self::F, Dyn, S1>,
        out: &mut Vector<Self::F, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug,
    {
        out[0] = -3.0;
        out[1] = -2.0;
    }
}

impl Hessian for Question6 {
    fn hessian<S1, S2>(
        &mut self,
        _param: &Vector<Self::F, Dyn, S1>,
        _out: &mut Matrix<Self::F, Dyn, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug,
    {
        // do nothing, `out` is already filled with zeros
    }
}

impl LinearConstraints for Question6 {
    #[inline(always)]
    fn num_linear_constraints(&self) -> usize {
        0
    }

    fn mat_a<S>(&self, _out: &mut Matrix<Self::F, Dyn, Dyn, S>)
    where
        S: StorageMut<Self::F, Dyn, Dyn>,
    {
        // Keep zeros
    }

    fn vec_b<S>(&self, _out: &mut Vector<Self::F, Dyn, S>)
    where
        S: StorageMut<Self::F, Dyn>,
    {
        // Keep zeros
    }
}

impl ConvexConstraints for Question6 {
    #[inline(always)]
    fn num_convex_constraints(&self) -> usize {
        10
    }

    fn convex_constraints<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [Self::F])
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let x1 = param[0];
        let x2 = param[1];

        out[0] = -1.0 + x1 - 2.0 * x2;
        out[1] = -2.0 + x1 - x2;
        out[2] = -6.0 + 2.0 * x1 - x2;
        out[3] = -5.0 + x1;
        out[4] = -16.0 + 2.0 * x1 + x2;
        out[5] = -12.0 + x1 + x2;
        out[6] = -21.0 + x1 + 2.0 * x2;
        out[7] = -10.0 + x2;
        out[8] = -x1;
        out[9] = -x2;
    }

    fn convex_gradients<S1, S2>(
        &self,
        _param: &Vector<Self::F, Dyn, S1>,
        out: &mut [Vector<Self::F, Dyn, S2>],
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug,
    {
        out[0].rows_mut(0, 2).copy_from_slice(&[1.0, -2.0]);
        out[1].rows_mut(0, 2).copy_from_slice(&[1.0, -1.0]);
        out[2].rows_mut(0, 2).copy_from_slice(&[2.0, -1.0]);
        out[3].rows_mut(0, 2).copy_from_slice(&[1.0, 0.0]);
        out[4].rows_mut(0, 2).copy_from_slice(&[2.0, 1.0]);
        out[5].rows_mut(0, 2).copy_from_slice(&[1.0, 1.0]);
        out[6].rows_mut(0, 2).copy_from_slice(&[1.0, 2.0]);
        out[7].rows_mut(0, 2).copy_from_slice(&[0.0, 1.0]);
        out[8].rows_mut(0, 2).copy_from_slice(&[-1.0, 0.0]);
        out[9].rows_mut(0, 2).copy_from_slice(&[0.0, -1.0]);
    }

    fn convex_hessians<S1, S2>(
        &self,
        _param: &Vector<Self::F, Dyn, S1>,
        _out: &mut [Matrix<Self::F, Dyn, Dyn, S2>],
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug,
    {
        // Do nothing, just zeros
    }
}

fn main() {
    let mut q6 = Question6;

    let lparams = LineSearchParams::new(0.3, 0.7);
    let nparams = NewtonParams::new(1e-5, lparams, 10, 100);
    let bparams = BarrierParams::new(0.1, 10.0, 1e-3, nparams);

    let x0 = DVector::from_vec(vec![1.0, 1.0]);

    let t0 = Instant::now();

    let sol = barrier_method_infeasible(&mut q6, &x0, &bparams);

    let dur = t0.elapsed();

    println!("Solution: {}", sol.arg);
    println!("Cost: {}", sol.cost);
    println!("Elapsed: {dur:?}");
}
