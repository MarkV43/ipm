#![warn(clippy::pedantic)]
#![deny(clippy::perf)]
#![allow(clippy::toplevel_ref_arg, clippy::missing_panics_doc)]

use nalgebra::{
    DMatrix, DVector, Dyn, Matrix, RawStorage, Scalar, Storage, StorageMut, Vector, stack,
};
use num_traits::NumAssign;
use std::fmt::Debug;

pub mod alg;

pub trait CostFunction
where
    Self::F: Scalar,
{
    type F;

    fn cost<S>(&mut self, param: &Vector<Self::F, Dyn, S>, out: &mut Self::F)
    where
        S: RawStorage<Self::F, Dyn> + Debug;

    fn dims(&self) -> usize;
}

pub trait Gradient: CostFunction
where
    Self::F: Scalar,
{
    fn gradient<S1, S2>(
        &mut self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut Vector<Self::F, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug;
}

pub trait Hessian: Gradient
where
    Self::F: Scalar,
{
    fn hessian<S1, S2>(
        &mut self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut Matrix<Self::F, Dyn, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug;
}

/// Constrains the problem to $A*x=b$
pub trait LinearConstraints: CostFunction
where
    Self::F: Scalar,
{
    fn mat_a(&self) -> DMatrix<Self::F>;
    fn vec_b(&self) -> DVector<Self::F>;
}

/// Constrains the problem to `f_i(x) \leq 0, i=1,\dots,m`
pub trait ConvexConstraints: Hessian {
    fn number_of_constraints(&self) -> usize;

    fn convex_constraints<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [Self::F])
    where
        S: RawStorage<Self::F, Dyn> + Debug;

    fn convex_gradients<S1, S2>(
        &self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut [Vector<Self::F, Dyn, S2>],
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug;

    fn convex_hessians<S1, S2>(
        &self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut [Matrix<Self::F, Dyn, Dyn, S2>],
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug;
}

pub trait PrimalDual: Gradient + LinearConstraints
where
    Self::F: Scalar + NumAssign,
{
    fn residual<S1, S2>(
        &mut self,
        xv: &Vector<Self::F, Dyn, S1>,
        out: &mut Vector<Self::F, Dyn, S2>,
    ) where
        S1: Storage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug;
}

impl<T> PrimalDual for T
where
    Self::F: Scalar + NumAssign,
    T: Gradient + LinearConstraints,
{
    fn residual<S1, S2>(
        &mut self,
        xv: &Vector<Self::F, Dyn, S1>,
        out: &mut Vector<Self::F, Dyn, S2>,
    ) where
        S1: Storage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug,
    {
        let dims = self.dims();
        let mat_a = self.mat_a();
        let vec_b = self.vec_b();

        let x = xv.rows(0, dims);
        let v = xv.rows_range(dims..);

        let mut grad = DVector::zeros(dims);
        self.gradient(&x, &mut grad);

        let r_dual = grad + mat_a.tr_mul(&v);
        let r_primal = mat_a * x - vec_b;

        out.copy_from(&stack![r_dual; r_primal]);
    }
}
