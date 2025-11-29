#![warn(clippy::pedantic)]
#![deny(clippy::perf)]
#![allow(
    clippy::toplevel_ref_arg,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]

use nalgebra::{
    DMatrix, DVector, Dyn, Matrix, RawStorage, Scalar, Storage, StorageMut, Vector, stack,
};
use num_traits::NumAssign;
use std::fmt::Debug;

pub mod alg;
pub mod observer;

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
    fn num_linear_constraints(&self) -> usize;
    fn mat_a<S>(&self, out: &mut Matrix<Self::F, Dyn, Dyn, S>)
    where
        S: StorageMut<Self::F, Dyn, Dyn>;
    fn vec_b<S>(&self, out: &mut Vector<Self::F, Dyn, S>)
    where
        S: StorageMut<Self::F, Dyn>;
}

/// Constrains the problem to `f_i(x) \leq 0, i=1,\dots,m`
pub trait ConvexConstraints: Hessian {
    fn num_convex_constraints(&self) -> usize;

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
        let nconstr = self.num_linear_constraints();

        let mut mat_a = DMatrix::zeros(nconstr, dims);
        let mut vec_b = DVector::zeros(nconstr);

        self.mat_a(&mut mat_a);
        self.vec_b(&mut vec_b);

        let x = xv.rows(0, dims);
        let v = xv.rows_range(dims..);

        let mut grad = DVector::zeros(dims);
        self.gradient(&x, &mut grad);

        let r_dual = grad + mat_a.tr_mul(&v);
        let r_primal = mat_a * x - vec_b;

        out.copy_from(&stack![r_dual; r_primal]);
    }
}
