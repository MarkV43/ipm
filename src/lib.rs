use nalgebra::{DMatrix, DVector, Dyn, RawStorage, Scalar, Storage, Vector, stack};
use num_traits::NumAssign;
use std::fmt::Debug;

pub mod alg;

pub trait CostFunction {
    type F;

    fn cost<S>(&self, param: &Vector<Self::F, Dyn, S>) -> Self::F
    where
        Self::F: Debug + Scalar,
        S: RawStorage<Self::F, Dyn> + Debug;

    fn dims(&self) -> usize;
}

pub trait Gradient: CostFunction {
    fn gradient<S>(&self, param: &Vector<Self::F, Dyn, S>) -> DVector<Self::F>
    where
        Self::F: Debug + Scalar,
        S: RawStorage<Self::F, Dyn> + Debug;
}

pub trait Hessian: Gradient {
    fn hessian<S>(&self, param: &Vector<Self::F, Dyn, S>) -> DMatrix<Self::F>
    where
        Self::F: Debug + Scalar,
        S: RawStorage<Self::F, Dyn> + Debug;
}

/// Constrains the problem to $A*x=b$
pub trait LinearConstraints: CostFunction {
    fn mat_a(&self) -> DMatrix<Self::F>
    where
        Self::F: Debug + Scalar;

    fn vec_b(&self) -> DVector<Self::F>
    where
        Self::F: Debug + Scalar;
}

/// Constrains the problem to $f_i(x) \leq 0, i=1,\dots,m$
pub trait ConvexConstraints: Hessian {
    fn number_of_constraints(&self) -> usize;

    fn convex_constraints<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [Self::F])
    where
        S: RawStorage<Self::F, Dyn> + Debug;

    fn convex_gradients<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [DVector<Self::F>])
    where
        S: RawStorage<Self::F, Dyn> + Debug;

    fn convex_hessians<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [DMatrix<Self::F>])
    where
        S: RawStorage<Self::F, Dyn> + Debug;
}

pub trait PrimalDual: CostFunction {
    fn residual<S>(&self, xv: &Vector<Self::F, Dyn, S>) -> DVector<Self::F>
    where
        Self::F: Debug + Scalar + NumAssign,
        S: Storage<Self::F, Dyn> + Debug;
}

impl<T> PrimalDual for T
where
    T: Gradient + LinearConstraints,
{
    fn residual<S>(&self, xv: &Vector<Self::F, Dyn, S>) -> DVector<Self::F>
    where
        Self::F: Debug + Scalar + NumAssign,
        S: Storage<Self::F, Dyn> + Debug,
    {
        let dims = self.dims();
        let mat_a = self.mat_a();
        let vec_b = self.vec_b();

        let x = xv.rows(0, dims);
        let v = xv.rows_range(dims..);

        let grad = self.gradient(&x);

        let r_dual = grad + mat_a.tr_mul(&v);
        let r_primal = mat_a * x - vec_b;

        stack![r_dual; r_primal]
    }
}
