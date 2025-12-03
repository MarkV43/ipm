use std::{fmt::Debug, iter::Sum};

use nalgebra::{ComplexField, Dyn, OVector, Scalar, Storage, Vector};
use num_traits::{Float, Inv, NumAssign};

use crate::{ConvexConstraints, CostFunction, Hessian, PrimalDual, observer::SolverObserver};

pub mod barrier;
pub mod infeasible;

pub struct IpmSolution<F: Scalar> {
    pub arg: OVector<F, Dyn>,
    pub cost: F,
}

pub trait InteriorPointMethod {
    type F: Scalar;

    fn optimize_observe<P, S, O>(
        &self,
        problem: &mut P,
        x0: &Vector<P::F, Dyn, S>,
        observer: &mut O,
    ) -> Result<IpmSolution<P::F>, String>
    where
        P: Hessian + ConvexConstraints + PrimalDual + CostFunction<F = Self::F>,
        P::F:
            Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
        S: Storage<P::F, Dyn> + Debug,
        O: SolverObserver<P::F>;

    fn optimize<P, S>(
        &self,
        problem: &mut P,
        x0: &Vector<P::F, Dyn, S>,
    ) -> Result<IpmSolution<P::F>, String>
    where
        P: Hessian + ConvexConstraints + PrimalDual + CostFunction<F = Self::F>,
        P::F:
            Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
        S: Storage<P::F, Dyn> + Debug,
    {
        self.optimize_observe(problem, x0, &mut ())
    }
}
