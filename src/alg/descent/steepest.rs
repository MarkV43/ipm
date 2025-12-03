use std::fmt::Debug;

use nalgebra::{ComplexField, DVector, Dyn, Scalar, Storage, Vector};
use num_traits::{Float, FromPrimitive, NumAssign, Zero};

use crate::{
    ConvexConstraints, CostFunction, Hessian, LinearConstraints, PrimalDual,
    alg::{
        descent::{DescentMethod, DescentSolution},
        line_search::LineSearch,
    },
    observer::{SolverObserver, SolverStep},
};

#[derive(Clone, Debug)]
pub struct SteepestDescent<F, L: LineSearch<F = F>> {
    pub(crate) tolerance: F,
    pub(crate) line_search: L,
    pub(crate) max_its: usize,
}

impl<F, L: LineSearch<F = F>> SteepestDescent<F, L> {
    pub fn new(tolerance: F, line_search: L, max_its: usize) -> Result<Self, String>
    where
        F: Float + FromPrimitive,
    {
        if F::zero() >= tolerance {
            return Err("tolerance must be strictly positive".to_owned());
        }

        Ok(Self {
            tolerance,
            line_search,
            max_its,
        })
    }
}

impl<F: Scalar, L: LineSearch<F = F>> DescentMethod for SteepestDescent<F, L> {
    type F = F;

    fn optimize_observe<P, S, O>(
        &self,
        problem: &mut P,
        x0: &Vector<F, Dyn, S>,
        observer: &mut O,
    ) -> Result<DescentSolution<F>, String>
    where
        P: Hessian + LinearConstraints + ConvexConstraints + PrimalDual + CostFunction<F = F>,
        F: Debug
            + Scalar
            + NumAssign
            + ComplexField<RealField = F>
            + PartialOrd
            + Copy
            + FromPrimitive
            + Zero,
        S: Storage<F, Dyn> + Debug,
        O: SolverObserver<F>,
    {
        let tol2 = self.tolerance * self.tolerance;
        let dims = problem.dims();

        let mut x = x0.clone_owned();
        let mut its = 0;
        let mut gradient = DVector::zeros(dims);

        loop {
            problem.gradient(&x, &mut gradient);

            let dx = -&gradient;

            let directional_derivative = gradient.dot(&dx);
            let mut current_cost = P::F::zero();
            problem.cost(&x, &mut current_cost);

            let step = SolverStep::NewtonsPoint {
                primal: &x.clone_owned(),
                dual: &DVector::zeros(0),
                cost: current_cost,
            };

            observer.on_step(step);

            let t = self
                .line_search
                .search(problem, current_cost, directional_derivative, &x, &dx);

            x += &dx * t;

            if dx.norm_squared() <= tol2 || its > self.max_its {
                break;
            }

            its += 1;
        }

        let mut cost = P::F::zero();
        problem.cost(&x, &mut cost);

        Ok(DescentSolution { cost, arg: x })
    }
}
