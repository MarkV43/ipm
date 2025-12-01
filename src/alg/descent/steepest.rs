use std::fmt::Debug;

use nalgebra::{ComplexField, DVector, Dyn, OVector, Scalar, Storage, Vector};
use num_traits::{FromPrimitive, Num, NumAssign, Zero};

use crate::{
    ConvexConstraints, Gradient, LinearConstraints,
    alg::line_search::{
        LineSearch, backtrack::backtrack_line_search, guarded::guarded_line_search,
    },
    observer::{SolverObserver, SolverStep},
};

#[derive(Debug, Clone, PartialEq)]
pub struct SteepestSolution<F: Scalar> {
    pub arg: OVector<F, Dyn>,
    pub cost: F,
}

#[derive(Clone, Debug)]
pub struct SteepestParams<F> {
    pub(crate) tolerance: F,
    pub(crate) ls_params: LineSearch<F>,
    pub(crate) max_its: usize,
}

impl<F> SteepestParams<F> {
    pub fn new(tolerance: F, ls_params: LineSearch<F>, max_its: usize) -> Self
    where
        F: Num + PartialOrd,
    {
        assert!(F::zero() < tolerance);
        Self {
            tolerance,
            ls_params,
            max_its,
        }
    }
}

pub fn steepest_descent_method_with_observer<P, S, O>(
    problem: &mut P,
    x0: &Vector<P::F, Dyn, S>,
    params: &SteepestParams<P::F>,
    observer: &mut O,
) -> Result<SteepestSolution<P::F>, String>
where
    P: Gradient + ConvexConstraints,
    S: Storage<P::F, Dyn> + Debug,
    P::F: Debug
        + Scalar
        + NumAssign
        + ComplexField<RealField = P::F>
        + PartialOrd
        + Copy
        + FromPrimitive
        + Zero,
    O: SolverObserver<P::F>,
{
    let tol2 = params.tolerance * params.tolerance;
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

        let t = match &params.ls_params {
            LineSearch::Backtracking(backtrack_params) => {
                backtrack_line_search(problem, current_cost, &x, &dx, backtrack_params)
            }
            LineSearch::Guarded(backtrack_params) => guarded_line_search(
                problem,
                current_cost,
                directional_derivative,
                &x,
                &dx,
                backtrack_params,
            ),
        };

        x += &dx * t;

        if dx.norm_squared() <= tol2 || its > params.max_its {
            break;
        }

        its += 1;
    }

    let mut cost = P::F::zero();
    problem.cost(&x, &mut cost);

    Ok(SteepestSolution { cost, arg: x })
}

pub fn steepest_descent_method<P, S>(
    problem: &mut P,
    x0: &Vector<P::F, Dyn, S>,
    params: &SteepestParams<P::F>,
) -> Result<SteepestSolution<P::F>, String>
where
    P: Gradient + LinearConstraints + ConvexConstraints,
    P::F: Debug
        + Scalar
        + NumAssign
        + ComplexField<RealField = P::F>
        + PartialOrd
        + Copy
        + FromPrimitive
        + Zero,
    S: Storage<P::F, Dyn> + Debug,
{
    steepest_descent_method_with_observer(problem, x0, params, &mut ())
}
