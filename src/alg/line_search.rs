use nalgebra::{ComplexField, Dyn, Scalar, Storage, StorageMut, Vector};
use num_traits::{Float, FromPrimitive, Num, NumAssign, One, Zero, real::Real};
use std::{fmt::Debug, ops::Neg};

use crate::{ConvexConstraints, CostFunction, PrimalDual};

#[derive(Clone, Debug)]
pub struct LineSearchParams<F> {
    pub(crate) alpha: F,
    pub(crate) beta: F,
}

impl<F> LineSearchParams<F> {
    pub fn new(alpha: F, beta: F) -> Self
    where
        F: Num + FromPrimitive + PartialOrd,
    {
        assert!(F::zero() < alpha && alpha < F::from_f32(0.5).unwrap());
        assert!(F::zero() < beta && beta < F::one());
        Self { alpha, beta }
    }
}

pub fn backtrack_line_search<P, S1, S2>(
    problem: &mut P,
    current_cost: P::F,             // Pass the current cost
    gradient_dot_step: P::F,        // Directional derivative (grad^T * step)
    xv: &Vector<P::F, Dyn, S1>,     // Current position
    dir_xv: &Vector<P::F, Dyn, S2>, // Search direction
    params: &LineSearchParams<P::F>,
) -> P::F
where
    P: ConvexConstraints, // Note: We need CostFunction, not just PrimalDual
    P::F: FromPrimitive + NumAssign + PartialOrd + Debug + Copy,
    S1: Storage<P::F, Dyn>,
    S2: Storage<P::F, Dyn>,
{
    let mut t = P::F::one();
    let dims = problem.dims();
    let beta = params.beta;
    let alpha = params.alpha;

    // Standard Armijo condition: f(x + t*d) <= f(x) + alpha * t * (grad^T * d)
    let margin = P::F::from_f64(-1e-10).unwrap();

    let nconstr = problem.num_convex_constraints();
    let mut constraints = vec![P::F::zero(); nconstr];

    loop {
        let candidate = xv + dir_xv * t;
        // Extract primal part if xv contains dual variables
        let x_candidate = candidate.rows(0, dims);

        // 1. Feasibility Check (Strict)
        problem.convex_constraints(&x_candidate, &mut constraints);
        // In Barrier, we must stay strictly feasible ( < 0 )
        let is_feasible = constraints.iter().all(|&c| c < margin);

        if is_feasible {
            let mut new_cost = P::F::zero();
            problem.cost(&x_candidate, &mut new_cost);

            let expected_decrease = alpha * t * gradient_dot_step;

            // 2. Sufficient Decrease Check (Armijo)
            if new_cost <= current_cost + expected_decrease {
                return t;
            }
        }

        t *= beta;

        if t < P::F::from_f64(1e-16).unwrap() {
            // Prevent infinite loops if step vanishes
            return t;
        }
    }
}
