use nalgebra::{Dyn, Storage, Vector};
use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

use crate::{ConvexConstraints, alg::line_search::LineSearch};

#[derive(Clone, Debug)]
pub struct GuardedLineSearch<F> {
    alpha: F,
    beta: F,
}

impl<F> GuardedLineSearch<F> {
    pub fn new(alpha: F, beta: F) -> Result<Self, String>
    where
        F: Float + FromPrimitive,
    {
        if alpha <= F::zero() || alpha > F::from_f64(0.5).unwrap() {
            return Err("alpha must be in range (0, 0.5]".to_owned());
        }
        if beta <= F::zero() || beta >= F::one() {
            return Err("beta must be in range (0, 1)".to_owned());
        }

        Ok(Self { alpha, beta })
    }
}

impl<F> LineSearch for GuardedLineSearch<F> {
    type F = F;

    fn search<P, S1, S2>(
        &self,
        problem: &mut P,
        current_cost: F,            // Pass the current cost
        gradient_dot_step: F,       // Directional derivative (grad^T * step)
        x: &Vector<F, Dyn, S1>,     // Current position
        dir_x: &Vector<F, Dyn, S2>, // Search direction
    ) -> F
    where
        P: ConvexConstraints<F = F>, // Note: We need CostFunction, not just PrimalDual
        F: FromPrimitive + NumAssign + PartialOrd + Debug + Copy + 'static,
        S1: Storage<F, Dyn> + Debug,
        S2: Storage<F, Dyn> + Debug,
    {
        let mut t = P::F::one();
        let dims = problem.dims();
        let beta = self.beta;
        let alpha = self.alpha;

        // Standard Armijo condition: f(x + t*d) <= f(x) + alpha * t * (grad^T * d)
        let margin = P::F::from_f64(-1e-10).unwrap();

        let nconstr = problem.num_convex_constraints();
        let mut constraints = vec![P::F::zero(); nconstr];

        loop {
            let candidate = x + dir_x * t;
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

            assert!(
                t > P::F::from_f64(1e-16).unwrap(),
                "Line search in invalid direction"
            );
        }
    }
}
