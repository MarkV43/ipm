use nalgebra::{DVector, Dyn, Storage, Vector};
use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::Debug;

use crate::{ConvexConstraints, alg::line_search::LineSearch};

#[derive(Clone, Debug)]
pub struct BacktrackingLineSearch<F> {
    alpha: F,
    beta: F,
}

impl<F> BacktrackingLineSearch<F> {
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

impl<F> LineSearch for BacktrackingLineSearch<F> {
    type F = F;

    fn search<P, S1, S2>(
        &self,
        problem: &mut P,
        current_cost: F,            // Pass the current cost
        _: F,                       // Directional derivative (grad^T * step)
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
        let beta = self.beta;
        let alpha = self.alpha;

        let mut cost = current_cost;
        let mut current_grad = DVector::zeros(problem.dims());
        problem.gradient(x, &mut current_grad);
        let gr_dx = current_grad.dot(dir_x);

        loop {
            problem.cost(&(x + dir_x * t), &mut cost);

            if cost > gr_dx * alpha * t + current_cost {
                t *= beta;
            } else {
                return t;
            }
        }
    }
}
