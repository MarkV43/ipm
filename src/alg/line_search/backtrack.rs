use nalgebra::{DVector, Dyn, Storage, Vector};
use num_traits::{FromPrimitive, Num, NumAssign, One};
use std::fmt::Debug;

use crate::Gradient;

#[derive(Clone, Debug)]
pub struct BacktrackParams<F> {
    pub(crate) alpha: F,
    pub(crate) beta: F,
}

impl<F> BacktrackParams<F> {
    pub fn new(alpha: F, beta: F) -> Self
    where
        F: Num + FromPrimitive + PartialOrd,
    {
        assert!(F::zero() < alpha && alpha <= F::from_f32(0.5).unwrap());
        assert!(F::zero() < beta && beta < F::one());
        Self { alpha, beta }
    }
}

pub fn backtrack_line_search<P, S1, S2>(
    problem: &mut P,
    current_cost: P::F,            // Pass the current cost
    x: &Vector<P::F, Dyn, S1>,     // Current position
    dir_x: &Vector<P::F, Dyn, S2>, // Search direction
    params: &BacktrackParams<P::F>,
) -> P::F
where
    P: Gradient,
    P::F: Num + FromPrimitive + NumAssign + PartialOrd + Debug + Copy,
    S1: Storage<P::F, Dyn> + Debug,
    S2: Storage<P::F, Dyn> + Debug,
{
    let mut t = P::F::one();
    let beta = params.beta;
    let alpha = params.alpha;

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
