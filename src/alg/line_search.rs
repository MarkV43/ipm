use nalgebra::{ComplexField, Dyn, Scalar, Storage, Vector};
use num_traits::{NumAssign, One, Zero};
use std::fmt::Debug;

use crate::{ConvexConstraints, PrimalDual};

pub fn backtrack_line_search<P, S1, S2>(
    problem: &P,
    xv: &Vector<P::F, Dyn, S1>,
    dir_xv: &Vector<P::F, Dyn, S2>,
    alpha: P::F,
    beta: P::F,
) -> P::F
where
    P: PrimalDual + ConvexConstraints,
    P::F: Debug + Scalar + ComplexField<RealField = P::F> + NumAssign + PartialOrd + Copy + Zero,
    S1: Storage<P::F, Dyn> + Debug,
    S2: Storage<P::F, Dyn> + Debug,
{
    let mut t: P::F = P::F::one();
    let dims = problem.dims();
    let residual_at_xv = problem.residual(xv).norm();

    let nconstr = problem.number_of_constraints();
    let mut constraints = vec![P::F::zero(); nconstr];

    loop {
        let candidate = xv + dir_xv * t;
        let x_candidate = candidate.rows(0, dims).into_owned();

        problem.convex_constraints(&x_candidate, &mut constraints);
        let infeasible = constraints.iter().any(|&c| c >= P::F::zero());

        let bound = (P::F::one() - t * alpha) * residual_at_xv;

        let residual_too_large = problem.residual(&candidate).norm_squared() > bound * bound;

        if !infeasible && !residual_too_large {
            return t;
        }

        t *= beta;
    }
}
