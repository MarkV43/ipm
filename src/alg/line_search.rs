use nalgebra::{ComplexField, Dyn, Scalar, Storage, StorageMut, Vector};
use num_traits::{FromPrimitive, Num, NumAssign, One, Zero};
use std::fmt::Debug;

use crate::{ConvexConstraints, PrimalDual};

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

pub fn backtrack_line_search<P, S1, S2, S3>(
    problem: &mut P,
    residual: &mut Vector<P::F, Dyn, S1>,
    xv: &Vector<P::F, Dyn, S2>,
    dir_xv: &Vector<P::F, Dyn, S3>,
    params: &LineSearchParams<P::F>,
) -> P::F
where
    P: PrimalDual + ConvexConstraints,
    P::F: Scalar + ComplexField<RealField = P::F> + NumAssign + PartialOrd + Copy + FromPrimitive,
    S1: StorageMut<P::F, Dyn> + Debug,
    S2: Storage<P::F, Dyn> + Debug,
    S3: Storage<P::F, Dyn> + Debug,
{
    let mut t: P::F = P::F::one();
    let dims = problem.dims();

    problem.residual(xv, residual);
    let residual_at_xv = residual.norm_squared();

    let nconstr = problem.number_of_constraints();
    let mut constraints = vec![P::F::zero(); nconstr];

    let eps = P::F::from_f64(1e-9).unwrap();

    loop {
        let candidate = xv + dir_xv * t;
        let x_candidate = candidate.rows(0, dims).into_owned();

        problem.convex_constraints(&x_candidate, &mut constraints);
        let infeasible = constraints.iter().any(|&c| c >= P::F::zero());

        let bound = P::F::one() - t * params.alpha;

        problem.residual(&candidate, residual);
        let residual_too_large = residual.norm_squared() > bound * bound * residual_at_xv + eps;

        if !infeasible && !residual_too_large {
            return t;
        }

        t *= params.beta;
    }
}
