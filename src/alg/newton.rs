use crate::{
    ConvexConstraints, Hessian, LinearConstraints, PrimalDual,
    alg::line_search::{LineSearchParams, backtrack_line_search},
};
use nalgebra::{ComplexField, DMatrix, DVector, Dyn, OVector, Scalar, Storage, Vector, stack};
use num_traits::{FromPrimitive, Num, NumAssign, Zero};
use std::fmt::Debug;

#[derive(Debug, Clone, PartialEq)]
pub struct NewtonsMethodSolution<F: Scalar> {
    pub arg: OVector<F, Dyn>,
    pub cost: F,
}

pub struct NewtonParams<F> {
    tolerance: F,
    ls_params: LineSearchParams<F>,
}

impl<F> NewtonParams<F> {
    pub fn new(tolerance: F, ls_params: LineSearchParams<F>) -> Self
    where
        F: Num + PartialOrd,
    {
        assert!(F::zero() < tolerance);
        Self {
            tolerance,
            ls_params,
        }
    }
}

#[must_use]
pub fn newtons_method<P, S>(
    problem: &mut P,
    x0: &Vector<P::F, Dyn, S>,
    params: &NewtonParams<P::F>,
) -> NewtonsMethodSolution<P::F>
where
    P: Hessian + LinearConstraints + ConvexConstraints + PrimalDual,
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
    let tol2 = params.tolerance * params.tolerance;

    let dims = problem.dims();

    let mat_a = problem.mat_a();
    let mat_at = mat_a.transpose();
    let vec_b = problem.vec_b();

    let mut x = x0.clone_owned();
    let mut v = DVector::zeros(mat_a.nrows());

    let mut its = 0;

    let mut hessian = DMatrix::zeros(dims, dims);
    let mut gradient = DVector::zeros(dims);

    let mut residual = DVector::zeros(x.nrows() + v.nrows());

    loop {
        problem.hessian(&x, &mut hessian);
        problem.gradient(&x, &mut gradient);

        let new_a = stack![hessian, &mat_at; &mat_a, 0];
        let new_b = -stack![gradient; &mat_a * &x - &vec_b];

        let dxv = new_a.lu().solve(&new_b).expect("Failed to invert matrix");
        // let dxv = new_a.try_inverse().unwrap() * new_b;
        let dx = dxv.rows(0, dims);

        let new_v = dxv.rows_range(dims..);
        let dv = new_v - &v;

        let t = backtrack_line_search(
            problem,
            &mut residual,
            &stack![x; v],
            &stack![dx; dv],
            &params.ls_params,
        );

        x += dx * t;
        v += dv * t;

        let residual = residual.norm_squared();

        assert!(residual.is_finite());

        if residual <= tol2 || its >= 10 && dxv.norm_squared() < tol2 || its > 100 {
            break;
        }

        its += 1;
    }

    let mut cost = P::F::zero();
    problem.cost(&x, &mut cost);

    NewtonsMethodSolution { cost, arg: x }
}
