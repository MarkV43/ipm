use crate::{
    ConvexConstraints, CostFunction, Gradient, Hessian, LinearConstraints, PrimalDual,
    alg::line_search::backtrack_line_search,
};
use nalgebra::{ComplexField, DMatrix, DVector, Dyn, OVector, Scalar, Storage, Vector, stack};
use num_traits::{FromPrimitive, NumAssign, Zero};
use std::fmt::Debug;

#[derive(Debug, Clone, PartialEq)]
pub struct NewtonsMethodSolution<F: Scalar> {
    pub arg: OVector<F, Dyn>,
    pub cost: F,
}

#[must_use]
pub fn newtons_method<P, S>(
    problem: &mut P,
    x0: &Vector<P::F, Dyn, S>,
    tolerance: P::F,
    alpha: P::F,
    beta: P::F,
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
    let tol2 = tolerance * tolerance;

    let dims = problem.dims();

    let mat_a = problem.mat_a();
    let mat_at = mat_a.transpose();
    let vec_b = problem.vec_b();

    let mut x = x0.clone_owned();
    let mut v = DVector::zeros(mat_a.nrows());

    let mut its = 0;

    let mut hessian = DMatrix::zeros(dims, dims);
    let mut gradient = DVector::zeros(dims);

    loop {
        problem.hessian(&x, &mut hessian);
        problem.gradient(&x, &mut gradient);

        let new_a = stack![hessian, &mat_at; &mat_a, 0];
        let new_b = -stack![gradient; &mat_a * &x - &vec_b];

        // println!("{x}");

        let Some(dxv) = new_a.lu().solve(&new_b) else {
            break;
        };
        // let dxv = new_a.try_inverse().unwrap() * new_b;
        let dx = dxv.rows(0, dims);

        let new_v = dxv.rows_range(dims..);
        let dv = new_v - &v;

        let mut t = backtrack_line_search(problem, &stack![x; v], &stack![dx; dv], alpha, beta);

        let mut new_x = &x + &dx * t;
        let mut new_v = &v + &dv * t;

        let mut residual = problem.residual(&stack![new_x; new_v]).norm_squared();

        assert!(residual.is_finite());

        x = new_x;
        v = new_v;

        // println!("Residual: {residual} <= {tol2}");

        // let err = &mat_a * &x - &vec_b;
        if residual <= tol2 || its >= 10 && dxv.norm_squared() < tol2 || its > 100 {
            break;
        }

        its += 1;
    }

    let mut cost = P::F::zero();
    problem.cost(&x, &mut cost);

    NewtonsMethodSolution { cost, arg: x }
}
