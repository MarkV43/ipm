use crate::{
    ConvexConstraints, Hessian, LinearConstraints, PrimalDual,
    alg::{
        descent::{DescentMethod, DescentSolution},
        line_search::LineSearch,
    },
    observer::{SolverObserver, SolverStep},
};
use nalgebra::{ComplexField, DMatrix, DVector, Dyn, OVector, Scalar, Storage, Vector, stack};
use num_traits::{FromPrimitive, Num, NumAssign, Zero};
use std::fmt::Debug;

#[derive(Debug, Clone, PartialEq)]
pub struct NewtonsMethodSolution<F: Scalar> {
    pub arg: OVector<F, Dyn>,
    pub cost: F,
}

#[derive(Clone, Debug)]
pub struct NewtonsMethod<F, L: LineSearch<F = F>> {
    pub(crate) tolerance: F,
    pub(crate) line_search: L,
    pub(crate) min_its: usize,
    pub(crate) max_its: usize,
}

impl<F, L: LineSearch<F = F>> NewtonsMethod<F, L> {
    pub fn new(tolerance: F, line_search: L, min_its: usize, max_its: usize) -> Result<Self, String>
    where
        F: Num + PartialOrd,
    {
        if F::zero() >= tolerance {
            return Err("tolerance must be strictly positive".to_owned());
        }

        Ok(Self {
            tolerance,
            line_search,
            min_its,
            max_its,
        })
    }
}

impl<F: Scalar, L: LineSearch<F = F>> DescentMethod for NewtonsMethod<F, L> {
    type F = F;

    fn optimize_observe<P, S, O>(
        &self,
        problem: &mut P,
        x0: &Vector<F, Dyn, S>,
        observer: &mut O,
    ) -> Result<super::DescentSolution<F>, String>
    where
        P: Hessian
            + LinearConstraints
            + ConvexConstraints
            + PrimalDual
            + crate::CostFunction<F = F>,
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
        let nconstr = problem.num_linear_constraints();

        let mut mat_a = DMatrix::zeros(nconstr, dims);
        let mut vec_b = DVector::zeros(nconstr);

        problem.mat_a(&mut mat_a);
        problem.vec_b(&mut vec_b);

        let mat_at = mat_a.transpose();

        let mut x = x0.clone_owned();
        let mut v = DVector::zeros(mat_a.nrows());

        let mut its = 0;

        let mut hessian = DMatrix::zeros(dims, dims);
        let mut gradient = DVector::zeros(dims);

        let mut residual = DVector::zeros(x.nrows() + v.nrows());

        // let mut dxv;

        loop {
            problem.hessian(&x, &mut hessian);
            problem.gradient(&x, &mut gradient);

            let new_a = stack![hessian, &mat_at; &mat_a, 0];
            let new_b = -stack![gradient; &mat_a * &x - &vec_b]; // new_b

            let dxv = new_a
                .clone()
                .cholesky()
                .map(|x| x.solve(&new_b))
                .ok_or("Matrix `new_a` is singular".to_owned())?;

            // let dxv = new_a.try_inverse().unwrap() * new_b;
            let dx = dxv.rows(0, dims);

            let new_v = dxv.rows_range(dims..);
            let dv = new_v - &v;

            let directional_derivative = gradient.dot(&dx);
            let mut current_cost = P::F::zero();
            problem.cost(&x, &mut current_cost);

            let step = SolverStep::NewtonsPoint {
                primal: &x.clone_owned(),
                dual: &v.clone_owned(),
                cost: current_cost,
            };

            observer.on_step(step);

            let xv = stack![x; v];

            let t = self.line_search.search(
                problem,
                current_cost,
                directional_derivative,
                &xv,
                &stack![dx; dv],
            );

            x += dx * t;
            v += dv * t;

            problem.residual(&xv, &mut residual);
            let residual = residual.norm_squared();

            assert!(residual.is_finite());

            if residual <= tol2
                || its >= self.min_its && dxv.norm_squared() < tol2
                || its > self.max_its
            {
                break;
            }

            its += 1;
        }

        let mut cost = P::F::zero();
        problem.cost(&x, &mut cost);

        Ok(DescentSolution { cost, arg: x })
    }
}
