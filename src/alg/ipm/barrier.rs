use std::{fmt::Debug, iter::Sum};

use nalgebra::{
    ComplexField, DMatrix, DVector, Dyn, Matrix, RawStorage, Scalar, Storage, StorageMut, Vector,
};
use num_traits::{Float, Inv, Num, NumAssign, One, Zero};

use crate::{
    ConvexConstraints, CostFunction, Gradient, Hessian, LinearConstraints, PrimalDual,
    alg::{
        descent::DescentMethod,
        ipm::{InteriorPointMethod, IpmSolution},
    },
    observer::{SolverObserver, SolverStep},
};

struct BarrierProblem<'a, P: CostFunction>
where
    P::F: ComplexField<RealField = P::F>,
{
    problem: &'a mut P,
    accuracy: P::F,

    // Buffers (Data)
    const_buffer: Vec<P::F>,
    const_grad_buffer: Vec<DVector<P::F>>,
    const_hess_buffer: Vec<DMatrix<P::F>>,
}

impl<P> CostFunction for BarrierProblem<'_, P>
where
    P: ConvexConstraints,
    P::F: Inv<Output = P::F> + Sum + ComplexField<RealField = P::F> + Copy,
{
    type F = P::F;

    fn cost<S>(&mut self, param: &Vector<Self::F, Dyn, S>, out: &mut Self::F)
    where
        Self::F: Debug + Scalar,
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let mut cost = P::F::zero();
        self.problem.cost(param, &mut cost);

        let m_size = self.problem.matrix_constraint_size();
        let mut matrix_buffer = DMatrix::zeros(m_size, m_size);

        debug_assert_eq!(
            self.const_buffer.len(),
            self.problem.num_convex_constraints()
        );
        self.problem
            .convex_constraints(param, &mut self.const_buffer);
        self.problem.matrix_constraints(param, &mut matrix_buffer);

        let cnvx_barrier = self.const_buffer.iter().map(|&x| (-x).ln()).sum();

        let matrix_cholesky = matrix_buffer.cholesky().unwrap();
        let matrix_barrier = matrix_cholesky.ln_determinant();

        let res = self.accuracy * cost - cnvx_barrier - matrix_barrier;
        *out = res;
    }

    fn dims(&self) -> usize {
        self.problem.dims()
    }
}

impl<P> Gradient for BarrierProblem<'_, P>
where
    P: ConvexConstraints,
    P::F: Scalar + ComplexField<RealField = P::F> + Inv<Output = P::F> + Sum + NumAssign + Copy,
{
    fn gradient<S1, S2>(
        &mut self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut Vector<Self::F, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug,
    {
        self.problem.gradient(param, out);

        *out *= self.accuracy;

        debug_assert_eq!(
            self.const_grad_buffer.len(),
            self.problem.num_convex_constraints()
        );

        self.problem
            .convex_constraints(param, &mut self.const_buffer);
        self.problem
            .convex_gradients(param, &mut self.const_grad_buffer);

        for (g, c) in self.const_grad_buffer.iter().zip(self.const_buffer.iter()) {
            let ci = c.inv(); // 1 / f_i
            *out -= g * ci; // add h * g / f_i
        }

        let m_size = self.problem.matrix_constraint_size();
        let mut matrix_buffer = DMatrix::zeros(m_size, m_size);
        self.problem.matrix_constraints(param, &mut matrix_buffer);
        let matrix_cholesky = matrix_buffer.cholesky().unwrap();
        let cholesky_inv = matrix_cholesky.inverse();
        let inv_flat = DVector::from_column_slice(cholesky_inv.as_slice());

        let mut out_p = out.rows_range_mut(self.problem.matrix_contraints_range());

        out_p -= inv_flat;
    }
}

impl<P> Hessian for BarrierProblem<'_, P>
where
    P: ConvexConstraints,
    P::F: Scalar + ComplexField<RealField = P::F> + Inv<Output = P::F> + Sum + NumAssign + Copy,
{
    fn hessian<S1, S2>(
        &mut self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut Matrix<Self::F, Dyn, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug,
    {
        self.problem.hessian(param, out);

        *out *= self.accuracy;

        debug_assert_eq!(
            self.const_hess_buffer.len(),
            self.problem.num_convex_constraints()
        );

        self.problem
            .convex_hessians(param, &mut self.const_hess_buffer);
        self.problem
            .convex_gradients(param, &mut self.const_grad_buffer);
        self.problem
            .convex_constraints(param, &mut self.const_buffer);

        for (h, (g, c)) in self
            .const_hess_buffer
            .iter()
            .zip(self.const_grad_buffer.iter().zip(self.const_buffer.iter()))
        {
            let ci = c.inv(); // 1 / f_i
            let ci_sq = ci * ci;

            *out -= h * ci;
            out.ger(ci_sq, g, g, -P::F::one());
        }

        let m_size = self.problem.matrix_constraint_size();
        let mut matrix_buffer = DMatrix::zeros(m_size, m_size);
        self.problem.matrix_constraints(param, &mut matrix_buffer);
        let matrix_cholesky = matrix_buffer.cholesky().unwrap();
        let inv = matrix_cholesky.inverse();
        let kron = inv.kronecker(&inv);

        let p_range = self.problem.matrix_contraints_range();

        let mut out_p = out.view_range_mut(p_range.clone(), p_range);

        if out_p.nrows() == kron.nrows() {
            out_p -= kron;
        } else {
            panic!("Hessian dimension mismatch: Is your param vector just the matrix P?");
        }
    }
}

impl<P> LinearConstraints for BarrierProblem<'_, P>
where
    P: LinearConstraints + ConvexConstraints,
    P::F: Scalar + ComplexField<RealField = P::F> + Inv<Output = P::F> + Sum + NumAssign + Copy,
{
    fn num_linear_constraints(&self) -> usize {
        self.problem.num_linear_constraints()
    }

    #[allow(clippy::semicolon_if_nothing_returned)]
    fn mat_a<S>(&self, out: &mut Matrix<Self::F, Dyn, Dyn, S>)
    where
        S: StorageMut<Self::F, Dyn, Dyn>,
    {
        self.problem.mat_a(out)
    }

    #[allow(clippy::semicolon_if_nothing_returned)]
    fn vec_b<S>(&self, out: &mut Vector<Self::F, Dyn, S>)
    where
        S: StorageMut<Self::F, Dyn>,
    {
        self.problem.vec_b(out)
    }
}

#[allow(clippy::semicolon_if_nothing_returned)]
impl<P> ConvexConstraints for BarrierProblem<'_, P>
where
    P: ConvexConstraints,
    P::F: Scalar + ComplexField<RealField = P::F> + Inv<Output = P::F> + Sum + NumAssign + Copy,
{
    #[inline]
    fn num_convex_constraints(&self) -> usize {
        self.problem.num_convex_constraints()
    }

    #[allow(clippy::semicolon_if_nothing_returned)]
    fn convex_constraints<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [Self::F])
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        self.problem.convex_constraints(param, out)
    }

    #[allow(clippy::semicolon_if_nothing_returned)]
    fn convex_gradients<S1, S2>(
        &self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut [Vector<Self::F, Dyn, S2>],
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug,
    {
        self.problem.convex_gradients(param, out)
    }

    #[allow(clippy::semicolon_if_nothing_returned)]
    fn convex_hessians<S1, S2>(
        &self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut [Matrix<Self::F, Dyn, Dyn, S2>],
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug,
    {
        self.problem.convex_hessians(param, out)
    }

    fn matrix_constraints<S1, S2>(
        &self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut nalgebra::SquareMatrix<Self::F, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug,
    {
        self.problem.matrix_constraints(param, out);
    }

    fn matrix_constraint_size(&self) -> usize {
        self.problem.matrix_constraint_size()
    }
}

#[derive(Clone, Debug)]
pub struct BarrierMethod<F: Scalar, D: DescentMethod<F = F>> {
    t0: F,
    mu: F,
    tolerance: F,
    centering_solver: D,
}

impl<F: Scalar, D: DescentMethod<F = F>> BarrierMethod<F, D> {
    pub fn new(t0: F, mu: F, tolerance: F, centering_solver: D) -> Result<Self, String>
    where
        F: Num + PartialOrd,
    {
        if F::zero() >= t0 {
            return Err("t0 must be strictly positive".to_owned());
        }
        if F::one() >= mu {
            return Err("mu must be strictly positive".to_owned());
        }
        if F::zero() >= tolerance {
            return Err("tolerance must be strictly positive".to_owned());
        }

        Ok(Self {
            t0,
            mu,
            tolerance,
            centering_solver,
        })
    }
}

impl<F: Scalar, D: DescentMethod<F = F>> InteriorPointMethod for BarrierMethod<F, D> {
    type F = F;

    fn optimize_observe<P, S, O>(
        &self,
        problem: &mut P,
        x0: &Vector<P::F, Dyn, S>,
        observer: &mut O,
    ) -> Result<super::IpmSolution<F>, String>
    where
        P: Hessian + ConvexConstraints + PrimalDual + CostFunction<F = F>,
        F: Float + Scalar + NumAssign + ComplexField<RealField = F> + Sum + Inv<Output = F>,
        S: Storage<F, Dyn> + Debug,
        O: SolverObserver<F>,
    {
        let nconstr = problem.num_convex_constraints();
        let mut constraints = vec![P::F::zero(); nconstr];
        problem.convex_constraints(x0, &mut constraints);

        if constraints.iter().any(|&x| x >= P::F::zero()) {
            return Err("The initial condition must be feasible. Use `barrier_method_infeasible` for infeasible starts".to_owned());
        }

        let nvar = problem.dims();

        let mut barrier = BarrierProblem {
            problem,
            accuracy: self.t0,
            const_buffer: vec![P::F::zero(); nconstr],
            const_grad_buffer: vec![DVector::zeros(nvar); nconstr],
            const_hess_buffer: vec![DMatrix::zeros(nvar, nvar); nconstr],
        };

        let nconstr = P::F::from_usize(nconstr).unwrap();

        let mut x = x0.clone_owned();

        loop {
            observer.on_step(SolverStep::BarrierIter(barrier.accuracy));

            let new_x = self
                .centering_solver
                .optimize_observe(&mut barrier, &x, observer)?
                .arg;

            x = new_x;

            if nconstr / barrier.accuracy < self.tolerance {
                break;
            }

            barrier.accuracy *= self.mu;
        }

        let mut bcost = P::F::zero();
        barrier.cost(&x, &mut bcost);

        let mut cost = P::F::zero();
        problem.cost(&x, &mut cost);

        let sol = IpmSolution { cost, arg: x };

        problem.convex_constraints(&sol.arg, &mut constraints);

        Ok(sol)
    }
}
