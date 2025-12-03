use std::{fmt::Debug, iter::Sum};

use nalgebra::{
    ComplexField, DMatrix, DVector, Dyn, Matrix, RawStorage, Scalar, Storage, StorageMut, Vector,
};
use num_traits::{Float, Inv, Num, NumAssign, One, Zero, real::Real};

use crate::{
    ConvexConstraints, CostFunction, Gradient, Hessian, LinearConstraints, PrimalDual,
    alg::{
        descent::DescentMethod,
        ipm::{InteriorPointMethod, IpmSolution},
    },
    observer::{SolverObserver, SolverStep},
};

struct BarrierProblem<'a, P: CostFunction> {
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
    P::F: Real + Inv<Output = P::F> + Sum,
{
    type F = P::F;

    fn cost<S>(&mut self, param: &Vector<Self::F, Dyn, S>, out: &mut Self::F)
    where
        Self::F: Debug + Scalar,
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let mut cost = P::F::zero();
        self.problem.cost(param, &mut cost);

        debug_assert_eq!(
            self.const_buffer.len(),
            self.problem.num_convex_constraints()
        );
        self.problem
            .convex_constraints(param, &mut self.const_buffer);

        let h = -self.accuracy.inv();
        let res = cost + h * self.const_buffer.iter().map(|&x| (-x).ln()).sum();
        *out = res;
    }

    fn dims(&self) -> usize {
        self.problem.dims()
    }
}

impl<P> Gradient for BarrierProblem<'_, P>
where
    P: ConvexConstraints,
    P::F: Scalar + Real + Inv<Output = P::F> + Sum + NumAssign,
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

        debug_assert_eq!(
            self.const_grad_buffer.len(),
            self.problem.num_convex_constraints()
        );

        self.problem
            .convex_constraints(param, &mut self.const_buffer);
        self.problem
            .convex_gradients(param, &mut self.const_grad_buffer);

        // barrier multiplier (must be negative)
        let h = -self.accuracy.inv();

        for (g, c) in self.const_grad_buffer.iter().zip(self.const_buffer.iter()) {
            let ci = c.inv(); // 1 / f_i
            *out += g * h * ci; // add h * g / f_i
        }
    }
}

impl<P> Hessian for BarrierProblem<'_, P>
where
    P: ConvexConstraints,
    P::F: Scalar + Real + Inv<Output = P::F> + Sum + NumAssign,
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

        // barrier multiplier (must be negative)
        let h = -self.accuracy.inv();

        for (hi, (g, c)) in self
            .const_hess_buffer
            .iter()
            .zip(self.const_grad_buffer.iter().zip(self.const_buffer.iter()))
        {
            let ci = c.inv(); // 1 / f_i
            let ci_sq = ci * ci;
            let alpha = -h * ci_sq;

            *out += hi * (ci * h);
            out.ger(alpha, g, g, P::F::one());

            // term = (H_f / f_i) - (g g^T / f_i^2)
            // let term = hi * ci - (g * g.transpose()) * (ci * ci);
            // multiply by h and add to total Hessian
            // *out += term * h;
        }
    }
}

impl<P> LinearConstraints for BarrierProblem<'_, P>
where
    P: LinearConstraints + ConvexConstraints,
    P::F: Scalar + Real + Inv<Output = P::F> + Sum,
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
    P::F: Scalar + Inv<Output = P::F> + Sum + Float + NumAssign,
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
