use std::{fmt::Debug, iter::Sum, ops::Neg};

use nalgebra::{
    ComplexField, DMatrix, DVector, Dyn, Matrix, RawStorage, Scalar, Storage, StorageMut, Vector,
};
use num_traits::{Float, FromPrimitive, Inv, Num, NumAssign, One, Zero, real::Real};

use crate::{
    ConvexConstraints, CostFunction, Gradient, Hessian, LinearConstraints, PrimalDual,
    alg::descent::{
        DescentMethod,
        newton::{NewtonsMethodSolution, newtons_method_with_observer},
        steepest::steepest_descent_method_with_observer,
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
pub struct BarrierParams<F> {
    t0: F,
    mu: F,
    tolerance: F,
    center_params: DescentMethod<F>,
}

impl<F> BarrierParams<F> {
    pub fn new(t0: F, mu: F, tolerance: F, center_params: DescentMethod<F>) -> Self
    where
        F: Num + PartialOrd,
    {
        assert!(F::zero() < t0);
        assert!(F::one() < mu);
        assert!(F::zero() < tolerance);

        Self {
            t0,
            mu,
            tolerance,
            center_params,
        }
    }
}

pub fn barrier_method_with_observer<P, S, O>(
    problem: &mut P,
    x0: &Vector<P::F, Dyn, S>,
    params: &BarrierParams<P::F>,
    observer: &mut O,
) -> Result<NewtonsMethodSolution<P::F>, String>
where
    P: Hessian + ConvexConstraints + PrimalDual,
    P::F: Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
    S: Storage<P::F, Dyn> + Debug,
    O: SolverObserver<P::F>,
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
        accuracy: params.t0,
        const_buffer: vec![P::F::zero(); nconstr],
        const_grad_buffer: vec![DVector::zeros(nvar); nconstr],
        const_hess_buffer: vec![DMatrix::zeros(nvar, nvar); nconstr],
    };

    let nconstr = P::F::from_usize(nconstr).unwrap();

    let mut x = x0.clone_owned();

    loop {
        observer.on_step(SolverStep::BarrierIter(barrier.accuracy));

        let new_x = match &params.center_params {
            DescentMethod::NewtonsMethod(newton_params) => {
                newtons_method_with_observer(&mut barrier, &x, newton_params, observer)?.arg
            }
            DescentMethod::SteepestDescent(steepest_params) => {
                steepest_descent_method_with_observer(&mut barrier, &x, steepest_params, observer)?
                    .arg
            }
        };

        x = new_x;

        if nconstr / barrier.accuracy < params.tolerance {
            break;
        }

        barrier.accuracy *= params.mu;
    }

    let mut bcost = P::F::zero();
    barrier.cost(&x, &mut bcost);

    let mut cost = P::F::zero();
    problem.cost(&x, &mut cost);

    let sol = NewtonsMethodSolution { cost, arg: x };

    problem.convex_constraints(&sol.arg, &mut constraints);

    Ok(sol)
}

pub fn barrier_method<P, S>(
    problem: &mut P,
    x0: &Vector<P::F, Dyn, S>,
    params: &BarrierParams<P::F>,
) -> Result<NewtonsMethodSolution<P::F>, String>
where
    P: Hessian + ConvexConstraints + PrimalDual,
    P::F: Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
    S: Storage<P::F, Dyn> + Debug,
{
    barrier_method_with_observer(problem, x0, params, &mut ())
}

struct AuxiliaryProblem<'a, P> {
    problem: &'a mut P,
}

impl<'a, P> AuxiliaryProblem<'a, P> {
    fn new(problem: &'a mut P) -> Self {
        Self { problem }
    }
}

impl<P> CostFunction for AuxiliaryProblem<'_, P>
where
    P: CostFunction,
    P::F: Copy + FromPrimitive + ComplexField<RealField = P::F>,
{
    type F = P::F;

    fn cost<S>(&mut self, param: &Vector<Self::F, Dyn, S>, out: &mut Self::F)
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let s_idx = self.problem.dims();
        let s_val = param[s_idx];

        let x = param.rows(0, s_idx);
        let reg = P::F::from_f64(1e-4).unwrap() * x.norm_squared();
        *out = s_val + reg;
    }

    fn dims(&self) -> usize {
        self.problem.dims() + 1
    }
}

impl<P> Gradient for AuxiliaryProblem<'_, P>
where
    P: Gradient,
    P::F: Copy + FromPrimitive + One + Neg<Output = P::F> + ComplexField<RealField = P::F>,
{
    #[inline]
    fn gradient<S1, S2>(
        &mut self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut Vector<Self::F, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug,
    {
        let s_idx = self.problem.dims();
        let x = param.rows(0, s_idx);

        // FIX: Gradient of regularization term: 2 * 1e-4 * x
        let coef = P::F::from_f64(2.0 * 1e-4).unwrap();

        // Calculate gradient for x part
        let mut grad_x = out.rows_mut(0, s_idx);
        grad_x.copy_from(&x);
        grad_x *= coef;

        // Gradient for s part is 1
        out[s_idx] = P::F::one();
    }
}

impl<P> Hessian for AuxiliaryProblem<'_, P>
where
    P: Hessian,
    P::F: Copy + FromPrimitive + One + Neg<Output = P::F> + ComplexField<RealField = P::F>,
{
    fn hessian<S1, S2>(
        &mut self,
        _param: &Vector<Self::F, Dyn, S1>,
        out: &mut Matrix<Self::F, Dyn, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug,
    {
        let s_idx = self.problem.dims();

        // FIX: Hessian is 2 * 1e-4 * I for the x-block, 0 for s-block
        let coef = P::F::from_f64(2.0 * 1e-4).unwrap();

        out.fill(P::F::zero());

        let mut block = out.view_mut((0, 0), (s_idx, s_idx));
        block.fill_diagonal(coef);
    }
}

impl<P> LinearConstraints for AuxiliaryProblem<'_, P>
where
    P: LinearConstraints,
    P::F: Copy + FromPrimitive + ComplexField<RealField = P::F>,
{
    #[inline]
    fn num_linear_constraints(&self) -> usize {
        self.problem.num_linear_constraints()
    }

    fn mat_a<S>(&self, out: &mut Matrix<Self::F, Dyn, Dyn, S>)
    where
        S: StorageMut<Self::F, Dyn, Dyn>,
    {
        let s = self.problem.dims();
        self.problem.mat_a(&mut out.view_range_mut(.., ..s));
    }

    fn vec_b<S>(&self, out: &mut Vector<Self::F, Dyn, S>)
    where
        S: StorageMut<Self::F, Dyn>,
    {
        self.problem.vec_b(&mut out.rows_range_mut(..));
    }
}

impl<P> ConvexConstraints for AuxiliaryProblem<'_, P>
where
    P: ConvexConstraints,
    P::F: Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Neg<Output = P::F>,
{
    #[inline]
    fn num_convex_constraints(&self) -> usize {
        self.problem.num_convex_constraints()
    }

    fn convex_constraints<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [Self::F])
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let s = self.problem.dims();
        self.problem.convex_constraints(&param.rows_range(..s), out);
        let s = param[s];

        for x in out.iter_mut() {
            *x -= s;
        }
    }

    fn convex_gradients<S1, S2>(
        &self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut [Vector<Self::F, Dyn, S2>],
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug,
    {
        let s = self.problem.dims();
        self.problem.convex_gradients(&param.rows_range(..s), out);

        for x in out.iter_mut() {
            x[s] = -P::F::one();
        }
    }

    fn convex_hessians<S1, S2>(
        &self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut [Matrix<Self::F, Dyn, Dyn, S2>],
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug,
    {
        let s = self.problem.dims();
        self.problem.convex_hessians(&param.rows_range(..s), out);
    }
}

pub fn barrier_method_infeasible_with_observer<P, S, O>(
    problem: &mut P,
    x0: &Vector<P::F, Dyn, S>,
    bar_params: &BarrierParams<P::F>,
    aux_params: &BarrierParams<P::F>,
    observer: &mut O,
) -> Result<NewtonsMethodSolution<P::F>, String>
where
    P: Hessian + ConvexConstraints + PrimalDual,
    P::F: Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
    S: Storage<P::F, Dyn> + Debug,
    O: SolverObserver<P::F>,
{
    let nconstr = problem.num_convex_constraints();
    let mut constraints = vec![P::F::zero(); nconstr];
    problem.convex_constraints(x0, &mut constraints);

    let max = constraints
        .into_iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .ok_or("There should be at least one convex constraint. If it is not needed, use Newton's method instead".to_owned())?;

    let mut new_x0 = DVector::zeros(x0.len() + 1);
    new_x0.rows_mut(0, x0.len()).copy_from(x0);
    new_x0[x0.len()] = max + P::F::one();

    // Solve the auxiliary problem
    let mut aux = AuxiliaryProblem::new(problem);

    observer.on_step(SolverStep::BarrierPhase(1));

    let sol = barrier_method_with_observer(&mut aux, &new_x0, aux_params, observer)?;

    if sol.cost >= P::F::zero() {
        return Err("Couldn't find feasible solution".to_owned());
    }

    observer.on_step(SolverStep::BarrierPhase(2));

    let sol = barrier_method_with_observer(
        problem,
        &sol.arg.rows(0, problem.dims()),
        bar_params,
        observer,
    )?;

    Ok(sol)
}

pub fn barrier_method_infeasible<P, S>(
    problem: &mut P,
    x0: &Vector<P::F, Dyn, S>,
    bar_params: &BarrierParams<P::F>,
    aux_params: &BarrierParams<P::F>,
) -> Result<NewtonsMethodSolution<P::F>, String>
where
    P: Hessian + ConvexConstraints + PrimalDual,
    P::F: Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
    S: Storage<P::F, Dyn> + Debug,
{
    barrier_method_infeasible_with_observer(problem, x0, bar_params, aux_params, &mut ())
}
