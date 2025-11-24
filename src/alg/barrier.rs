use std::{fmt::Debug, iter::Sum};

use nalgebra::{
    ComplexField, DMatrix, DVector, Dyn, Matrix, RawStorage, Scalar, Storage, StorageMut, Vector,
};
use num_traits::{Float, FromPrimitive, Inv, Num, NumAssign, One, Zero, real::Real};

use crate::{
    ConvexConstraints, CostFunction, Gradient, Hessian, LinearConstraints, PrimalDual,
    alg::newton::{NewtonParams, NewtonsMethodSolution, newtons_method},
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
            out.ger(alpha, &g, &g, P::F::one());

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

    fn mat_a<S>(&self, out: &mut Matrix<Self::F, Dyn, Dyn, S>)
    where
        S: StorageMut<Self::F, Dyn, Dyn>,
    {
        self.problem.mat_a(out)
    }

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

    fn convex_constraints<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [Self::F])
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        self.problem.convex_constraints(param, out)
    }

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
    center_params: NewtonParams<F>,
}

impl<F> BarrierParams<F> {
    pub fn new(t0: F, mu: F, tolerance: F, center_params: NewtonParams<F>) -> Self
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

pub fn barrier_method<P, S>(
    problem: &mut P,
    x0: &Vector<P::F, Dyn, S>,
    params: &BarrierParams<P::F>,
) -> NewtonsMethodSolution<P::F>
where
    P: Hessian + ConvexConstraints + PrimalDual,
    P::F: Debug
        + Float
        + Scalar
        + NumAssign
        + ComplexField<RealField = P::F>
        + PartialOrd
        + Real
        + One
        + Sum
        + Inv<Output = P::F>
        + Zero
        + FromPrimitive,
    S: Storage<P::F, Dyn> + Debug,
{
    let nconstr = problem.num_convex_constraints();
    let mut constraints = vec![P::F::zero(); nconstr];
    problem.convex_constraints(x0, &mut constraints);
    assert!(
        constraints.iter().all(|&x| x <= P::F::zero()),
        "The initial condition must be feasible. Use `barrier_method_infeasible` for infeasible starts"
    );

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
        let new_x = newtons_method(&mut barrier, &x, &params.center_params).arg;

        let mut cost = P::F::zero();
        barrier.cost(&new_x, &mut cost);
        // println!("=====  Step  =====");
        // println!("Cost = {cost}");
        // println!("S = {}", new_x[new_x.len() - 1]);

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

    // println!("\n");
    // println!("x = {:?}", sol.arg.as_slice());
    // println!("cost = {}", sol.cost);
    // println!("bcost = {}", bcost);
    // println!("cnvx = {:?}", constraints.as_slice());

    sol
}

struct AuxiliaryProblem<'a, P> {
    problem: &'a mut P,
}

impl<'a, P> AuxiliaryProblem<'a, P> {
    fn new(problem: &'a mut P) -> Self {
        Self { problem }
    }
}

impl<'a, P> CostFunction for AuxiliaryProblem<'a, P>
where
    P: CostFunction,
    P::F: Copy,
{
    type F = P::F;

    fn cost<S>(&mut self, param: &Vector<Self::F, Dyn, S>, out: &mut Self::F)
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        *out = param[self.problem.dims()];
    }

    fn dims(&self) -> usize {
        self.problem.dims() + 1
    }
}

impl<'a, P> Gradient for AuxiliaryProblem<'a, P>
where
    P: Gradient,
    P::F: Copy + One,
{
    fn gradient<S1, S2>(
        &mut self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut Vector<Self::F, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn> + Debug,
    {
        let s = self.problem.dims();
        out[0] = P::F::one();
        self.problem
            .gradient(&param.rows_range(..s), &mut out.rows_range_mut(..s));
    }
}

impl<'a, P> Hessian for AuxiliaryProblem<'a, P>
where
    P: Hessian,
    P::F: Copy + One,
{
    fn hessian<S1, S2>(
        &mut self,
        param: &Vector<Self::F, Dyn, S1>,
        out: &mut Matrix<Self::F, Dyn, Dyn, S2>,
    ) where
        S1: RawStorage<Self::F, Dyn> + Debug,
        S2: StorageMut<Self::F, Dyn, Dyn> + Debug,
    {
        let s = self.problem.dims();
        self.problem
            .hessian(&param.rows_range(..s), &mut out.view_range_mut(..s, ..s));
    }
}

impl<'a, P> LinearConstraints for AuxiliaryProblem<'a, P>
where
    P: LinearConstraints,
    P::F: Copy,
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

impl<'a, P> ConvexConstraints for AuxiliaryProblem<'a, P>
where
    P: ConvexConstraints,
    P::F: Copy + One + NumAssign,
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
        out.iter_mut().for_each(|x| *x -= s);
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
        out.iter_mut().for_each(|x| x[s] = P::F::one());
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

pub fn barrier_method_infeasible<P, S>(
    problem: &mut P,
    x0: &Vector<P::F, Dyn, S>,
    params: &BarrierParams<P::F>,
) -> NewtonsMethodSolution<P::F>
where
    P: Hessian + ConvexConstraints + PrimalDual,
    P::F: Debug
        + Float
        + Scalar
        + NumAssign
        + ComplexField<RealField = P::F>
        + PartialOrd
        + Real
        + One
        + Sum
        + Inv<Output = P::F>
        + Zero
        + FromPrimitive,
    S: Storage<P::F, Dyn> + Debug,
{
    let nconstr = problem.num_convex_constraints();
    let mut constraints = vec![P::F::zero(); nconstr];
    problem.convex_constraints(x0, &mut constraints);

    let max = constraints
        .into_iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap())
        .expect("There should be at least one convex constraint. If it is not needed, use Newton's method instead");

    let mut new_x0 = DVector::zeros(x0.len() + 1);
    new_x0.rows_mut(0, x0.len()).copy_from(&x0);
    new_x0[x0.len()] = max + P::F::one();

    // Solve the auxiliary problem
    let mut inf_params = params.clone();
    let mut aux = AuxiliaryProblem::new(problem);

    let t0 = std::time::Instant::now();

    let sol = loop {
        let sol = barrier_method(&mut aux, &new_x0, &inf_params);

        // println!("Cost: {}", sol.cost);

        if sol.cost < P::F::zero() {
            break sol;
        }

        let one = P::F::one();
        let mu = &mut inf_params.mu;
        *mu = one + (*mu - one) * P::F::from_f64(0.5).unwrap();
    };

    println!("Phase I: {:?}", t0.elapsed());

    let t1 = std::time::Instant::now();
    let sol = barrier_method(problem, &sol.arg.rows(0, problem.dims()), params);

    println!("Phase II: {:?}", t1.elapsed());

    sol
}
