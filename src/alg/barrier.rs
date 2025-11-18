use std::{fmt::Debug, iter::Sum};

use nalgebra::{ComplexField, DMatrix, DVector, Dyn, RawStorage, Scalar, Storage, Vector};
use num_traits::{Float, FromPrimitive, Inv, Num, NumAssign, One, Zero, real::Real};

use crate::{
    ConvexConstraints, CostFunction, Gradient, Hessian, LinearConstraints, PrimalDual,
    alg::newton::{NewtonsMethodSolution, newtons_method},
};

pub struct BarrierProblem<'a, P: CostFunction> {
    problem: &'a mut P,
    accuracy: P::F,
    const_buffer: Vec<P::F>,
    const_grad_buffer: Vec<DVector<P::F>>,
    const_hess_buffer: Vec<DMatrix<P::F>>,
}

impl<'a, P> CostFunction for BarrierProblem<'a, P>
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
            self.problem.number_of_constraints()
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

impl<'a, P> Gradient for BarrierProblem<'a, P>
where
    P: ConvexConstraints,
    P::F: Real + Inv<Output = P::F> + Sum + NumAssign,
{
    fn gradient<S>(&mut self, param: &Vector<Self::F, Dyn, S>, out: &mut DVector<Self::F>)
    where
        Self::F: Debug + Scalar + Num,
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        self.problem.gradient(param, out);

        debug_assert_eq!(
            self.const_grad_buffer.len(),
            self.problem.number_of_constraints()
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

impl<'a, P> Hessian for BarrierProblem<'a, P>
where
    P: ConvexConstraints,
    P::F: Real + Inv<Output = P::F> + Sum + NumAssign,
{
    fn hessian<S>(&mut self, param: &Vector<Self::F, Dyn, S>, out: &mut DMatrix<Self::F>)
    where
        Self::F: Debug + Scalar,
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        self.problem.hessian(param, out);

        debug_assert_eq!(
            self.const_hess_buffer.len(),
            self.problem.number_of_constraints()
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
            // term = (H_f / f_i) - (g g^T / f_i^2)
            let term = hi * ci - (g * g.transpose()) * (ci * ci);
            // multiply by h and add to total Hessian
            *out += term * h;
        }
    }
}

impl<'a, P> LinearConstraints for BarrierProblem<'a, P>
where
    P: LinearConstraints + ConvexConstraints,
    P::F: Real + Inv<Output = P::F> + Sum,
{
    fn mat_a(&self) -> nalgebra::DMatrix<Self::F>
    where
        Self::F: Debug + Scalar,
    {
        self.problem.mat_a()
    }

    fn vec_b(&self) -> nalgebra::DVector<Self::F>
    where
        Self::F: Debug + Scalar,
    {
        self.problem.vec_b()
    }
}

impl<'a, P> ConvexConstraints for BarrierProblem<'a, P>
where
    P: ConvexConstraints,
    P::F: Inv<Output = P::F> + Sum + Scalar + Float + NumAssign,
{
    #[inline]
    fn number_of_constraints(&self) -> usize {
        self.problem.number_of_constraints()
    }

    fn convex_constraints<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [Self::F])
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        self.problem.convex_constraints(param, out)
    }

    fn convex_gradients<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [DVector<Self::F>])
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        self.problem.convex_gradients(param, out)
    }

    fn convex_hessians<S>(&self, param: &Vector<Self::F, Dyn, S>, out: &mut [DMatrix<Self::F>])
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        self.problem.convex_hessians(param, out)
    }
}

pub fn barrier_method<P, S>(
    problem: &mut P,
    x0: &Vector<P::F, Dyn, S>,
    t0: P::F,
    mu: P::F,
    tolerance: P::F,
    centering_tolerance: P::F,
    alpha: P::F,
    beta: P::F,
) -> NewtonsMethodSolution<P::F>
where
    P: Hessian + LinearConstraints + ConvexConstraints + PrimalDual,
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
    assert!(mu > P::F::one());

    let nconstr = problem.number_of_constraints();
    let mut constraints = vec![P::F::zero(); nconstr];
    problem.convex_constraints(x0, &mut constraints);
    assert!(constraints.iter().all(|&x| x <= P::F::zero()));

    let nvar = problem.dims();

    let mut barrier = BarrierProblem {
        problem,
        accuracy: t0,
        const_buffer: vec![P::F::zero(); nconstr],
        const_grad_buffer: vec![DVector::zeros(nvar); nconstr],
        const_hess_buffer: vec![DMatrix::zeros(nvar, nvar); nconstr],
    };

    let nconstr = P::F::from_usize(nconstr).unwrap();

    let mut x = x0.clone_owned();

    loop {
        let new_x = newtons_method(&mut barrier, &x, centering_tolerance, alpha, beta).arg;

        // println!("=====  Step  =====");
        // println!("Cost = {}", problem.cost(&x));

        x = new_x;

        if nconstr / barrier.accuracy < tolerance {
            break;
        }

        barrier.accuracy *= mu;
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
