use std::{fmt::Debug, iter::Sum};

use nalgebra::{
    ComplexField, DMatrix, DVector, Dyn, RawStorage, Scalar, Storage, StorageMut, Vector, dvector,
};
use num_traits::{Float, FromPrimitive, Inv, Num, NumAssign, One, Zero, real::Real};

use crate::{
    ConvexConstraints, CostFunction, Gradient, Hessian, LinearConstraints, PrimalDual,
    alg::newton::{NewtonsMethodSolution, newtons_method},
};

pub struct BarrierProblem<'a, P: CostFunction> {
    problem: &'a P,
    accuracy: P::F,
}

impl<'a, P> CostFunction for BarrierProblem<'a, P>
where
    P: CostFunction + ConvexConstraints,
    P::F: Real + Inv<Output = P::F> + Sum,
{
    type F = P::F;

    fn cost<S>(&self, param: &Vector<Self::F, Dyn, S>) -> Self::F
    where
        Self::F: Debug + Scalar,
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let cost = self.problem.cost(param);
        let constr = self.problem.convex_constraints(param);

        let h = -self.accuracy.inv();
        cost + h * constr.into_iter().map(|x| (-x).ln()).sum()
    }

    fn dims(&self) -> usize {
        self.problem.dims()
    }
}

impl<'a, P> Gradient for BarrierProblem<'a, P>
where
    P: Gradient + ConvexConstraints,
    P::F: Real + Inv<Output = P::F> + Sum + NumAssign,
{
    fn gradient<S>(&self, param: &Vector<Self::F, Dyn, S>) -> nalgebra::DVector<Self::F>
    where
        Self::F: Debug + Scalar + Num,
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let mut out = self.problem.gradient(param);

        let grads = self.problem.convex_gradients(param);
        let costs = self.problem.convex_constraints(param);

        // barrier multiplier (must be negative)
        let h = -self.accuracy.inv();

        for (g, c) in grads.into_iter().zip(costs.into_iter()) {
            let ci = c.inv(); // 1 / f_i
            out += g * (h * ci); // add h * g / f_i
        }

        out
    }
}

impl<'a, P> Hessian for BarrierProblem<'a, P>
where
    P: Hessian + ConvexConstraints,
    P::F: Real + Inv<Output = P::F> + Sum + NumAssign,
{
    fn hessian<S>(&self, param: &Vector<Self::F, Dyn, S>) -> nalgebra::DMatrix<Self::F>
    where
        Self::F: Debug + Scalar,
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        let mut out = self.problem.hessian(param);

        let hessians = self.problem.convex_hessians(param);
        let grads = self.problem.convex_gradients(param);
        let costs = self.problem.convex_constraints(param);

        // barrier multiplier (must be negative)
        let h = -self.accuracy.inv();

        for (hi, (g, c)) in hessians
            .into_iter()
            .zip(grads.into_iter().zip(costs.into_iter()))
        {
            let ci = c.inv(); // 1 / f_i
            // term = (H_f / f_i) - (g g^T / f_i^2)
            let term = hi * ci - (&g * g.transpose()) * (ci * ci);
            // multiply by h and add to total Hessian
            out += term * h;
        }

        out
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
    fn convex_constraints<S>(&self, param: &Vector<Self::F, Dyn, S>) -> Vec<Self::F>
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        self.problem.convex_constraints(param)
    }

    fn convex_gradients<S>(&self, param: &Vector<Self::F, Dyn, S>) -> Vec<DVector<Self::F>>
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        self.problem.convex_gradients(param)
    }

    fn convex_hessians<S>(&self, param: &Vector<Self::F, Dyn, S>) -> Vec<DMatrix<Self::F>>
    where
        S: RawStorage<Self::F, Dyn> + Debug,
    {
        self.problem.convex_hessians(param)
    }
}

pub fn barrier_method<P, S>(
    problem: &P,
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

    let constraints = problem.convex_constraints(x0);
    assert!(constraints.iter().all(|&x| x <= P::F::zero()));
    let n_constraints = P::F::from_usize(constraints.len()).unwrap();

    let mut barrier = BarrierProblem {
        problem,
        accuracy: t0,
    };

    let mut x = x0.clone_owned();

    loop {
        let new_x = newtons_method(&barrier, &x, centering_tolerance, alpha, beta).arg;

        // println!("=====  Step  =====");
        // println!("Cost = {}", problem.cost(&x));

        x = new_x;

        if n_constraints / barrier.accuracy < tolerance {
            break;
        }

        barrier.accuracy *= mu;
    }

    let sol = NewtonsMethodSolution {
        cost: problem.cost(&x),
        arg: x,
    };

    println!("\n");
    println!("x = {:?}", sol.arg.as_slice());
    println!("cost = {}", problem.cost(&sol.arg));
    println!("bcost = {}", barrier.cost(&sol.arg));
    println!("grad = {:?}", problem.gradient(&sol.arg).as_slice());
    println!(
        "cnvx = {:?}",
        problem.convex_constraints(&sol.arg).as_slice()
    );

    sol
}
