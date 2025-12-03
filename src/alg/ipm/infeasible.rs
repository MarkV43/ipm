use std::{fmt::Debug, iter::Sum, ops::Neg};

use nalgebra::{
    ComplexField, DVector, Dyn, Matrix, RawStorage, Scalar, Storage, StorageMut, Vector,
};
use num_traits::{Float, FromPrimitive, Inv, NumAssign, One, Zero};

use crate::{
    ConvexConstraints, CostFunction, Gradient, Hessian, LinearConstraints, PrimalDual,
    alg::ipm::{InteriorPointMethod, IpmSolution},
    observer::{SolverObserver, SolverStep},
};

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

#[derive(Clone, Debug)]
pub struct InfeasibleIpm<I1, I2> {
    phase_1_solver: I1,
    phase_2_solver: I2,
}

impl<I1, I2> InfeasibleIpm<I1, I2>
where
    I1: InteriorPointMethod,
    I2: InteriorPointMethod<F = I1::F>,
{
    pub fn new(phase_1_solver: I1, phase_2_solver: I2) -> Self {
        Self {
            phase_1_solver,
            phase_2_solver,
        }
    }

    pub fn phase_1_observe<P, S, O>(
        &self,
        problem: &mut P,
        x0: &Vector<P::F, Dyn, S>,
        observer: &mut O,
    ) -> Result<IpmSolution<I1::F>, String>
    where
        P: Hessian + ConvexConstraints + PrimalDual + CostFunction<F = I1::F>,
        P::F:
            Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
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

        let IpmSolution { arg, cost } = self
            .phase_1_solver
            .optimize_observe(&mut aux, &new_x0, observer)?;

        if cost >= P::F::zero() {
            Err("Couldn't find feasible solution".to_owned())
        } else {
            Ok(IpmSolution {
                cost,
                arg: arg.rows(0, problem.dims()).clone_owned(),
            })
        }
    }

    pub fn phase_1<P, S>(
        &self,
        problem: &mut P,
        x0: &Vector<P::F, Dyn, S>,
    ) -> Result<IpmSolution<I1::F>, String>
    where
        P: Hessian + ConvexConstraints + PrimalDual + CostFunction<F = I1::F>,
        P::F:
            Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
        S: Storage<P::F, Dyn> + Debug,
    {
        self.phase_1_observe(problem, x0, &mut ())
    }

    pub fn phase_2_observe<P, S, O>(
        &self,
        problem: &mut P,
        x0: &Vector<P::F, Dyn, S>,
        observer: &mut O,
    ) -> Result<IpmSolution<P::F>, String>
    where
        P: Hessian + ConvexConstraints + PrimalDual + CostFunction<F = I1::F>,
        P::F:
            Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
        S: Storage<P::F, Dyn> + Debug,
        O: SolverObserver<P::F>,
    {
        assert!({
            let mut constraints = vec![P::F::zero(); problem.num_convex_constraints()];
            problem.convex_constraints(x0, &mut constraints);
            constraints.iter().all(|x| *x < P::F::zero())
        });

        observer.on_step(SolverStep::BarrierPhase(2));

        let sol = self
            .phase_2_solver
            .optimize_observe(problem, &x0, observer)?;

        Ok(sol)
    }

    pub fn phase_2<P, S>(
        &self,
        problem: &mut P,
        x0: &Vector<P::F, Dyn, S>,
    ) -> Result<IpmSolution<P::F>, String>
    where
        P: Hessian + ConvexConstraints + PrimalDual + CostFunction<F = I1::F>,
        P::F:
            Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
        S: Storage<P::F, Dyn> + Debug,
    {
        self.phase_2_observe(problem, x0, &mut ())
    }

    pub fn optimize_observe<P, S, O>(
        &self,
        problem: &mut P,
        x0: &Vector<P::F, Dyn, S>,
        observer: &mut O,
    ) -> Result<IpmSolution<P::F>, String>
    where
        P: Hessian + ConvexConstraints + PrimalDual + CostFunction<F = I1::F>,
        P::F:
            Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
        S: Storage<P::F, Dyn> + Debug,
        O: SolverObserver<P::F>,
    {
        let sol1 = self.phase_1_observe(problem, x0, observer)?;
        self.phase_2_observe(problem, &sol1.arg, observer)
    }

    pub fn optimize<P, S>(
        &self,
        problem: &mut P,
        x0: &Vector<P::F, Dyn, S>,
    ) -> Result<IpmSolution<P::F>, String>
    where
        P: Hessian + ConvexConstraints + PrimalDual + CostFunction<F = I1::F>,
        P::F:
            Float + Scalar + NumAssign + ComplexField<RealField = P::F> + Sum + Inv<Output = P::F>,
        S: Storage<P::F, Dyn> + Debug,
    {
        let sol1 = self.phase_1(problem, x0)?;
        self.phase_2(problem, &sol1.arg)
    }
}
