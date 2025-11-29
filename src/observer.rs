use nalgebra::{DVector, Scalar};
use num_traits::Zero;

pub enum SolverStep<'a, F> {
    NewtonsPoint {
        primal: &'a DVector<F>,
        dual: &'a DVector<F>,
        cost: F,
    },
    BarrierIter(F),
    BarrierPhase(u8),
}

pub trait SolverObserver<F> {
    fn on_step(&mut self, x: SolverStep<'_, F>);
}

impl<F> SolverObserver<F> for () {
    fn on_step(&mut self, _: SolverStep<'_, F>) {}
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathPoint<F> {
    pub aux: Option<F>,
    pub primal: DVector<F>,
    pub dual: DVector<F>,
    pub iter: usize,
    pub phase: u8,
    pub cost: F,
    pub accuracy: F,
}

pub struct PathRecorder<F> {
    pub path: Vec<PathPoint<F>>,
    iter: usize,
    phase: u8,
    accuracy: F,
}

impl<F> Default for PathRecorder<F>
where
    F: Zero,
{
    fn default() -> Self {
        Self {
            path: Vec::new(),
            iter: 0,
            phase: 0,
            accuracy: F::zero(),
        }
    }
}

impl<F: Scalar> SolverObserver<F> for PathRecorder<F> {
    fn on_step(&mut self, x: SolverStep<'_, F>) {
        match x {
            SolverStep::NewtonsPoint { primal, dual, cost } => {
                let has_aux = self.phase == 1;
                let aux = has_aux.then_some(primal[primal.len() - 1].clone());
                let primal = if has_aux {
                    primal.rows(0, primal.len() - 1)
                } else {
                    primal.rows_range(..)
                };
                self.path.push(PathPoint {
                    accuracy: self.accuracy.clone(),
                    primal: primal.clone_owned(),
                    dual: dual.clone_owned(),
                    phase: self.phase,
                    iter: self.iter,
                    cost,
                    aux,
                });
            }
            SolverStep::BarrierIter(accuracy) => {
                self.accuracy = accuracy;
                self.iter += 1;
            }
            SolverStep::BarrierPhase(phase) => self.phase = phase,
        }
    }
}
