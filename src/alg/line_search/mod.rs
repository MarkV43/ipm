use std::fmt::Debug;

use nalgebra::{Dyn, Storage, Vector};
use num_traits::{FromPrimitive, NumAssign};

use crate::ConvexConstraints;

pub mod backtrack;
pub mod guarded;

pub trait LineSearch {
    type F;

    fn search<P, S1, S2>(
        &self,
        problem: &mut P,
        current_cost: P::F,            // Pass the current cost
        gradient_dot_step: P::F,       // Directional derivative (grad^T * step)
        x: &Vector<P::F, Dyn, S1>,     // Current position
        dir_x: &Vector<P::F, Dyn, S2>, // Search direction
    ) -> P::F
    where
        P: ConvexConstraints<F = Self::F>, // Note: We need CostFunction, not just PrimalDual
        P::F: FromPrimitive + NumAssign + PartialOrd + Debug + Copy + 'static,
        S1: Storage<P::F, Dyn> + Debug,
        S2: Storage<P::F, Dyn> + Debug;
}
