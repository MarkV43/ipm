use crate::alg::descent::{newton::NewtonParams, steepest::SteepestParams};

pub mod newton;
pub mod steepest;

#[derive(Clone, Debug)]
pub enum DescentMethod<F> {
    NewtonsMethod(NewtonParams<F>),
    SteepestDescent(SteepestParams<F>),
}
