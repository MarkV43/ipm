use crate::alg::line_search::backtrack::BacktrackParams;

pub mod backtrack;
pub mod guarded;

#[derive(Clone, Debug)]
pub enum LineSearch<F> {
    Backtracking(BacktrackParams<F>),
    Guarded(BacktrackParams<F>),
}
