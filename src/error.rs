use thiserror::Error;

/// Errors that can occur during local search optimization.
#[derive(Error, Debug)]
pub enum LocalsearchError {
    /// Failed to generate a random solution.
    #[error("Failed to generate random solution")]
    RandomGenerationError,
    /// Preprocessing of the solution failed.
    #[error("Preprocessing failed")]
    PreprocessError,
}
