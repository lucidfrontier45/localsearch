//! Per-algorithm transition handlers.
//!
//! Each handler owns its state directly (no `Rc<RefCell>` wrappers) and
//! implements [`crate::optim::transition::TransitionHandler`].

mod adaptive_annealing;
mod epsilon_greedy;
mod great_deluge;
mod logistic_annealing;
mod metropolis;
mod relative_annealing;
mod simulated_annealing;
mod tsallis;

pub use adaptive_annealing::{AdaptiveAnnealing, AdaptiveScheduler, TargetAccScheduleMode};
pub use epsilon_greedy::EpsilonGreedy;
pub use great_deluge::GreatDeluge;
pub use logistic_annealing::LogisticAnnealing;
pub use metropolis::{Metropolis, tune_temperature};
pub(crate) use metropolis::{calculate_temperature_from_acceptance_prob, gather_energy_diffs};
pub use relative_annealing::RelativeAnnealing;
pub use simulated_annealing::SimulatedAnnealing;
pub use simulated_annealing::tune_cooling_rate;
pub use tsallis::TsallisAnnealing;

/// Handler alias for [`AdaptiveAnnealing`].
pub type AdaptiveAnnealingHandler = AdaptiveAnnealing;

/// Handler alias for [`EpsilonGreedy`].
pub type EpsilonGreedyHandler = EpsilonGreedy;

/// Handler alias for [`GreatDeluge`].
pub type GreatDelugeHandler = GreatDeluge;

/// Handler alias for [`LogisticAnnealing`].
pub type LogisticAnnealingHandler = LogisticAnnealing;

/// Handler alias for [`Metropolis`].
pub type MetropolisHandler = Metropolis;

/// Handler alias for [`RelativeAnnealing`].
pub type RelativeAnnealingHandler = RelativeAnnealing;

/// Handler alias for [`SimulatedAnnealing`].
pub type SimulatedAnnealingHandler = SimulatedAnnealing;

/// Handler alias for [`TsallisAnnealing`].
pub type TsallisHandler = TsallisAnnealing;
