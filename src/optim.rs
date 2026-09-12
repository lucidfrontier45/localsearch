//! Optimization Algorithm

mod adaptive_annealing;
mod base;
mod epsilon_greedy;
mod generic;
mod great_deluge;
mod handlers;
mod hill_climbing;
mod logistic_annealing;
mod metropolis;
mod parallel_tempering;
mod population_annealing;
mod random;
mod relative_annealing;
mod simulated_annealing;
mod tabu_search;
mod transition;
mod tsallis;

pub use adaptive_annealing::AdaptiveAnnealingOptimizer;
pub use base::LocalSearchOptimizer;
pub use epsilon_greedy::EpsilonGreedyOptimizer;
pub use generic::{GenericLocalSearchOptimizer, StepResult};
pub use great_deluge::GreatDelugeOptimizer;
pub use handlers::{
    AdaptiveAnnealing, AdaptiveAnnealingHandler, AdaptiveScheduler, EpsilonGreedy,
    EpsilonGreedyHandler, GreatDeluge, GreatDelugeHandler, LogisticAnnealing,
    LogisticAnnealingHandler, Metropolis, MetropolisHandler, RelativeAnnealing,
    RelativeAnnealingHandler, SimulatedAnnealing, SimulatedAnnealingHandler, TargetAccScheduleMode,
    TsallisAnnealing, TsallisHandler, tune_cooling_rate, tune_temperature,
};
pub(crate) use handlers::{calculate_temperature_from_acceptance_prob, gather_energy_diffs};
pub use hill_climbing::HillClimbingOptimizer;
pub use logistic_annealing::LogisticAnnealingOptimizer;
pub use metropolis::MetropolisOptimizer;
pub use parallel_tempering::ParallelTemperingOptimizer;
pub use population_annealing::PopulationAnnealingOptimizer;
pub use random::RandomSearchOptimizer;
pub use relative_annealing::RelativeAnnealingOptimizer;
pub use simulated_annealing::SimulatedAnnealingOptimizer;
pub use tabu_search::{TabuList, TabuSearchOptimizer};
pub use transition::{TransitionHandler, UpdateCtx};
pub use tsallis::TsallisRelativeAnnealingOptimizer;
