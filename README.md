# Treatment Effect Estimation Examples

I use this repo for keeping various simulations and use-cases relating to estimators of (Conditional) Average Treatment Effect ((C)ATE) 

Relevant references:
- Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects. Electronic Journal of Statistics, 17(2), 3008-3049.
- Curth, A., & Van der Schaar, M. (2021, March). Nonparametric estimation of heterogeneous treatment effects: From theory to learning algorithms. In International Conference on Artificial Intelligence and Statistics (pp. 1810-1818). PMLR.
- Rosenblum, M., & Van Der Laan, M. J. (2010). Simple, efficient estimators of treatment effects in randomized trials using generalized linear models to leverage baseline variables. The international journal of biostatistics, 6(1), 13.
- Van der Laan, M. J., & Rose, S. (2011). Targeted learning: causal inference for observational and experimental data (Vol. 4). New York: Springer.

Examples:

- Ex1: Simple demonstration of CATE estimation using DR and IPW estimators
- Ex2: Impact of practical violations of positivity in CATE estimation using DR and IPW estimators
- Ex3: Demonstration of TMLE estimator for ATE in a case where treatment is independent of the covariates
- Ex4: Demonstration of TMLE estimator for ATE for binary outcomes and treatment in a case with observed confounders
