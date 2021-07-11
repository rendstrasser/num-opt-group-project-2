"""
MinimizationProblem, relevant functions and implemented methods.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Optional, Tuple

import numpy as np

from shared.constraints import Constraint, LinearConstraint, LinearCallable, EquationType, combine_linear
from shared.gradient_approximation import gradient_approximation, hessian_approximation


@dataclass
class MinimizationProblem:
    """
    Data class containing all necessary information of a minimization problem to support
    unconstrained optimization.

    Args:
        f (Callable): The function (objective) we are trying to minimize.
        n (int): Dimensionality of input x for function
        constraints (Sequence[Constraint]): Sequence of constraints.
        solution (np.ndarray): The solution to the minimization problem. May be None if unknown.
        x0 (np.ndarray): The starting point for the minimization procedure. May be None if unknown.
    """
    f: Callable[[np.ndarray], float]
    n: int
    constraints: Sequence[Constraint]
    x0: Optional[np.ndarray]
    solution: Optional[np.ndarray]

    # --- Objective function methods ---

    def active_set_at(self, x: np.ndarray, as_equalities: bool) -> List[Constraint]:
        """Return list of active constraints at point x.

        Args:
            as_equalities (bool): Whether to return constraints as equalities or not.
            x: Point to evaluate constraints at.

        Returns:
            List[Constraint]: List of active constraints.
        """
        return [c.as_equality() if as_equalities else c for c in self.constraints if c.is_active(x)]

    def calc_f_at(self, x: np.ndarray) -> float:
        return self.f(x)

    def calc_gradient_at(self, x: np.ndarray) -> np.ndarray:
        return gradient_approximation(self.f, x)

    def calc_hessian_at(self, x: np.ndarray) -> np.ndarray:
        return hessian_approximation(self.f, x)

    # --- Constraint methods ---

    def calc_constraints_at(self, x: np.ndarray) -> np.ndarray:
        return np.array([c(x) for c in self.constraints])

    def calc_constraint_at(self, i: int, x: np.ndarray) -> float:
        return self.constraints[i](x)

    def calc_constraints_jacobian_at(self, x: np.ndarray) -> np.ndarray:
        return np.array([gradient_approximation(c.c, x) for c in self.constraints])

    def calc_constraint_gradient_at(self, i: int, x: np.ndarray) -> np.ndarray:
        return gradient_approximation(self.constraints[i], x)

    # --- Lagrangian methods ---

    def calc_lagrangian_at(self, x, lambda_) -> float:
        assert len(lambda_) == len(self.constraints)

        result = self.calc_f_at(x)

        for i, lambda_i in enumerate(lambda_):
            result -= lambda_i * self.calc_constraint_at(i, x)

        return result

    # gradient wrt x
    def calc_lagrangian_gradient_at(self, x, lambda_) -> np.ndarray:

        def lagrangian(x_):
            return self.calc_lagrangian_at(x_, lambda_)

        return gradient_approximation(lagrangian, x)

    # hessian wrt x
    def calc_lagrangian_hessian_at(self, x, lambda_) -> np.ndarray:

        def lagrangian(x_):
            return self.calc_lagrangian_at(x_, lambda_)

        return hessian_approximation(lagrangian, x)


@dataclass
class LinearConstraintsProblem(MinimizationProblem):
    """
    Data class containing all necessary information of a minimization problem to support
    unconstrained optimization.

    Holds linear constraints in the form of Ax=b
    """
    constraints: Sequence[LinearConstraint]
    A: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self):
        self.A, self.b = combine_linear([constraint.c for constraint in self.constraints])

    def standardized_constraints(self) -> Tuple[Sequence[LinearConstraint], np.ndarray, int]:
        """Return tuple of List[standardized constraints], np.ndarray[non-pos c. indices] and number of slack variables.

        Returns:
            (List of standardized constraints, np.ndarray of non-positively constrained indices and number of slack variables.)
        """
        non_positive_constrained_idx, real_constraints, slack_var_count = self._non_positively_constrained()

        # Convert non-positivity-constraints to standard form.
        standard_constraints = self._to_standard_form(non_positive_constrained_idx, real_constraints, slack_var_count)

        return standard_constraints, non_positive_constrained_idx, slack_var_count

    @staticmethod
    def _to_standard_form(non_positive_constrained_idx, real_constraints, slack_var_count):
        # TODO: Doc

        standard_constraints = []
        slack_var_idx = 0
        for constraint in real_constraints:

            a = constraint.c.a

            # Multiply `a` with -1 if we have a greater than or equal sign.
            if constraint.equation_type == EquationType.GE:
                a *= -1

            # All non-positivity-constraint indices need to be added with a x^- case,
            # as we require all variables to have positivity constraints at the end.
            neg_a = -a[non_positive_constrained_idx]

            # Create constraint coefficients e for slack variable, 
            # if we need one to replace an inequality.
            if constraint.equation_type != EquationType.EQ:
                e = np.eye(slack_var_count)[slack_var_idx]
                slack_var_idx += 1
            else:
                e = np.zeros(slack_var_count)

            # Bring to standard form (13.41) by assuming x+, x-, z,
            # as shown in page 357.
            new_a = np.concatenate((a, neg_a, e))

            standard_constraints.append(LinearConstraint(
                c=LinearCallable(a=new_a, b=constraint.c.b),
                equation_type=EquationType.EQ))

        return standard_constraints

    def _non_positively_constrained(self):
        # TODO: Doc

        real_constraints = []
        non_positive_constrained_idx = np.arange(self.n)  # for start, assume none is positivity-constrained
        slack_var_count = 0

        for i, constraint in enumerate(self.constraints):
            idx = constraint.positivity_constraint_idx()
            if idx is None:

                # Per inequality, one slack variable is needed.
                slack_var_count += constraint.equation_type != EquationType.EQ

                real_constraints.append(constraint)
            else:
                # consider as positivity-constrained, remove from non-pos.-constr. list
                non_positive_constrained_idx = non_positive_constrained_idx[non_positive_constrained_idx != idx]

        return non_positive_constrained_idx, real_constraints, slack_var_count
