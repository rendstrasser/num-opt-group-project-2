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
class StandardizingMetaInfo:
    """
    Holds meta-info required for standardizing and destandardizing, as described in 13.41.

    Args:
        original_n: Dimensionality of x of original problem.
            Can be used to find which variables of the standardized problem represent x^+.
        indices_of_non_positive_constrained_vars: As the name suggests, contains a list of indices that
            represent which elements of x had no positivity constraints in the original problem, e.g.,
            x_4 >= 0. In this example 4 would not be part of the list, but maybe 3, if it doesn't have
            a positivity constraint. This list can be used to find which variables of the standardized problem
            represent x^-.
        slack_var_count: Count of the number of slack variables
            that need to be introduced to replace inequalities with equalities.
        real_constraints: Constraints of original problem without
            simple positivity-constraints, e.g. x_4>=0.
    """
    original_n: int
    indices_of_non_positive_constrained_vars: np.ndarray
    slack_var_count: int
    real_constraints: Sequence[LinearConstraint]

    def calc_standardized_n(self) -> int:
        """
        Calculates the dimensionality of the standardized problem.

        Returns:
            Integer for dimensionality of standardized problem.
        """
        return self.original_n + len(self.indices_of_non_positive_constrained_vars) + self.slack_var_count

    def destandardize_x(self, x: np.ndarray) -> np.ndarray:
        """
        Destandardizes x based on the original problem.

        :param x: x in terms of standardized problem, containing x+, x- and slack variables.

        Returns:
            x in terms of the original problem
        """
        n = self.original_n

        x_plus = x[:n] # take x_+ part
        x_neg = x[n:n + len(self.indices_of_non_positive_constrained_vars)]

        # subtract x_- from x_+ to get x
        x_plus[self.indices_of_non_positive_constrained_vars] -= x_neg

        return x_plus

    @classmethod
    def from_pre_standardized(cls, problem: 'LinearConstraintsProblem') -> 'StandardizingMetaInfo':
        """
        Factory method to create standardizing meta info for a problem that is already 
        standardized. This represents a default instance in the sense of a non-standardized meta info.
        
        :param problem: Problem that is already in standardized form.
        Returns:
            Standardizing meta info for problem.
        """
        return StandardizingMetaInfo(problem.n, np.empty(0, dtype=int), 0, problem.constraints)


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

    def standardized_constraints(self) -> Tuple[Sequence[LinearConstraint], StandardizingMetaInfo]:
        """Return tuple of List[standardized constraints], np.ndarray[non-pos c. indices] and number of slack variables.

        Returns:
            (List of standardized constraints, meta info required for destandardizing)
        """
        standardizing_meta_info = self._extract_standardizing_meta_info()

        # Convert non-positivity-constraints to standard form.
        standard_constraints = self._to_standard_form(standardizing_meta_info)

        return standard_constraints, standardizing_meta_info

    @staticmethod
    def _to_standard_form(standardizing_meta_info: StandardizingMetaInfo) -> Sequence[LinearConstraint]:
        """
        Converts the given constraints to standard form as described in 13.41.

        Returns:
            Sequence of standardized constraints.
        """

        # unpack meta info for easy access
        indices_of_non_positive_constrained_vars = standardizing_meta_info.indices_of_non_positive_constrained_vars
        slack_var_count = standardizing_meta_info.slack_var_count
        real_constraints = standardizing_meta_info.real_constraints

        standard_constraints = []
        slack_var_idx = 0
        for constraint in real_constraints:

            a = constraint.c.a
            b = constraint.c.b

            # Multiply `a` and `b` with -1 if we have a greater than or equal sign
            # to first convert the GE constraint to a LE constraint.
            if constraint.equation_type == EquationType.GE:
                a *= -1
                b *= -1

            # All non-positivity-constraint indices need to be added with a x^- case,
            # as we require all variables to have positivity constraints at the end.
            neg_a = -a[indices_of_non_positive_constrained_vars]

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
                c=LinearCallable(a=new_a, b=b),
                equation_type=EquationType.EQ))

        return standard_constraints

    def _extract_standardizing_meta_info(self) -> StandardizingMetaInfo:
        """
        Extracts meta info required for standardizing and destandardizing.

        Returns:
            Meta info required for standardizing/destandardizing
        """

        real_constraints = []
        indices_of_non_positive_constrained_vars = np.arange(self.n)  # for start, assume none is positivity-constrained
        slack_var_count = 0

        for constraint in self.constraints:
            idx = constraint.positivity_constraint_idx()
            if idx is None:
                # Per inequality, one slack variable is needed.
                slack_var_count += constraint.equation_type != EquationType.EQ

                real_constraints.append(constraint)
            else:
                # consider as positivity-constrained, remove from non-pos.-constr. list
                indices_of_non_positive_constrained_vars = indices_of_non_positive_constrained_vars[
                    indices_of_non_positive_constrained_vars != idx]

        return StandardizingMetaInfo(self.n, indices_of_non_positive_constrained_vars, slack_var_count, real_constraints)
