import numpy as np

def final_printout(x_0, x_optimal, x_appr, iter_count, f, grad, tolerance=1e-7, **kwargs):
    """
    Parameters
    --------------------------------------------------------------------------------------------------------------
    x_0: numpy 1D array, corresponds to initial point
    x_optimal: numpy 1D array, corresponds to optimal point, which you know, or have solved analytically
    x_appr: numpy 1D array, corresponds to approximated point, which your algorithm returned
    iter_count: number of executed iterations for minimization procedure to find x_appr
    --------------------------------------------------------------------------------------------------------------
    f: function which takes 2 inputs: x (initial, optimal, or approximated)
                                      **args
       Function f returns a scalar output.
    --------------------------------------------------------------------------------------------------------------
    grad: function which takes 2 inputs: x (initial, optimal, or approximated), 
                                         args (which are submitted, because you might need
                                              to call f(x,**args) inside your gradient function implementation). 
          Function grad approximates gradient at given point and returns a 1d np array.
    --------------------------------------------------------------------------------------------------------------
    args: dictionary, additional (except of x) arguments to function f
    tolerance: float number, absolute tolerance, precision to which, you compare optimal and approximated solution.
    """
    
    print(f'Initial x is :\t\t{x_0}')
    print(f'Optimal x is :\t\t{x_optimal}')
    print(f'Approximated x is :\t{x_appr}')
    print(f'Is close verification: \t{np.isclose(x_appr,x_optimal,atol=tolerance)}\n')
    f_opt = f(x_optimal,**kwargs)
    f_appr = f(x_appr,**kwargs)
    print(f'Function value in optimal point:\t{f_opt}')
    print(f'Function value in approximated point:   {f_appr}')
    print(f'Is close verification:\t{np.isclose(f_opt,f_appr,atol=tolerance)}\n')
    grad_optimal = grad(x_optimal,**kwargs)
    print(f'Gradient approximation in optimal point is:\n{grad_optimal}\n')
    grad_appr = grad(x_appr,**kwargs)
    print(f'Gradient approximation in approximated point is:\n{grad_appr}\n')
    print(f'Is close verification:\n{np.isclose(grad_appr,grad_optimal,atol=tolerance)}\n')
    print(f'Number of iterations required: {iter_count}')