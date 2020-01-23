# Unconstrained Nonlinear Optimization

## Introduction

This repository contains implementations of some algorithms for unconstrained nonlinear optimization, namely:

- Newton-Raphson;
- Fixed-Newton;
- Fixed-Newton with Restarts;
- Broyden;
- Colum-Updating Method (COLUM);
- Inverse  Column-Updating  Method (ICUM).

The file *nonlinear_equations_library.jl* is also available, containing the implementation of some functions of the set of problems proposed by Moré-Garbow-Hillstrom.

The algorithms were written in the *Julia* programming language, version *1.0.4*.

## How to use

To use any of the methods, just compile their respective code and call the desired function, informing all the necessary parameters. If you have any questions, type ```?"Function_name"´´´ after compiling the code.

## References

> Martínez, J. M. (2000). Practical quasi-Newton methods for solving nonlinear systems. In Journal of Computational and Applied Mathematics}, 124(1-2), 97-121.

> Martínez, J. M., and Santos, S. A. (1995). Métodos computacionais de otimização. In Colóquio Brasileiro de Matemática}, Books, 20.

> Moré, J. J., Garbow, B. S., and Hillstrom, K. E. (1978). Testing unconstrained optimization software (No. ANL-AMD-TM-324). Argonne National Lab., IL (USA).


