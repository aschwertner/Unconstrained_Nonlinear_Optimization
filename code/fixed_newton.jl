# PACKAGES
using LinearAlgebra: norm, qr!, ldiv!
using Printf

# ALGORITHM

"""
fixednewton!(F!, J!, xk, ε, itmax, mstlen, verbose)

# ARGUMENTS
- 'F!::Function': Objective Function.
- 'J!::Function': Jacobian matrix of F!.
- 'xk::Array(1-dimensional)': Initial point.
- 'ε::Float': Tolerance.
- 'itmax::Int64': Maximum number of iterations (optional, by default itmax = 500).
- 'mstlen::Float64': Maximum relative step length (optinal, by default mstlen = 10.0^(- 8.0)).
- 'verbose::Bool': Show information about each iteration (optional, by default verbose = False).
"""
function fixednewton!(F!, J!, xk, vec, ε::Float64=(10.0 ^ (-8.0)), itmax::Int64=500, mstlen::Float64=10.0^(- 8.0), verbose::Bool=true)
    
    itime = time()
    k = 0
    kf = 0
    kj = 0
    n = length(xk)
    fk = zeros(n)
    u = zeros(n)
    vaux = zeros(n)
    jk = zeros(n, n)
    qrjk = zeros(n, n)

    F!(vec, u)
    u .= u.^(- 1.0)
    kf += 1

    if verbose == true
        @printf("%s\n", "-----------------------------------------------------------")
        @printf("Diagonal matrix U: %s\n", u)
    end

    F!(xk, fk)
    kf += 1
    normfk = norm(u.*fk, Inf)

    if verbose == true
        @printf("%s\n", "-----------------------------------------------------------")
        @printf("Iteration: %s\n", k)
        @printf("Inf.-norm of U*f(x): %s\n", normfk)
    end

    J!(xk, jk)
    kj += 1
    qrjk = qr!(jk, Val(true))

    while normfk > ε
        
        ldiv!(qrjk, fk)
        xk .= xk .- fk

        #for i=1:n
            #vaux[i] = max(abs(xk[i]), 10.0 ^ (- 4.0))
        #end
        #vaux .= fk ./ vaux

        #if norm(vaux, Inf) < mstlen
            #@printf("%s\n", "-----------------------------------------------------------")
            #@printf("%s\n", "The minimum of step length has been reached!")
            #break
        #end

        F!(xk, fk)
        normfk = norm(u.*fk, Inf)
        k += 1
        kf += 1

        if verbose == true
            @printf("%s\n", "-----------------------------------------------------------")
            @printf("Iteration: %s\n", k)
            @printf("Inf.-norm of U*f(x): %s\n", normfk)
            @printf("Relative Step length: %s\n", norm(vaux, Inf))
        end

        if k > itmax
            @printf("%s\n", "-----------------------------------------------------------")
            @printf("%s\n", "The maximum number of iterations has been reached!")
            break
        end

    end

    etime = time()

    score = (5.0 * kj) + kf

    @printf("%s\n", "-----------------------------------------------------------")
    @printf("Solution: %s\n", xk)
    @printf("Inf-norm: %10.3e | U*f Norm: %10.3e\n",  norm(fk, Inf), normfk,)
    @printf("Iterations: %8d | Time: %10.3e\n", k,  etime - itime)
    @printf("N. functions: %6d | N. Jacobian: %6d | Score: %5d\n", kf, kj, score)
    @printf("%s\n", "-----------------------------------------------------------")

end