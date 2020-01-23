# PACKAGES
using LinearAlgebra: norm, qr!, ldiv!
using Printf

# ALGORITHM

"""
newtonraphson!(F!, J!, xk, ε, itmax, mstlen, verbose)

# ARGUMENTS
- 'F!::Function': Objective Function.
- 'J!::Function': Jacobian matrix of F!.
- 'xk::Array(1-dimensional)': Initial point.
- 'ε::Float': Tolerance (optional, by default ε = 10.0 ^ (-8.0)).
- 'itmax::Int64': Maximum number of iterations (optional, by default itmax = 500).
- 'mstlen::Float64': Maximum relative step length (optinal, by default mstlen = 10.0^(- 8.0)).
- 'verbose::Bool': Show information about each iteration (optional, by default verbose = False).
"""
function newtonraphson!(F!, J!, xk, vec, ε::Float64=(10.0 ^ (-8.0)), itmax::Int64=500, mstlen::Float64=10.0^(- 8.0), verbose::Bool=true)
    
    itime = time()
    k = 0                                # Initialize the counter k.
    kf = 0
    kj = 0
    n = length(xk)                       # Compute the dimension of vector xk.
    fk = zeros(n)                        # Initialize the vector fk.
    vaux = zeros(n)
    u = zeros(n)
    jk = zeros(n, n)                     # Initialize the matrix jk.

    F!(vec, u)
    kf += 1
    u .= u.^(- 1.0)
    if verbose == true
        @printf("%s\n", "-----------------------------------------------------------")
        @printf("Diagonal matrix U: %s\n", u)
    end

    F!(xk, fk)                           # Compute F(xk) and store in fk.
    J!(xk, jk)                           # Compute J(xk) and store in jk.
    normfk = norm(u.*fk, Inf)            # Compute the Inf-norm of u*fk.
    kf += 1
    kj += 1

    if verbose == true
        @printf("%s\n", "-----------------------------------------------------------")
        @printf("Iteration: %s\n", k)
        @printf("Inf.-norm of U*f(x): %s\n", normfk)
    end

    while normfk > ε
        qrjk = qr!(jk, Val(true))      # Compute the QR factorization of jk and store in jk.
        ldiv!(qrjk, fk)                # Solve (QR)*dk = fk for dk and store the solution in fk.
        xk .= xk .- fk                 # Updates the current point.
      
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
        J!(xk, jk)
        normfk = norm(u.*fk, Inf)
        k += 1                         # Updates the counter.
        kf += 1
        kj += 1

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