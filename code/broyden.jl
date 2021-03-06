# PACKAGES

using LinearAlgebra: dot, norm, qr!, ldiv!
using Printf

#------------------------------------------------------------------------------

# ALGORITHM

"""
broyden!(F!, B!, xk, ε, itmax, mstlen, verbose)

# ARGUMENTS
- 'F!::Function': Objective Function.
- 'B!::Function': Approximate Jacobian matrix of F!.
- 'xk::Array(1-dimensional)': Initial point.
- 'ε::Float': Tolerance.
- 'itmax::Int64': Maximum number of iterations (optional, by default itmax = 500).
- 'mstlen::Float64': Maximum relative step length (optinal, by default mstlen = 10.0^(- 8.0)).
- 'verbose::Bool': Show information about each iteration (optional, by default verbose = False).
"""
function broyden!(F!, B!, xk, vec, ε::Float64=(10.0 ^ (-8.0)), itmax::Int64=500, mstlen::Float64=10.0^(- 8.0), verbose::Bool=true)

    itime = time()

    k = 0
    kf = 0
    kj = 0
    n = length(xk)
    ak = zeros(n)
    bk = zeros(n)
    fk = zeros(n)
    yk = zeros(n)
    w = zeros(n)
    vaux = zeros(n)
    b = zeros(n, n)
    s = Vector{Float64}[]
    u = Vector{Float64}[]

    F!(vec, w)
    kf += 1
    w .= w.^(- 1.0)
    if verbose == true
        @printf("%s\n", "-----------------------------------------------------------")
        @printf("Diagonal matrix U: %s\n", w)
    end


    F!(xk, fk)                               # Compute F!(x_0) and store in fk.
    B!(xk, b)                                # Compute B!(x_0) and store in b.
    kf += 1
    kj += 1
    normfk = norm(w.*fk, Inf)                # Compute the Inf-norm of u*fk.

    qrb = qr!(b, Val(true))           # Compute the QR fact. of B!(x_0) and store in b.

    while normfk > ε

        ldiv!(ak, qrb, fk)

        for i = 1:size(s)[1]
            ak .= ak .+ dot(s[i], ak) .* u[i]
        end

        xk .= xk .- ak

        push!(s, (.- ak))

        #for i=1:n
            #vaux[i] = max(abs(xk[i]), 10.0 ^ (- 4.0))
        #end

        #vaux .= ak ./ vaux

        #if norm(vaux, Inf) < mstlen
            #@printf("%s\n", "-----------------------------------------------------------")
            #@printf("%s\n", "The minimum of step length has been reached!")
            #break
        #end

        yk .= (.- fk)
        F!(xk, fk)
        kf += 1
        yk .= yk .+ fk

        normfk = norm(w.*fk, Inf)

        ldiv!(qrb, yk)

        for i = 1:(size(s)[1] - 1)
            yk .= yk .+ dot(s[i], yk) .* u[i]          # Solve B!(x_k)*y = y_k.
        end
        
        push!(u, ((s[size(s)[1]] .- yk) / dot(s[size(s)[1]], yk)))

        k += 1

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