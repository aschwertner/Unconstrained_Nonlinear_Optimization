using LinearAlgebra

#------------------------------------------------------------------------------

function rosenbrock!(x, f)

    f[1] = (10.0 * (x[2] - (x[1] ^ 2.0)))
    f[2] = (1.0 - x[1])

end

function rosenbrockJ!(x, j)

    j[1, 1] = (- 20.0 * x[1])
    j[1, 2] = 10.0
    j[2, 1] = -1.0
    j[2, 2] = 0.0

end

#------------------------------------------------------------------------------

function powellbadlyscaled!(x, f)

    f[1] = (10.0 ^ (4.0) * x[1] * x[2] - 1.0)
    f[2] = (exp(- x[1]) + exp(- x[2]) - 1.0001)

end

function powellbadlyscaledJ!(x, j)

    j[1, 1] = (10.0 ^ (4.0) * x[2])
    j[1, 2] = (10.0 ^ (4.0) * x[1])
    j[2, 1] = (- exp(- x[1]))
    j[2, 2] = (- exp(- x[2]))

end

#------------------------------------------------------------------------------

function helicalvalley!(x, f)

    if x[1] > 0
        f[1] = (10.0 * (x[3] - 10.0 * ((1 / (2 * pi)) * atan(x[2] / x[1]))))
    end
    if x[1] < 0
        f[1] = (10.0 * (x[3] - 10.0 * ((1 / (2 * pi)) * atan(x[2] / x[1]) +
                0.5)))
    end
    f[2] = (10.0 * (((x[1] ^ 2.0) + (x[2] ^ 2.0)) ^ (1/2) - 1))
    f[3] = x[3]

end

function helicalvalleyJ!(x, j)

    if x[1] > 0
        j[1, 1] = ((50.0 * x[2]) / (pi * ((x[1] ^ 2.0) + (x[2] ^ 2.0))))
        j[1, 2] = ((- 50.0 * x[1]) / (pi * ((x[1] ^ 2.0) + (x[2] ^ 2.0))))
        j[1, 3] = 10.0
    end
    if x[1] < 0
        j[1, 1] = ((50.0 * x[2]) / (pi * ((x[1] ^ 2.0) + (x[2] ^ 2.0))))
        j[1, 2] = ((- 50.0 * x[1]) / (pi * ((x[1] ^ 2.0) + (x[2] ^ 2.0))))
        j[1, 3] = 10.0
    end
    j[2, 1] = ((10.0 * x[1]) / (((x[1] ^ 2.0) + (x[2] ^ 2.0)) ^ (1/2)))
    j[2, 2] = ((10.0 * x[2]) / (((x[1] ^ 2.0) + (x[2] ^ 2.0)) ^ (1/2)))
    j[2, 3] = 0.0
    j[3, 1] = 0.0
    j[3, 2] = 0.0
    j[3, 3] = 1.0
    
end

#------------------------------------------------------------------------------

function powellsingular!(x, f)

    f[1] = x[1] + 10.0 * x[2]
    f[2] = (5.0 ^ (0.5)) * (x[3] - x[4])
    f[3] = (x[2] - 2.0 * x[3]) ^ (2.0)
    f[4] = (10.0 ^ (0.5)) * (x[1] - x[4]) ^ (2.0)
    
end

function powellsingularJ!(x, j)

    j[1, 1] = 1.0
    j[1, 2] = 10.0
    j[1, 3] = 0.0
    j[1, 4] = 0.0

    j[2, 1] = 0.0
    j[2, 2] = 0.0
    j[2, 3] = 5.0 ^ (0.5)
    j[2, 4] = - (5.0 ^ (0.5))

    j[3, 1] = 0.0
    j[3, 2] = 2.0 * x[2] - 4.0 * x[3]
    j[3, 3] = 8.0 * x[3] - 4.0 * x[2]
    j[3, 4] = 0.0

    j[4, 1] = 2.0 * (10.0 ^ (0.5)) * (x[1] - x[4])
    j[4, 2] = 0.0
    j[4, 3] = 0.0
    j[4, 4] = - 2.0 * (10.0 ^ (0.5)) * (x[1] - x[4])
    
end

#------------------------------------------------------------------------------

function watson!(x, f)

    aux = 0.0
    aux1 = 0.0
    aux2 = 0.0

    for i = 1:29
        aux = (i / 29.0)
        aux1 = 0.0
        aux2 = 0.0
        for j = 2:31
            aux1 = aux1 + (j - 1.0) * x[j] * aux ^ (j - 2.0)
            aux2 = aux2 + x[j] * aux ^ (j - 1.0)
        end
        aux2 = aux2 + x[1]
        f[i] = aux1 - aux2 ^ (2.0) - 1.0
    end

    f[30] = x[1]
    f[31] = x[2] - x[1] ^ (2.0) - 1.0

end

function watsonJ!(x, j)

    aux = 0.0
    aux1 = 0.0
    aux2 = 0.0

    for i=1:29
        aux = (i / 29.0)
        for k=1:31
            aux1 = 0.0
            aux2 = 0.0
            if i == 1
                for l=1:31
                    aux1 = aux1 + x[l] * aux ^ (l - 1.0)
                end

                j[1, k] = - 2 * aux ^ (k - 1.0) * aux1

            else
                for l=1:31
                    aux2 = aux2 + x[l] * aux ^ (l - 1.0)
                end
                
                j[i, k] = (k - 1.0) * aux ^ (k - 2.0) - 2 * aux ^
                          (k - 1.0) * aux2

            end
        end
    end

    j[30, :] = zeros(31)
    j[30, 1] = 1.0

    j[31, :] = zeros(31)
    j[31, 1] = - 2.0 * x[1]
    j[31, 2] = 1.0

end

#------------------------------------------------------------------------------

function trigonometric!(x, f)
    m = length(f)
    aux = 0.0

    for i=1:m
        aux = aux + cos(x[i])
    end

    for i=1:m
        f[i] = m - aux + i * (1.0 - cos(x[i])) - sin(x[i])
    end

end

function trigonometricJ!(x, j)
    m = size(j)[1]

    for i=1:m
        for k=1:m
            if k == i
                j[i, k] = (i + 1.0) * sin(x[k]) - cos(x[k])
            else
                j[i, k] = sin(x[k])
            end
        end
    end
end

#------------------------------------------------------------------------------

function brownalmostlinear!(x, f)
    m = length(f)

    for i=1:(m-1)
        f[i] = x[i] + sum(x) - (m + 1.0)
    end

    f[m] = prod(x) - 1.0

end

function brownalmostlinearJ!(x, j)
    m = size(j)[1]

    j[:, :] = ones(m, m)

    for i=1:(m-1)
        j[i, i] = 2.0
    end

    for i=1:m
        j[m, i] = prod(x[Array(1:m).!=i])
    end
end

#------------------------------------------------------------------------------

function discreteboundary!(x, f)
    m = length(f)
    c1 = 1.0 / (m + 1.0)
    c2 = (c1 ^ (2.0)) / 2.0
    aux = 0.0

    f[1] = 2 * x[1] - x[2] + c2 * (x[1] + c1 + 1.0) ^ 3.0
    
    for i=2:(m - 1)
        aux = i * c1
        f[i] = 2 * x[i] - x[i - 1] - x[i + 1] + c2 * (x[i] + aux + 1.0) ^ 3.0
    end

    f[m] = 2 * x[m] - x[m - 1] + c2 * (x[m] + m * c1 + 1.0) ^ 3.0

end

function discreteboundaryJ!(x, j)
    m = size(j)[1]
    c1 = 1.0 / (m + 1.0)
    c2 = (3.0 * (c1 ^ (2.0))) / 2.0
    
    j[:,:] = zeros(m, m)
    j[1, 1] = 2.0 + c2 * (x[1] + c1 + 1.0) ^ (2.0)
    j[1, 2] = - 1.0

    for i=2:(m-1)
        aux = i * c1
        j[i, i] = 2.0 + c2 * (x[i] + i * c1 + 1) ^ (2.0)
        j[i, (i - 1)] = - 1.0
        j[i, (i + 1)] = - 1.0
    end

    j[m, m] = 2.0 + c2 * (x[1] + m* c1 + 1.0) ^ (2.0)
    j[m, (m - 1)] = - 1.0
end

#------------------------------------------------------------------------------

function discreteintegral!(x, f)
    m = length(f)
    c1 = 1.0 / (m + 1.0)
    c2 = c1 / 2.0
    aux = 0.0
    aux1 = 0.0
    aux2 = 0.0
    aux3 = 0.0

    for i=1:m
        aux = i * c1
        aux1 = 0.0
        aux2 = 0.0
        aux3 = 0.0

        for j = 1:i
            aux1 = j * c1
            aux2 = aux2 + aux1 * (x[j] + aux1 + 1) ^ (3.0)
        end

        for j = (i + 1):m
            aux1 = j * c1
            aux3 = aux3 + (1.0 - aux1) * (x[j] + aux1 + 1.0) ^ (3.0)
        end

        f[i] = x[i] + c2 * ((1.0 - aux) * aux2 + aux * aux3)

    end

end

function discreteintegralJ!(x, j)
    m = size(j)[1]
    c1 = 1.0 / (m + 1.0)
    c2 = 1.5 * c1
    aux = 0.0
    aux1 = 0.0

    j[:, :] = zeros(m, m)
    for i=1:m
        aux = i * c1
        j[i, i] = 1.0 + c2 * (1.0 - aux) * aux * (x[i] + aux + 1.0) ^ (2.0)

        for k=1:(i - 1)
            aux1 = k * c1
            j[i, k] = c2 * (1.0 - aux) * aux1 * (x[k] + aux1 + 1) ^ (2.0)
        end

        for k = (i + 1):m
            aux1 = k * c1
            j[i, k] = c2 * aux * (1.0 - aux1) * (x[k] + aux1 + 1) ^ (2.0)
        end

    end

end

#------------------------------------------------------------------------------

function broydentridiagonal!(x, f)
    m = length(f)
    f[1] = (3.0 - 2.0 * x[1]) * x[1] - 2.0 * x[2] + 1.0
    f[m] = (3.0 - 2.0 * x[m]) * x[m] - x[m - 1] + 1.0
    for i=2:(m-1)
        f[i] = (3.0 - 2.0 * x[i]) * x[i] - x[i - 1] - 2.0 * x[i + 1] + 1.0
    end
end

function broydentridiagonalJ!(x, j)
    m = size(j)[1]
    j[:, :] = zeros(m, m)

    for i=1:m
        j[i, i] = 3.0 - 4.0 * x[i]
    end

    for i=1:(m-1)
        j[i, i + 1] = - 2.0
    end
    
    for i=2:m
        j[i, i - 1] = - 1.0
    end

end

#------------------------------------------------------------------------------

function broydenbanded!(x, f)
    m = length(f)
    aux = 0.0

    for i=1:m
        aux = 0.0

        for j=max(1, i - 5):min(m, i + 1)
            if j != i
                aux = aux + x[j] * (1.0 + x[j])
            end
        end

        f[i] = x[i] * (2.0 + 5.0 * x[i] ^ (2.0)) + 1.0 - aux

    end
end

function broydenbandedJ!(x, j)
    m = size(j)[1]
    j[:, :] = zeros(m, m)
    
    for i=1:m
        j[i, i] = 15.0 * x[i] ^ (2.0) + 2.0

        for k=max(1, i - 5):min(m, i + 1)
            if k != i
                j[i, k] = 2.0 * x[k] + 1.0
            end
        end
    end
end

#------------------------------------------------------------------------------

function chebyquad!(x, f)
    m = length(f)
    aux = 0.0
    
    function chebyshev(t, n)
        return cos(n * acos(2 * t - 1))
    end

    for i = 1:m
        if mod(i,2) == 1
            aux = 0.0
            for j = 1:m
                aux = aux + chebyshev(x[j], i)
            end
            f[i] = (1.0 / m ) * aux
        end

        if mod(i,2) == 0
            aux = 0.0
            for j = 1:m
                aux = aux + chebyshev(x[j], i)
            end
            f[i] = (1.0 / m ) * aux + (1 / (i ^ (2.0) - 1.0))
        end
    end
end

function chebyquadJ!(x, j)
    m = size(j)[1]
    j[:, :] = zeros(m, m)

    function chebJ(t, n)
        return (n / (m * sqrt(- t ^ (2.0) + t))) * sin(n * acos(2 * t - 1.0))
    end

    for i = 1:m
       for k = 1:m
            j[i, k] = chebJ(x[k], i)
       end
    end

end

#------------------------------------------------------------------------------

function chebyquadmodified!(x, f)
    m = length(f)
    aux = 0.0
    
    function chebypol(t, n)
        aux = 0.0

        if abs(t) <= 1.0
            aux = cos(n * acos(t))
        elseif t > 1.0
            aux = cosh(n * acosh(t))
        else
            aux = ((- 1.0) ^ (n)) * cosh(n * acosh(- t))
        end

        return aux
    end

    for i = 1:m
        aux = 0.0
        for j = 1:m
            aux = aux + chebypol(x[j], i)
        end
        f[i] = (1.0 / m ) * aux
    end

end

function chebyquadmodifiedJ!(x, j)
    m = size(j)[1]
    j[:, :] = zeros(m, m)

    function chebpolJ(t, n)
        aux = 0.0

        if abs(t) <= 1.0
            aux = (n / (m * sqrt(1.0 - t ^ (2.0)))) * sin(n * acos(t))
        elseif t > 1.0
            aux = (n / (m * sqrt(t ^ (2.0) - 1.0))) * sinh(n * acosh(t))
        else
            aux = (((- 1.0) ^ (n + 1.0) * n) / (m * sqrt(- t - 1.0) * sqrt(1.0 - t))) * sinh(n * acosh(- t))
        end
        return aux
    end

    for i = 1:m
       for k = 1:m
            j[i, k] = chebpolJ(x[k], i)
       end
    end

end
#------------------------------------------------------------------------------

#function chebyquad!(x, f)
    #m = length(f)
    #aux = 0.0

    #function cheb(x, l)
    
        #if l == 1
            #x = 2.0 * x - 1.0
    
        #elseif l == 2
            #x = 8.0 * x ^ (2.0) - 8.0 * x + 1.0
    
        #elseif l == 3
            #x = 32.0 * x ^ (3.0) - 48.0 * x ^ (2.0) + 18.0 * x - 1.0
        
        #elseif l == 4
            #x = 128.0 * x ^ (4.0) - 256.0 * x ^ (3.0) + 160.0 * x ^ (2.0) - 32.0 * x + 1.0
    
        #elseif l == 5
            #x = 512.0 * x ^ (5.0) - 1280.0 * x ^ (4.0) + 1120.0 * x ^ (3.0) - 400.0 * x ^ (2.0) + 50.0 * x - 1.0
        
        #elseif l == 6
            #x = 2048.0 * x ^ (6.0) - 6144.0 * x ^ (5.0) + 6912.0 * x ^ (4.0) - 3584.0 * x ^ (3.0) + 840.0 * x ^ (2.0) - 72.0 * x + 1.0
        
        #elseif l == 7
            #x = 8192.0 * x ^ (7.0) - 28672.0 * x ^ (6.0) + 39424.0 * x ^ (5.0) - 26880.0 * x ^ (4.0) + 9408.0 * x ^ (3.0) - 1568.0 * x ^ (2.0) + 98.0 * x - 1.0
        
        #else
            #print("The dimension of x must be less or equal to 7.\n")
                    
        #end
                
    #end
    
    #for i=1:m
        #aux = 0.0

        #for j=1:m
            #aux = aux + cheb(x[j], i)
        #end

        #if mod(i, 2) == 0
            #f[i] = (1.0 / m) * aux + (1.0 / (i ^ (2.0) - 1.0))
        #else
            #f[i] = (1.0 / m) * aux
        #end
    #end
#end

#function chebyquadJ!(x, j)
    #m = size(j)[1]

    #function chebJ(x, l)
    
        #if l == 1
            #x = 2.0
    
        #elseif l == 2
            #x = 16.0 * x - 8.0
    
       #elseif l == 3
            #x = 96.0 * x ^ (2.0) - 96.0 * x + 18.0
        
        #elseif l == 4
            #x = 32.0 * (16.0 * x ^ (3.0) - 24.0 * x ^ (2.0) + 10.0 * x - 1.0)
    
        #elseif l == 5
            #x = 10.0 * (256.0 * x ^ (4.0) - 512.0 * x ^ (3.0) + 336.0 * x ^ (2.0) - 8.0 * x + 5.0)
        
        #elseif l == 6
            #x = 24.0 * (512.0 * x ^ (5.0) - 1280.0 * x ^ (4.0) + 1152.0 * x ^ (3.0) - 448.0 * x ^ (2.0) + 70.0 * x - 3.0)
        
        #elseif l == 7
            #x = 14.0 * (4097.0 * x ^ (6.0) - 12288.0 * x ^ (5.0) + 14080.0 * x ^ (4.0) - 7680.0 * x ^ (3.0) + 2016.0 * x ^ (2.0) - 224.0 * x + 7.0)
        
        #else
            #print("The dimension of x must be less or equal to 7.\n")
                    
        #end
                
    #end

    #for i=1:m
        #for k=1:m
            #j[i, k] = (1.0 / m) * chebJ(x[k], i)
        #end

    #end

#end