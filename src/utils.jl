function label_2_vec(x::Int)
    vec_label = zeros(10)
    vec_label[x+1] = 1
    return Float32.(vec_label)
end

function MNIST_assess(out, exp)
    _, ind_out = findmax(out)
    _, ind_exp = findmax(exp)
    if ind_out == ind_exp
        return 1
    else
        return 0
    end
end

function MNIST_half_compress(matrix_in)
    comp_mat = zeros(14,14)
    for i = 1:14, j = 1:14
        comp_mat[i,j] = (matrix_in[2*i-1,2*j-1]+matrix_in[2*i,2*j-1]+matrix_in[2*i-1,2*j]+matrix_in[2*i,2*j])/4
    end
    return comp_mat
end