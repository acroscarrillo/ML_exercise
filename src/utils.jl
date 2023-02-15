function label_2_vec(x::Int)
    vec_label = zeros(10)
    vec_label[x+1] = 1
    return vec_label
end

function MNIST_assess(out::Vector, exp::Vector)
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
        comp_mat[i,j] = (matrix_in[i,j]+matrix_in[i+1,j]+matrix_in[i,j+1]+matrix_in[i+1,j+1])/4
    end
    return comp_mat
end