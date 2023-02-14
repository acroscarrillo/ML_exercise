using MLDatasets: MNIST
using ProgressBars
include("src/src.jl")

function label_2_vec(x::Int)
    vec_label = zeros(10)
    vec_label[x+1] = 1
    return vec_label
end

training_set = MNIST(:train)
testing_set = MNIST(:test)

MNIST_net = NeuNet("MNIST network", [28*28,100,10])

alpha = .05
runs = 100

#train
for n in ProgressBar(1:runs)
    for i in 1:60000
        data_in, label_e = vec(training_set[i].features), label_2_vec(training_set[i].targets)
        back_prop(MNIST_net, data_in, label_e, sigmoid, sigmoid_d, alpha/sqrt(n))
    end
end

function assess(out, exp)
    _, ind_out = findmax(out)
    _, ind_exp = findmax(exp)
    if ind_out == ind_exp
        return 1
    else
        return 0
    end
end
#test
net_acc = 0
for i in ProgressBar(1:10000)
    data_in, label_e = vec(testing_set[i].features), label_2_vec(testing_set[i].targets)
    global net_acc += assess(propagate(MNIST_net,data_in,sigmoid)[end],label_e)/10000
end

display("Net accurancy:")
display(net_acc)

