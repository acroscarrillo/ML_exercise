using MLDatasets: MNIST
using ProgressBars
using CUDA
include("src/src.jl")

training_set = MNIST(:train)
testing_set = MNIST(:test)

MNIST_net = NeuNet("MNIST network", [14*14,50,10])

alpha = .001
runs = 100
# batches = 

# train

for n in ProgressBar(1:runs)
    for i in 1:60000
        data_in, label_e =  CuArray(vec( MNIST_half_compress(training_set[i].features) )),  CuArray(label_2_vec( training_set[i].targets ))
        back_prop(MNIST_net, data_in, label_e, sigmoid, sigmoid_d, alpha)
    end
end

# test
net_acc = 0
for i in ProgressBar(1:10000)
    data_in, label_e =  CuArray(vec( MNIST_half_compress(training_set[i].features) )),  CuArray(label_2_vec(testing_set[i].targets))
    global net_acc += MNIST_assess(propagate(MNIST_net,data_in,sigmoid)[end],label_e)/10000
end

display("Net accurancy:")
display(net_acc)


