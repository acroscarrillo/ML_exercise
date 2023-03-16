using MLDatasets: MNIST
using ProgressBars
include("src/src.jl")

training_set = MNIST(:train)
testing_set = MNIST(:test)

N_train = length(training_set)
N_test = length(testing_set)

vect_train = [vec(MNIST_half_compress(training_set[i].features)) for i=1:N_train]
vect_test = [vec(MNIST_half_compress(testing_set[i].features)) for i=1:N_test]

MNIST_net = mtl( NeuNet("MNIST network", [14*14,25,10]) )

alpha = Float32(0.05)
runs = 100
batch_size = 6000

train_ten_d = [ MtlArray( Float32.( hcat( vect_train[i:i+batch_size-1]... ) ))  for i=1:batch_size:N_train]
train_ten_l = [ MtlArray( Float32.( transpose(permutedims( hcat( label_2_vec.( training_set[i:i+batch_size-1].targets )... ))) ))  for i=1:batch_size:N_train] 

# train, run loop outside
cost_array = zeros(runs)
# for run in ProgressBar(1:runs)
for run=1:runs
    for i=1:Int32(N_train/batch_size)
        data_in, label_e =  train_ten_d[i],  train_ten_l[i]
        global cost = back_prop(MNIST_net, data_in, label_e, sigmoid, sigmoid_d, alpha)
        # display("Progress = "*string(100*run/runs)*"%")
        # display("Cost="*string(cost))
    end
    display("Progress = "*string(100*run/runs)*"%")
    display("Cost="*string(cost))
end

# test
net_acc = 0
for i in ProgressBar(1:N_test)
    data_in, label_e =  reshape(vect_test[i],(196,1)),  reshape(label_2_vec(testing_set[i].targets),(10,1))
    global net_acc += MNIST_assess(propagate(MNIST_net,data_in,sigmoid)[end],label_e) / N_test
end

println("Net accurancy: "*string(net_acc))
println("With params: α = "*string(alpha)*", runs = "*string(runs)*", batch_size = "*string(batch_size))