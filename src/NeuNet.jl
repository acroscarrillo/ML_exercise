using LinearAlgebra
using BenchmarkTools
using Metal

struct NeuNet{W,B}
	name::String
	layer_arch::Vector{Int}
	weights::W
	biases::B
end

function mtl(net::NeuNet)
	return NeuNet(net.name, net.layer_arch, MtlArray.(net.weights), MtlArray.(net.biases))
end

function NeuNet(name::String, layer_arch::Vector{Int})
    # if length(layer_arch)<2
    #     error(string("Neural Net must have at least two neuron layers but ",length(layer_arch)," where given."))
	# end
    # weights = [randn(Float32, (layer_arch[i], layer_arch[i+1])) for i=1:(length(layer_arch)-1)] 
    weights = [randn(Float32, (layer_arch[i], layer_arch[i+1]))/layer_arch[i] for i=1:(length(layer_arch)-1)] 

    # biases = [randn(Float32, (layer_arch[i+1])) for i=1:(length(layer_arch)-1)]
    biases = [zeros(Float32, layer_arch[i+1]) for i=1:(length(layer_arch)-1)]
	return NeuNet(name, layer_arch, weights, biases)
end

function propagate(net::NeuNet, input, activation::Function)
    layer_actv  = [ fill!(similar(input, (net.layer_arch[i]),size(input)[1]), 0.0) for i=1:length(net.layer_arch) ]
    # layer_actv[1] = activation.(input)
    layer_actv[1] = input
    for i=1:length(net.layer_arch)-1
        layer_actv[i+1] = activation.( transpose(net.weights[i])*layer_actv[i] .+ net.biases[i] )
    end
    return layer_actv
end

function sigmoid(x::T) where T
    return  one(T) / (one(T) + exp(-x))
end

function sigmoid_d(x) 
    return  (1-sigmoid(x))*sigmoid(x)
end

function back_prop(net::NeuNet, input, output_e, activation::Function, activation_d::Function,learning_rate::Real)
    layer_actv = propagate(net,input,activation)
    layer_actv_D = [sigmoid_d.(x) for x in layer_actv]

    # grad = transpose( layer_actv[end] - output_e )
    grad1 = layer_actv[end] .- output_e 
    cost = grad1
    # Del_b = transpose(grad1) .* layer_actv_D[end]
    # net.biases[end]  = net.biases[end] - learning_rate*(Del_b)

    Del_b1 = grad1 .* layer_actv_D[end]
    batch_dim = size(Del_b1)[2]
    net.biases[end]  = net.biases[end] .- vec(sum(learning_rate*(Del_b1),dims=2))/batch_dim

    # Del_w = transpose(Del_b1).*layer_actv[end-1]
    # net.weights[end] = net.weights[end] - learning_rate*(Del_w)
    Del_w_ten = [layer_actv[end-1][:,i].*transpose(Del_b1[:,i]) for i=1:size(Del_b1, 2)]
    net.weights[end] = net.weights[end] .- learning_rate*(sum(Del_w_ten,dims=1)[1])/batch_dim
    
    grad = [transpose(grad1[:,i]) for i in 1:size(grad1,2)]

    for i=length(net.layer_arch)-2:-1:1
        # grad = grad*(layer_actv_D[i+2].*transpose(net.weights[i+1]))
        tmp = [layer_actv_D[i+2][:,j].*transpose(net.weights[i+1]) for j=1:size(layer_actv_D[i+2],2)]
        
        grad .= grad .* tmp

        # Del_b = transpose(grad).*layer_actv_D[i+1]
        # net.biases[i]  = net.biases[i] - learning_rate*(Del_b)

        Del_b = [transpose(grad[j]) .* layer_actv_D[i+1][:,j] for j=1:length(grad)]
        
        net.biases[i]  = net.biases[i] - learning_rate.*sum(Del_b)/batch_dim

        # Del_w = transpose(Del_b).*layer_actv[i]
        Del_w = [transpose(Del_b[j]).*layer_actv[i][:,j] for j=1:length(Del_b)]
        
        net.weights[i] = net.weights[i] - learning_rate.*sum(Del_w)/batch_dim
        # net.weights[i] = net.weights[i] - learning_rate*(Del_w)
    end
    return norm(cost)
end

#this is technically the derivative of the quadratic cost function
function qcost(real_output, expected_output)
    return norm(real_output-expected_output)
end

function Base.show(io::IO, net::NeuNet)
    print(io,"NeuNet type with name '", net.name, "' and architecture ", net.layer_arch,".","\n")
    print(io,"Weights:","\n")
    print(io,net.weights,"\n")
    print(io,"Biases:","\n")
    print(io,net.biases,"\n")
end