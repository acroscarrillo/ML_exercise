using LinearAlgebra

struct NeuNet
    name::String
    layer_arch::Vector{Int}
    weights::Vector{Matrix{Float32}}
    biases::Vector{Vector{Float32}}
end

#to do: initialise weights and biases with the catalan's algorithm
function NeuNet(name::String, layer_arch::Vector{Int})
    if length(layer_arch)<2
        error(string("Neural Net must have at least two neuron layers but ",length(layer_arch)," where given."))
	end
    weights = [randn(Float32, (layer_arch[i], layer_arch[i+1])) for i in 1:(length(layer_arch)-1)]
    biases = [randn(Float32, (layer_arch[i])) for i in 1:length(layer_arch)]
	return NeuNet(name, layer_arch, weights, biases)
end

function propagate(net::NeuNet, input::Vector, activation::Function)
    if length(input) != net.layer_arch[1]
        error( string( "Input data of size ",length(input)," is incompatible with input layer of size ",net.layer_arch[1] ) )
    end

    layer_actv  = [ zeros(net.layer_arch[i]) for i in 1:length(net.layer_arch) ]
    layer_actv[1] = activation.( input + net.biases[1] )
    for i in 1:length(net.layer_arch)
        layer_actv[i+1] = activation.( transpose(net.weights[i])*layer_actv[i] + net.biases[i+1] )
    end
    return layer_actv
end

function sigmoid(x::Float32)
    return  1.0 / (1.0 + exp(-x))
end

function back_prop(net::NeuNet, input::Vector, output_e::Vector, activation::Function, learning_rate::Float32)
    layer_actv = propagate(net,input,activation)
    grad = qcost(layer_actv[end], output_e)
    for i in length(net.layer_arch):1
        grad *= 
        net.biases[i] = net.biases[i] - learning_rate*(grad)
    end
end

#this is technically the derivative of the quadratic cost function
function qcost(real_output, expected_output)
    return norm(real_output-expected_output)
end