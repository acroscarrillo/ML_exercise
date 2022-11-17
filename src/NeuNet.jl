struct NeuNet
    name::String
    layer_arch::Vector{Int}
    weights::Vector{Matrix{Float32}}
    biases::Vector{Vector{Float32}}
end

#to do: initialise weights and biases with the Catalan's algorithm
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
        error(string("Input data of size ",length(input)," is incompatible with input layer of size ",length(net.layer_arch[0])))
    end
    previous_layer = input
    for i in 1:length(net.weights)
        previous_layer = activation.( transpose(net.weights[i])*previous_layer + net.biases[i] )
    end
    return previous_layer
end

function sigmoid(x::Real)
    return  1.0 / (1.0 + exp(-x))
end

function cost(real_output, expected_output)
    
end