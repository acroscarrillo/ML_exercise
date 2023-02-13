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
    weights = [randn(Float32, (layer_arch[i], layer_arch[i+1])) for i=1:(length(layer_arch)-1)]
    biases = [randn(Float32, (layer_arch[i+1])) for i=1:(length(layer_arch)-1)]
	return NeuNet(name, layer_arch, weights, biases)
end

function propagate(net::NeuNet, input::Vector, activation::Function)
    if length(input) != net.layer_arch[1]
        error( string( "Input data of size ",length(input)," is incompatible with input layer of size ",net.layer_arch[1],"." ) )
    end
    layer_actv  = [ zeros(net.layer_arch[i]) for i=1:length(net.layer_arch) ]
    layer_actv[1] = activation.(input)
    for i=1:length(net.layer_arch)-1
        layer_actv[i+1] = activation.( transpose(net.weights[i])*layer_actv[i] + net.biases[i] )
    end
    return layer_actv
end

function sigmoid(x::Real)
    return  1.0 / (1.0 + exp(-x))
end

function sigmoid_d(x::Real)
    return  (1-sigmoid(x))*sigmoid(x)
end

function delta_rec(n,m)
    aux_mat = zeros(n,m)
    aux_mat[diagind(aux_mat)] .= 1
    return aux_mat
end

function back_prop(net::NeuNet, input::Vector, output_e::Vector, activation::Function, activation_d::Function,learning_rate::Real)
    layer_actv = propagate(net,input,activation)
    layer_actv_D = [sigmoid_d.(x) for x in layer_actv]

    grad = transpose( layer_actv[end]- output_e )

    Del_b = grad*( layer_actv_D[end].*delta_rec( net.layer_arch[end], net.layer_arch[end] ) )
    net.biases[end]  = net.biases[end] - learning_rate*(transpose(Del_b))
    Del_w = kron(layer_actv[end-1], Del_b)
    net.weights[end] = net.weights[end] - learning_rate*(Del_w)
    
    for i=length(net.layer_arch)-2:-1:1
        grad = grad*(layer_actv_D[i+2].*transpose(net.weights[i+1]))

        Del_b = grad*(( layer_actv_D[i+1].*delta_rec( size( layer_actv_D[i+1] )...,size( layer_actv_D[i+1] )...)  ) )
        net.biases[i]  = net.biases[i] - learning_rate*((transpose(Del_b)))
        Del_w = kron(layer_actv[i], Del_b)
        net.weights[i] = net.weights[i] - learning_rate*(Del_w)
    end
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