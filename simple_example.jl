include("src/src.jl")

my_net = NeuNet("net name", [2,5,3])
display("Initial net:")
display(my_net)
display("Initial propagation with input [.1, .6]:")
display( propagate(my_net,[.1,.6],sigmoid) )
display("Train to aim at output [.6,.2,.9]...")
for i=1:1000
    back_prop(my_net,[.1,.6],[.6,.2,.9],sigmoid,sigmoid_d,1)
end
display("Propagation after learning:")
display( propagate(my_net,[.1,.6],sigmoid) )
display("Updated net:")
display(my_net)