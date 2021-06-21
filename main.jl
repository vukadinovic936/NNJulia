
using MLDatasets
using ImageView
using Printf
using Random

struct Network
    num_layers::Integer
    sizes::Array{Integer}
    weights::Array{Array{Float64}}
end

Network(sizes) = Network(length(sizes),
                         sizes,
                         [ rand( sizes[i]+1 ,sizes[i+1]) for i in 1:length(sizes)-1] )

function sigmoid(z)
    return 1.0/(1.0 + exp(-z))
end

function sigmoid_prime(z)
    return sigmoid(z)*(1-sigmoid(z))
end

function feedforward(network, input, keep_grad=false)
    # TODO: Optimize
    output = transpose( reshape(input, (size(input)[1]*size(input)[2]) ))
    zs = []
    activations = []
    for i in 1:length(network.weights)
        # adding 1 for biases
        output=hcat(output,1)
        w = network.weights[i]
        output = output * w

        if keep_grad
            append!(zs,output)
        end

        output = sigmoid.(output)

        if keep_grad
            append!(activations, output)
        end
    end
    if keep_grad
        return zs,activations
    else
        return output
    end
end

function cost(output,labels)
    return (output-labels)
end
    
function SGD(network::Network,
             train_data,
             train_labels,
             epochs,
             mini_batch_size,
             η,
             test_data=nothing,
             test_labels=nothing)

    if test_data != nothing
        n_test = length(test_data)
        n = length(train_data)
    end
    for j in 1:epochs
        # TODO: Shuffle training data
        mini_batches = []
        mini_batches_labels = []
        for k in 1:mini_batch_size:size(train_data)[3]
            push!(mini_batches, @view train_data[:,:,k:k+mini_batch_size-1])
            push!(mini_batches_labels, @view train_labels[k:k+mini_batch_size-1])
        end

        for mini_batch in mini_batches
            # TODO create update_mini_batch
            network = update_mini_batch(network, mini_batch, mini_batch_labels, η)
        end
#
#        if test_data != nothing
#            # TODO add evaluate test_data
#            @printf "Epoch %d: %d / %d" j test_data n_test
#        else
#            @printf "Epoch %d: complete" j 
#        end
    end
end
function update_mini_batch(network::Network, data, labels, η)
    ∇w = [zero(network.weights)]
    for (img,label) in zip(data,labels)
        ∇w += backprop(network,img,label)
    end
    network.weights = network.weights - η * ∇w
    return network
end

function backprop(network::Network, x, y)

    ∇w = zero(network.weights)
    zs, activations = feedforward(network,x,true)
    δ = cost((@view activations[1:length(activations)-1]),y) * sigmoid_prime(@view zs[1:length(zs)-1])
    ∇w = δ * transpose(@view activations[1:length(activations)-2])
    for l in 2:network.num_layers
        z = @view zs[1:length(zs)-l]
        sp = sigmoid_prime(z)
        δ = transpose(@view network.weights[1:length(zs)-l+1])
        ∇w = 
    end
end
############### MAIN #################
Random.seed!(123);
train_x, train_y = MNIST.traindata()
sample = train_x[:,:,1]
n = Network([784,2,2])
print(size(n.weights[1]))
feedforward(n, sample)
SGD(n,train_x,train_y,1,8,0.01)
#### NEXT TODO: MAKE INPUT (X,Y) where X is an image and Y a labeligof

for (i,j) in zip(1:10,10:20)
    print(i,j)
end
a = [1,2,3,4]
@view a[:-1]
