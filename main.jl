using Random
using MLDatasets
using ImageView
using Printf
using Random

mutable struct Network
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
function eval_single(sample)
    mxval, mxindex =findmax(feedforward(n, sample))
    print("Digits predicted is\n")
    print(mxindex[2]-1)
    print("\n")
    print("I'm this certain that it's correct\n")
    print(mxval)
    print("\n")
end
function sigmoid_prime(z)
    return sigmoid(z)*(1-sigmoid(z))
end

function feedforward(network, input, keep_grad=false)
    # TODO: Optimize
    output = transpose( reshape(input, (size(input)[1]*size(input)[2]) ))
    zs = []
    activations = []
    append!(activations,[output])
    for i in 1:length(network.weights)
        # adding 1 for biases
        output=hcat(output,1)
        w = network.weights[i]
        output = output * w

        if keep_grad
            append!(zs, [output])
        end

        output = sigmoid.(output)

        if keep_grad
            append!(activations, [output])
        end
    end
    if keep_grad
        return zs,activations
    else
        return output
    end
end

function cost(output,labels)
    return output-labels
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
        for it in 1:length(mini_batches_labels)
            mini_batch = mini_batches[it]
            labels = mini_batches_labels[it]
            network = update_mini_batch(network,mini_batch,labels,0.01)
        end

        print( "###############")
        print("EPOCH DONE")
        print(j)
        print( "###############")
end
end
function update_mini_batch(network::Network, data, labels, η)
    ∇w = copy(network.weights)
#    ∇w .= 0.0
    for it in 1:(size(data)[3])
        img = @view data[:,:,it]
        # make one hot encoded
        label = zeros(10)
        label[labels[it]+1]=1
        ∇w = backprop(network,img,label)
    end
    network.weights = network.weights - η * ∇w
    return network
end

function backprop(network::Network, x, y)
    ∇w = copy(network.weights)
#    ∇w .= 0.0
    zs, activations = feedforward(network,x,true)
    δ = cost(reshape(activations[length(activations)],10),y)
    ∇w[length(∇w)] = transpose(hcat(δ * activations[length(activations)-1],δ))

    for l in 1:(network.num_layers-2)
        z =  zs[length(zs)-l]
        sp = sigmoid_prime.(z)
        δ = (network.weights[length(zs)-l+1] * δ) #.* sp
        δ = @view δ[1:length(δ)-1]
        ∇w[length(∇w)-l] = transpose(hcat(δ * activations[length(activations)-l-1],δ))
    end
    return ∇w
end
############### MAIN ################# Random.seed!(123);
train_x, train_y = MNIST.traindata()
test_x,test_y = MNIST.testdata()
sample = test_x[:,:,10]
correct_label= test_y[10]
imshow(sample)
print(correct_label)
eval_single(sample)
#n = Network([784,30,10])
feedforward(n, sample,false)
reshape(n.weights,(2,))
SGD(n,train_x,train_y,1,8,0.01)

#### NEXT TODO: MAKE INPUT (X,Y) where X is an image and Y a labeligof