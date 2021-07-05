using Random
using MLDatasets
using ImageView
using Printf
using Random

mutable struct Network
    num_layers::Integer
    sizes::Array{Integer}
    weights::Array{Array{Float64}}
    biases::Array{Array{Float64}}
end

Network(sizes) = Network(length(sizes),
                         sizes,
                         [randn( sizes[i] ,sizes[i+1]) for i in 1:length(sizes)-1],
                         [randn(sizes[i+1]) for i in 1:length(sizes)-1 ])

function sigmoid(z)
    return 1.0/(1.0 + exp(-z))
end

function sigmoid_prime(z)
    return sigmoid(z)*(1-sigmoid(z))
end

function cost(output,labels)
    return output-labels
end

function eval_single(sample)
    mxval, mxindex = findmax(feedforward(n, sample))
    print("Digits predicted is\n")
    print(mxindex[2]-1)
    print("\n")
    print("I'm this certain that it's correct\n")
    print(mxval)
    print("\n")
end

function feedforward(network::Network, input, keep_grad=false)

    output = transpose( reshape(input, (size(input)[1]*size(input)[2]) ))
    zs = []
    activations = []
    append!(activations,[output])
    for i in 1:length(network.weights)

        w = network.weights[i]
        b = reshape(network.biases[i], (1,size(network.biases[i])[1]))
        output = output * w + b

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

function evaluate(network,test_data,test_labels) 
    cor=0
    for it in 1:size(test_data)[3]
        sample = @view test_data[:,:,it]
        label = test_labels[it]
        mxval, mxindex =findmax(feedforward(network, sample))
        if mxindex[2]-1 == label
            cor+=1
        end
    end
    return cor/size(test_data)[3]*100
end

function SGD(network::Network,
             train_data,
             train_labels,
             epochs,
             mini_batch_size,
             η,
             test_data=nothing,
             test_labels=nothing)

    for j in 1:epochs

        # Shuffle data
        shuffle_ids = shuffle(1: (size(train_data)[3]))
        train_data = train_data[:,:,shuffle_ids]
        train_labels = train_labels[shuffle_ids]

        mini_batches = []
        mini_batches_labels = []
        
        for k in 1:mini_batch_size:size(train_data)[3]
            push!(mini_batches, @view train_data[:,:,k:k+mini_batch_size-1])
            push!(mini_batches_labels, @view train_labels[k:k+mini_batch_size-1])
        end

        for it in 1:length(mini_batches_labels)
            mini_batch = mini_batches[it]
            labels = mini_batches_labels[it]
            network = update_mini_batch(network,mini_batch,labels,η)
        end

        println("### EPOCH DONE $j ###")
        if test_data!==nothing
            println(evaluate(network,train_data,train_labels))
            println(evaluate(network,test_data,test_labels))
        end
        println()

    end

end

function update_mini_batch(network::Network, data, labels, η)
    ∇w = [zeros( (network.sizes[i],network.sizes[i+1]) ) for i in 1:length(network.sizes)-1]
    ∇b = [zeros(network.sizes[i+1]) for i in 1:length(network.sizes)-1]
    for it in 1:(size(data)[3])
        img = @view data[:,:,it]

        # make labels one hot encoded
        label = zeros(10)
        label[labels[it]+1]=1

        temp_b, temp_w = backprop(network,img,label)
        ∇w = ∇w + temp_w
        ∇b = ∇b + temp_b
    end
    network.weights = network.weights - η * ∇w
    network.biases = network.biases - η * ∇b
    return network
end

function backprop(network::Network, x, y)
    ## TODO: FIX BACKPROP ADD SIGMOID PRIME
    ∇w = [zeros( (network.sizes[i],network.sizes[i+1]) ) for i in 1:length(network.sizes)-1]
    ∇b = [ zeros((network.sizes[i+1])) for i in 1:length(network.sizes)-1]

    zs, activations = feedforward(network,x,true)
    sp = sigmoid_prime.(zs[length(zs)])
    δ = cost(reshape(activations[length(activations)],10),y)  .* reshape(sp,size(sp)[2],1)

    δ = vec(δ)
    ∇b[length(∇b)] = δ
    ∇w[length(∇w)] = transpose(δ * activations[length(activations)-1])
    for l in 1:(network.num_layers-2)
        z =  zs[length(zs)-l]
        sp = sigmoid_prime.(z)
        δ = (network.weights[length(zs)-l+1] * δ) .* reshape(sp, (size(sp)[2],1))
        δ = vec(δ)
        ∇b[length(∇b)-l] = δ
        ∇w[length(∇w)-l] = transpose(δ * activations[length(activations)-l-1])
    end
    return ∇b,∇w
end

############### MAIN #################
Random.seed!(123);
train_x, train_y = MNIST.traindata()
test_x,test_y = MNIST.testdata()
n = Network([784,20,10])
SGD(n,train_x,train_y,30,10,0.3,test_x,test_y)
#zero([1,1,1])
#TODO: Hyperoptimize, test why when you increase the number of neurons accuracy drops