
struct Network
    num_layers::Integer
    sizes::Array{Integer}
    biases::Array{Array{Float64}}
    weights::Array{Any}
end

Network(sizes) = Network(length(sizes),
                         sizes,
                         rand.(view(sizes,2:length(sizes)),1),
                         [ rand( sizes[i],sizes[i+1]) for i in 1:length(sizes)-1] )

function sigmoid(z)
    return 1.0/(1.0 + exp(-z))
end

function feedforward(network, input)
    # TODO: throw out the for loop and run on GPU
    input = transpose(input)
    for i in 1:length(network.weights)
        w = network.weights[i]
        b = network.biases[i]
        input =  input * w + transpose(b)
    end
    return input
end

function SGD(network,
             training_data,
             epochs,
             mini_batch_size,
             eta,
             test_data=nothing)

    if test_data != nothing
        n_test = length(test_data)
        n = length(training_data)
    end
    for j in 1:epochs
        # TODO: Shuffle training data
        mini_batches = [ train_x[:,:,k:k+mini_batch_size-1] for k in 1:mini_batch_size:size(train_x)[3] ]
        for mini_batch in mini_batches
            # TODO create update_mini_batch
            update_mini_batch(mini_batch, eta)
        end

        if test_data != nothing
            # TODO add evaluate test_data
            @printf "Epoch %d: %d / %d" j test_data n_test
        else
            @printf "Epoch %d: complete" j 
        end
    end
end

############### MAIN #################
using MLDatasets
using ImageView
using Printf

train_x, train_y = MNIST.traindata()

print(size(train_x)) # (28,28, 60000)
sample = train_x[:,:,2]
sample = reshape(sample,(28*28))
print(size(sample)) # (28,28)

n = Network([784,2,2])
feedforward(n, sample)
mini_batch_size = 4
#### NEXT TODO: MAKE INPUT (X,Y) where X is an image and Y a label