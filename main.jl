
struct Network
    num_layers::Integer
    sizes::Array{Integer}
    biases::Array{Array{Float64}}
    weights::Array{Any}
end

Network(sizes) = Network(length(sizes),
                         sizes,
                         rand.(sizes,1),
                         [ rand( sizes[i],sizes[i+1]) for i in 1:length(sizes)-1] )

function sigmoid(z)
    return 1.0/(1.0_+ exp(-z))
end

function feedforward(network, input)
    for i in 1:length(network.weights)
        w = network.weights[i]
        b = network.biases[i]
        input =  w*input +b
    end
    return input
end

#function SGD(network, training_data, epohcs, mini_batch_size, eta, test_data=None)
    #if test_data
        #n_test = length(test_data)
        #n = len(training_data)
    #end
    #for j in range(epohcs)
        ## random shuffle training data
        #mini_batches = [view(training_data[])]
    #end
#end

n = Network([2,2,2])
feedforward(n,[1,1])
