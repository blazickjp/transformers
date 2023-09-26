using Flux
using Random
using LinearAlgebra
using Random: rand


# Set hyperparameters
const BATCH_SIZE = 64
const BLOCK_SIZE = 256
const N_EMBED = 384
const N_HEAD = 6
const N_LAYER = 6
const DROPOUT = 0.2
const LEARNING_RATE = 3e-4
const MAX_ITERS = 5000
const EVAL_INTERVAL = 500
const EVAL_ITERS = 200

Random.seed!(1337)

# Reading Data
function read_data(path::String)
    return read(path, String)
end

text = read_data("data/shakespeare.txt")
vocab = unique(text)
vocab_size = length(vocab)

encoder = Dict(char => i for (i, char) in enumerate(vocab))
decoder = Dict(i => char for (i, char) in enumerate(vocab))
println(encoder['a'])
function encode(x)
    return [encoder[char] for char in x]
end

function decode(x)
    return join([decoder[i] for i in x])
end

data = encode(text)
n = Int(floor(0.9 * length(data)))
train_data = data[1:n]
val_data = data[n+1:end]

# Implement your data loading and preprocessing here
# For now, assume data is loaded into a variable called 'data', and vocab size in 'vocab_size'

mutable struct Head
    key::Dense
    query::Dense
    value::Dense
    dropout::Dropout
end

function Head(d_model, d_key)
    key = Dense(d_model, d_key)
    query = Dense(d_model, d_key)
    value = Dense(d_model, d_key)
    dropout = Dropout(DROPOUT)
    return Head(key, query, value, dropout)
end

function (head::Head)(x)
    # Implement attention logic here similar to PyTorch
end

mutable struct MultiHeadAttention
    heads::Vector{Head}
    proj::Dense
    dropout::Dropout
end

function MultiHeadAttention(d_model, num_heads, d_key)
    heads = [Head(d_model, d_key) for _ = 1:num_heads]
    proj = Dense(d_key * num_heads, d_model)
    dropout = Dropout(DROPOUT)
    return MultiHeadAttention(heads, proj, dropout)
end

function (mha::MultiHeadAttention)(x)
    # Implement multi-head attention logic here
end

mutable struct FeedForward
    net::Chain
end

function FeedForward(d_model)
    net = Chain(
        Dense(d_model, 4 * d_model),
        Flux.relu,
        Dense(4 * d_model, d_model),
        Dropout(DROPOUT)
    )
    return FeedForward(net)
end

function (ff::FeedForward)(x)
    ff.net(x)
end

mutable struct Block
    sa::MultiHeadAttention
    ffwd::FeedForward
    ln1::LayerNorm
    ln2::LayerNorm
end

function Block(d_model, num_heads, d_key)
    sa = MultiHeadAttention(d_model, num_heads, d_key)
    ffwd = FeedForward(d_model)
    ln1 = LayerNorm(d_model)
    ln2 = LayerNorm(d_model)
    return Block(sa, ffwd, ln1, ln2)
end

function (block::Block)(x)
    x = x .+ block.sa(LayerNorm(block.ln1(x)))
    x = x .+ block.ffwd(LayerNorm(block.ln2(x)))
    return x
end

mutable struct GPTLanguageModel
    token_embedding_table::Embedding
    position_embedding_table::Embedding
    blocks::Chain
    ln_f::LayerNorm
    lm_head::Dense
end

function GPTLanguageModel(vocab_size, d_model, n_layer, n_head, d_key)
    token_embedding_table = Embedding(vocab_size, d_model)
    position_embedding_table = Embedding(BLOCK_SIZE, d_model)
    blocks = Chain([Block(d_model, n_head, d_key) for _ = 1:n_layer]...)
    ln_f = LayerNorm(d_model)
    lm_head = Dense(d_model, vocab_size)
    return GPTLanguageModel(token_embedding_table, position_embedding_table, blocks, ln_f, lm_head)
end

function (model::GPTLanguageModel)(x, y)
    # Implement forward logic similar to PyTorch code
end

# Initialize the model
d_key = N_EMBED ÷ N_HEAD
model = GPTLanguageModel(vocab_size, N_EMBED, N_LAYER, N_HEAD, d_key)

# Create optimizer
opt = Flux.Optimiser(ADAM(LEARNING_RATE))

# Implement training and evaluation loop

function get_batch(split, train_data::Array{Int, 1}, val_data::Array{Int,1}, batch_size::Int, block_size::Int)
    # Length of the data
    if split == "train"
        data = train_data
    else
        data = val_data
    end
    n = length(data)
    
    # Randomly select starting indices
    ix = rand(1:(n - block_size), batch_size)
    
    # Create empty arrays for the batch
    X = zeros(Int, batch_size, block_size)
    Y = zeros(Int, batch_size, block_size)
    
    # Populate X and Y
    for i in 1:batch_size
        X[i, :] = data[ix[i] : ix[i] + block_size - 1]
        Y[i, :] = data[ix[i] + 1 : ix[i] + block_size]
    end
    
    return X, Y
end


# Function to estimate loss (assume you implement this)
function estimate_loss(data, model)
    # implement logic to evaluate model on some evaluation data
    # return the estimated loss
end

# Your training loop
for iter in 1:MAX_ITERS
    # Get a mini-batch of data (you need to implement get_batch)
    X, Y = get_batch("train", BATCH_SIZE)
    
    # Forward pass and compute the loss
    # Assume that your model's forward function returns both logits and loss
    function loss_fn()
        logits, loss = model(X, Y)
        return loss
    end
    
    # Backward pass and optimization
    gs = Flux.gradient(params(model)) do
        loss_fn()
    end
    Flux.update!(opt, params(model), gs)
    
    # Periodic evaluation
    if iter % EVAL_INTERVAL == 0 || iter == MAX_ITERS
        train_loss = estimate_loss(train_data, model)
        val_loss = estimate_loss(val_data, model)
        println("Step $iter: Train loss = $train_loss, Val loss = $val_loss")
    end
end