using MyProject.MyMLP: load_data, train_test_split, Chain, Dense, ReLU, Softmax, predict
using MyProject.MyMLP: SGD, Momentum, Adam, train!
using Statistics: mean, std

X, Y = load_data()        

X_mean = mean(X, dims=1)
X_std = std(X, dims=1)
X = (X .- X_mean) ./ (X_std .+ 1e-8)

X_train, Y_train, X_test, Y_test = train_test_split(X, Y; test_ratio=0.2, seed=42)
X_train = Matrix(X_train')
X_test = Matrix(X_test')
Y_train = Matrix(Y_train')
Y_test = Matrix(Y_test')

@info "Data dimensions" input_shape=size(X_train) target_shape=size(Y_train)

model = Chain(
    Dense(4, 8), 
    ReLU(),
    Dense(8, 3),
    Softmax()
)

opt = Adam(0.01)  

@info "Starting training..."
train!(model, X_train, Y_train, opt; epochs=100, batchsize=16)  

ŷtest = predict(model, X_test)       
y_test = [argmax(Y_test[:,i]) for i in 1:size(Y_test,2)]  
acc = mean(ŷtest .== y_test)
@info "Test accuracy" accuracy=acc