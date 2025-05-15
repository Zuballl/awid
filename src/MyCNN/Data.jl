module Data

using JSON
using Random
using Statistics
using LinearAlgebra

export load_data, create_data_loader, split_data, normalize_data, augment_data

function load_data(file_path::String)
    data = JSON.parsefile(file_path)
    
    # Konwersja danych do odpowiedniego formatu
    # Dane są już podzielone na treningowe i testowe
    X_train = Float32.(hcat(data["X_train"]...))
    y_train = Int.(data["y_train"])
    X_test = Float32.(hcat(data["X_test"]...))
    y_test = Int.(data["y_test"])
    
    # Normalizacja danych osobno dla treningowych i testowych
    X_train = normalize_data(X_train)
    X_test = normalize_data(X_test)
    
    # Sprawdź wymiary danych
    println("X_train shape: ", size(X_train))
    println("X_test shape: ", size(X_test))
    
    # Reshape dla CNN (batch_size, channels, height, width)
    # Dla danych tekstowych: (batch_size, channels, sequence_length, embedding_dim)
    n_train = size(X_train, 2)
    n_test = size(X_test, 2)
    embedding_dim = size(X_train, 1)
    
    # Przekształcamy dane do formatu (batch_size, 1, 1, embedding_dim)
    X_train = reshape(X_train, n_train, 1, 1, embedding_dim)
    X_test = reshape(X_test, n_test, 1, 1, embedding_dim)
    
    # Łączymy dane treningowe i testowe
    X = cat(X_train, X_test, dims=1)
    y = vcat(y_train, y_test)
    
    return X, y
end

function normalize_data(X::Array{Float32, 2})
    # Normalizacja do zakresu [0, 1]
    X = (X .- minimum(X)) ./ (maximum(X) - minimum(X))
    return X
end

function augment_data(X::Array{Float32, 4}, y::Vector{Int})
    batch_size = size(X, 1)
    augmented_X = copy(X)
    augmented_y = copy(y)
    
    for i in 1:batch_size
        # Losowe przesunięcie
        if rand() < 0.5
            shift_x = rand(-2:2)
            shift_y = rand(-2:2)
            augmented_X[i, :, :, :] = circshift(X[i, :, :, :], (0, 0, shift_x, shift_y))
        end
        
        # Losowe obrócenie
        if rand() < 0.5
            angle = rand(-15:15)
            augmented_X[i, :, :, :] = imrotate(X[i, :, :, :], angle)
        end
        
        # Losowe skalowanie
        if rand() < 0.5
            scale = rand(0.9:1.1)
            augmented_X[i, :, :, :] = imresize(X[i, :, :, :], scale)
        end
    end
    
    return augmented_X, augmented_y
end

function create_data_loader(X::Array{Float32, 4}, y::Vector{Int}; batch_size=32, shuffle=true, augment=false)
    n_samples = size(X, 1)
    indices = shuffle ? randperm(n_samples) : 1:n_samples
    
    # Tworzymy kanał do przekazywania batchy
    ch = Channel{Tuple{Array{Float32, 4}, Vector{Int}}}(Inf)
    
    # Funkcja produkująca dane
    function producer()
        for i in 1:batch_size:n_samples
            batch_indices = indices[i:min(i+batch_size-1, n_samples)]
            batch_X = X[batch_indices, :, :, :]
            batch_y = y[batch_indices]
            
            if augment
                batch_X, batch_y = augment_data(batch_X, batch_y)
            end
            
            put!(ch, (batch_X, batch_y))
        end
    end
    
    # Uruchamiamy producenta w tle
    @async producer()
    
    return ch
end

function split_data(X::Array{Float32, 4}, y::Vector{Int}; train_ratio=0.8, val_ratio=0.1)
    n_samples = size(X, 1)
    indices = randperm(n_samples)
    
    train_size = floor(Int, train_ratio * n_samples)
    val_size = floor(Int, val_ratio * n_samples)
    
    train_indices = indices[1:train_size]
    val_indices = indices[train_size+1:train_size+val_size]
    test_indices = indices[train_size+val_size+1:end]
    
    X_train = X[train_indices, :, :, :]
    y_train = y[train_indices]
    
    X_val = X[val_indices, :, :, :]
    y_val = y[val_indices]
    
    X_test = X[test_indices, :, :, :]
    y_test = y[test_indices]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
end

# Funkcje pomocnicze do augmentacji
function imrotate(img::Array{Float32, 3}, angle::Int)
    # Konwersja kąta na radiany
    θ = deg2rad(angle)
    
    # Macierz rotacji
    R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    
    # Wymiary obrazu
    _, h, w = size(img)
    center = [h/2, w/2]
    
    # Nowy obraz
    rotated = zeros(Float32, size(img))
    
    for i in 1:h
        for j in 1:w
            # Przesunięcie do środka, rotacja i powrót
            p = [i, j] .- center
            p_rot = R * p
            p_rot = p_rot .+ center
            
            # Interpolacja najbliższego sąsiada
            i_rot, j_rot = round.(Int, p_rot)
            
            if 1 ≤ i_rot ≤ h && 1 ≤ j_rot ≤ w
                rotated[:, i, j] = img[:, i_rot, j_rot]
            end
        end
    end
    
    return rotated
end

function imresize(img::Array{Float32, 3}, scale::Float32)
    # Wymiary obrazu
    c, h, w = size(img)
    new_h = round(Int, h * scale)
    new_w = round(Int, w * scale)
    
    # Nowy obraz
    resized = zeros(Float32, c, new_h, new_w)
    
    # Skalowanie
    for i in 1:new_h
        for j in 1:new_w
            # Mapowanie współrzędnych
            src_i = round(Int, i / scale)
            src_j = round(Int, j / scale)
            
            if 1 ≤ src_i ≤ h && 1 ≤ src_j ≤ w
                resized[:, i, j] = img[:, src_i, src_j]
            end
        end
    end
    
    return resized
end

end # module Data 