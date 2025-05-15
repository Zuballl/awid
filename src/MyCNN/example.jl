# using MyProject
# using MyProject.MyCNN
# using Plots
# 
# # Ładowanie danych
# println("Ładowanie danych...")
# include("Data.jl")
# X, y = Data.load_data("../../../imdb_dataset_prepared.json")
# 
# # Podział danych
# println("Podział danych...")
# (X_train, y_train), (X_val, y_val), (X_test, y_test) = Data.split_data(X, y)
# 
# # Tworzenie data loaderów
# println("Tworzenie data loaderów...")
# train_loader = Data.create_data_loader(X_train, y_train, batch_size=32, augment=true)
# val_loader = Data.create_data_loader(X_val, y_val, batch_size=32)
# test_loader = Data.create_data_loader(X_test, y_test, batch_size=32)
# 
# # Stwórz model
# println("Tworzenie modelu...")
# include("Models.jl")
# model = Models.CNN()
# 
# # Prosty optymalizator SGD
# learning_rate = 0.001f0
# 
# # Trenuj model
# println("Trenowanie modelu...")
# include("Training.jl")
# train_losses = Float32[]
# train_accs = Float32[]
# val_losses = Float32[]
# val_accs = Float32[]
# 
# for epoch in 1:20
#     # Training
#     train_loss, train_acc = Training.train_epoch(model, train_loader, learning_rate, Training.cross_entropy_loss)
#     push!(train_losses, train_loss)
#     push!(train_accs, train_acc)
#     
#     # Validation
#     val_loss, val_acc = Training.evaluate(model, val_loader, Training.cross_entropy_loss)
#     push!(val_losses, val_loss)
#     push!(val_accs, val_acc)
#     
#     println("Epoch $epoch:")
#     println("  Train Loss: $train_loss, Train Acc: $train_acc")
#     println("  Val Loss: $val_loss, Val Acc: $val_acc")
# end
# 
# # Wizualizuj historię treningu
# println("Wizualizacja historii treningu...")
# include("Utils.jl")
# Utils.plot_training_history(train_losses, train_accs, val_losses, val_accs)
# savefig("training_history.png")
# 
# # Ewaluuj model
# println("Ewaluacja modelu...")
# metrics = Utils.evaluate_model(model, X_test, y_test)
# println("Test accuracy: $(metrics["accuracy"])")
# println("F1-score: $(metrics["f1_score"])")
# 
# # Wizualizuj macierz pomyłek
# println("Wizualizacja macierzy pomyłek...")
# y_pred = argmax(Models.forward(model, X_test), dims=1)
# Utils.plot_confusion_matrix(y_test, y_pred, 10)
# savefig("confusion_matrix.png")
# 
# # Zapisz model
# println("Zapisywanie modelu...")
# Utils.save_model(model, "best_model.json")
# 
# # Wizualizuj przykładowe predykcje
# println("Wizualizacja przykładowych predykcji...")
# Utils.visualize_predictions(model, X_test, y_test, n_samples=5)
# 
# println("Zakończono!") 