# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the auto-encoder
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate=0.1, activation='relu'):
        super(Autoencoder, self).__init__()

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            raise ValueError("Unsupported activation function")

        # Encoder
        for i in range(len(hidden_dims)):
            if i == 0:
                self.encoder_layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:
                self.encoder_layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dims[i]))

        # Decoder
        for i in range(len(hidden_dims) - 1, -1, -1):
            if i == 0:
                self.decoder_layers.append(nn.Linear(hidden_dims[i], input_dim))
            else:
                self.decoder_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i - 1]))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Encoder
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x = self.bn_layers[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if i != len(self.decoder_layers) - 1:
                x = self.activation(x)
                x = self.dropout(x)

        return x

# Define training the auto-encoder
def train_improved_autoencoder(model, X_train, X_test, num_epochs=1000, batch_size=32, learning_rate=0.001,
                               noise_factor=0.1, l1_lambda=0.0, l2_lambda=0.0, loss_fn='mse'):
    model = model.to(device)

    if loss_fn == 'mse':
        criterion = nn.MSELoss()
    elif loss_fn == 'l1':
        criterion = nn.L1Loss()
    elif loss_fn == 'huber':
        criterion = nn.SmoothL1Loss()
    else:
        raise ValueError("Unsupported loss function")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    train_dataset = TensorDataset(X_train, X_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            noisy_batch = batch_x + noise_factor * torch.randn_like(batch_x)

            outputs = model(noisy_batch)
            loss = criterion(outputs, batch_x)

            if l1_lambda > 0:
                l1_reg = torch.tensor(0., requires_grad=True)
                for param in model.parameters():
                    l1_reg = l1_reg + torch.norm(param, 1)
                loss = loss + l1_lambda * l1_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            noisy_test = X_test + noise_factor * torch.randn_like(X_test)
            test_outputs = model(noisy_test)
            test_loss = criterion(test_outputs, X_test).item()

        scheduler.step(test_loss)

        if test_loss < best_loss:
            best_loss = test_loss
            best_model = model.state_dict()

    model.load_state_dict(best_model)
    return model

# Define objective function for auto-encoder
def objective(trial, X_train, X_test):
    hidden_dims = trial.suggest_categorical('hidden_dims', [
        [256, 128, 64, 32],
        [512, 256, 128, 64, 32],
        [1024, 512, 256, 128, 64, 32]
    ])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    noise_factor = trial.suggest_float('noise_factor', 0.05, 0.2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])
    l1_lambda = trial.suggest_float('l1_lambda', 1e-8, 1e-3, log=True)
    l2_lambda = trial.suggest_float('l2_lambda', 1e-8, 1e-3, log=True)
    loss_fn = trial.suggest_categorical('loss_fn', ['mse', 'l1', 'huber'])

    model = Autoencoder(X_train.shape[1], hidden_dims, dropout_rate=dropout_rate, activation=activation).to(
        device)
    trained_model = train_improved_autoencoder(model, X_train, X_test, num_epochs=100,
                                               learning_rate=learning_rate, noise_factor=noise_factor,
                                               l1_lambda=l1_lambda, l2_lambda=l2_lambda, loss_fn=loss_fn)

    trained_model.eval()
    with torch.no_grad():
        noisy_test = X_test + noise_factor * torch.randn_like(X_test)
        test_outputs = trained_model(noisy_test)
        if loss_fn == 'mse':
            loss = nn.MSELoss()(test_outputs, X_test)
        elif loss_fn == 'l1':
            loss = nn.L1Loss()(test_outputs, X_test)
        elif loss_fn == 'huber':
            loss = nn.SmoothL1Loss()(test_outputs, X_test)

    return loss.item()

# Define Optuna initialisation
def hyperparameter_tuning(X_train, X_test):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, X_test), n_trials=100)  # You can adjust the number of trials

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return trial.params

def data_prep1():

    # Define columns to be normalized and one-hot encoded
    numerical_cols = ['Li', 'Ni', 'Mn', 'O', 'Sintering Temperature (°C) 1st step', 'Sintering Temperature (°C) 2nd step',
                  'Sintering Temperature (°C) 3rd step', 'Sintering Time (h) 1st step', 'Sintering Time (h) 2nd step',
                  'Sintering Time (h) 3rd step', 'Lattice Constant', 'Current Density/C rate', 'Capacity Retention (Cycles)',
                  'Weighted Average Sintering Temperature', 'Total Sintering Temperature', 'Average Sintering Temperature', 'Total Sintering Time', 'Average Sintering Time', 'Lower Voltage Limit', 'Upper Voltage Limit', 'Sintering Temperature Range', 'Time-Temperature Product', 'Voltage Window Range']
    categorical_cols = ['Cathode Material Composition', 'Synthesis Method', 'Li Source', 'Ni Source', 'Mn Source',
                    'Mixing method', 'Furnace', 'Gas atmosphere', 'Heating Rate', 'Cooling Rate',
                    'Space group', 'Electrolyte salt', 'Electrolyte solvent', 'voltage window', 'Surface Morphology']

    # Pipeline for numerical features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
    ])

    # Pipeline for categorical features
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False))
    ])

    # Combine the pipelines into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

    # Create target variable
    y = df['Capacity Retention (mAh/g)']

    # Separate input features
    X =  df.drop(columns=['Discharge Capacity [mAh/g]', 'Coulombic efficiency%', 'Capacity Retention (%)', 'Rate Capability (mAh/g)', 'Rate Capability (C rate)', 'Capacity Retention (mAh/g)'])

    # Store original column names
    original_feature_names = X.columns.tolist()

    # Fit the preprocessor
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Scale the data
    scaler = MinMaxScaler()
    X_preprocessed_scaled = scaler.fit_transform(X_preprocessed)

    # Get feature names after preprocessing
    numeric_feature_names = numerical_cols
    categorical_feature_names =  preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    preprocessed_feature_names = np.concatenate([numeric_feature_names, categorical_feature_names])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_scaled, y.values, test_size=0.15, random_state=42, shuffle=True)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    # Hyperparameter tuning with Optuna
    best_params = hyperparameter_tuning(X_train_tensor, X_test_tensor)

    # Train the autoencoder with best parameters
    autoencoder = Autoencoder(X_train.shape[1], best_params['hidden_dims'],
                                      dropout_rate=best_params['dropout_rate'],
                                      activation=best_params['activation']).to(device)
    autoencoder = train_improved_autoencoder(autoencoder, X_train_tensor, X_test_tensor,
                                             learning_rate=best_params['learning_rate'],
                                             noise_factor=best_params['noise_factor'],
                                             l1_lambda=best_params['l1_lambda'],
                                             l2_lambda=best_params['l2_lambda'],
                                             loss_fn=best_params['loss_fn'])

    # Use the autoencoder to denoise the data
    autoencoder.eval()
    with torch.no_grad():
        X_train_denoised = autoencoder(X_train_tensor).cpu().numpy()
        X_test_denoised = autoencoder(X_test_tensor).cpu().numpy()

    return (df, X_preprocessed_scaled, y, X_train_denoised, X_test_denoised,
            y_train, y_test, preprocessed_feature_names, categorical_feature_names, categorical_cols)

# Call the denoised data
df1, X1, y1, X_train1, X_test1, y_train1, y_test1, preprocessed_feature_names, categorical_feature_names, categorical_cols = data_prep1()

print(f"Training set shape: {X_train1.shape}")
print(f"Test set shape: {X_test1.shape}")

# Define function to re-call denoised data
def data_prep():
    df = df1
    X = X1
    y = y1
    X_train = X_train1
    X_test = X_test1
    y_train = y_train1
    y_test = y_test1
    return df, X, y, X_train, X_test, y_train, y_test, preprocessed_feature_names, categorical_feature_names, categorical_cols


df, X, y, X_train, X_test, y_train, y_test, preprocessed_feature_names, categorical_features, categorical_cols = data_prep()
