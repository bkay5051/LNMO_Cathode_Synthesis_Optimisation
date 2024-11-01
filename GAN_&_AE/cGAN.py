def data_prep1():

    # Define columns to be normalized and one-hot encoded
    numerical_cols = ['Li', 'Ni', 'Mn', 'O', 'Sintering Temperature (°C) 1st step', 'Sintering Temperature (°C) 2nd step',
                  'Sintering Temperature (°C) 3rd step', 'Sintering Time (h) 1st step', 'Sintering Time (h) 2nd step',
                  'Sintering Time (h) 3rd step', 'Lattice Constant', 'Current Density/C rate',
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
    y = df['Discharge Capacity [mAh/g]']

    # Separate input features
    X =  df.drop(columns=['Discharge Capacity [mAh/g]', 'Coulombic efficiency%', 'Capacity Retention (%)', 'Rate Capability (mAh/g)', 'Rate Capability (C rate)', 'Capacity Retention (Cycles)'])

    # Store original column names
    original_feature_names = X.columns.tolist()

    # Fit the preprocessor
    X_preprocessed = preprocessor.fit_transform(X)

    # Get feature names after preprocessing
    numeric_feature_names = numerical_cols
    categorical_feature_names =  preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    preprocessed_feature_names = np.concatenate([numeric_feature_names, categorical_feature_names])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.15, random_state=42, shuffle=True)
    
    # Scale the features
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale the target
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    return  X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y, df, X_preprocessed, y, X_train, X_test, y_train, y_test, preprocessed_feature_names, categorical_feature_names, categorical_cols

# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, condition_dim, output_dim, hidden_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()
        )

    def forward(self, z, condition):
        x = torch.cat([z, condition], 1)
        return self.model(x)

# Define the Discriminator 
class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, condition):
        x = torch.cat([x, condition], 1)
        return self.model(x)

#Train the cGAN
def train_cgan(X_train, y_train, params):
    latent_dim = params['latent_dim']
    condition_dim = 1
    input_dim = X_train.shape[1]

    generator = Generator(latent_dim, condition_dim, input_dim, params['hidden_dim']).to(params['device'])
    discriminator = Discriminator(input_dim, condition_dim, params['hidden_dim']).to(params['device'])

    g_optimizer = optim.Adam(generator.parameters(), lr=params['g_lr'], betas=(params['beta1'], params['beta2']))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=params['d_lr'], betas=(params['beta1'], params['beta2']))

    g_losses = []
    d_losses = []

    patience = params['patience']
    best_loss = float('inf')
    counter = 0
    

    for epoch in range(params['epochs']):
        epoch_g_losses = []
        epoch_d_losses = []

        for i in range(0, X_train.shape[0], params['batch_size']):
            batch_X = torch.FloatTensor(X_train[i:i+params['batch_size']]).to(params['device'])
            batch_y = torch.FloatTensor(y_train[i:i+params['batch_size']]).unsqueeze(1).to(params['device'])
            batch_size = batch_X.size(0)

            # Train Discriminator
            d_optimizer.zero_grad()
            
            z = torch.randn(batch_size, latent_dim).to(params['device'])
            fake_X = generator(z, batch_y)
            
            real_loss = torch.mean(discriminator(batch_X, batch_y))
            fake_loss = torch.mean(discriminator(fake_X.detach(), batch_y))
            d_loss = fake_loss - real_loss

            d_loss.backward()
            d_optimizer.step()

            epoch_d_losses.append(d_loss.item())

            for p in discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)

            # Train Generator
            if i % params['n_critic'] == 0:
                g_optimizer.zero_grad()
                
                z = torch.randn(batch_size, latent_dim).to(params['device'])
                fake_X = generator(z, batch_y)
                g_loss = -torch.mean(discriminator(fake_X, batch_y))

                g_loss.backward()
                g_optimizer.step()

                epoch_g_losses.append(g_loss.item())

        avg_g_loss = np.mean(epoch_g_losses)
        avg_d_loss = np.mean(epoch_d_losses)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        current_loss = avg_g_loss + avg_d_loss
        if current_loss < best_loss:
            best_loss = current_loss
            counter = 0
            best_generator = generator.state_dict()
            best_discriminator = discriminator.state_dict()
        else:
            counter += 1
            if counter >= patience:
                generator.load_state_dict(best_generator)
                discriminator.load_state_dict(best_discriminator)
                break

    return generator, g_losses, d_losses

# Define generating synthetic data
def generate_synthetic_data(generator, num_samples, latent_dim, y_train, device):
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        y = torch.FloatTensor(y_train[:num_samples]).unsqueeze(1).to(device)
        synthetic_X = generator(z, y).cpu().numpy()
        synthetic_y = y.cpu().numpy().flatten()
    return synthetic_X, synthetic_y

# Define objective for cGAN
def objective(trial):
    params = {
        'latent_dim': trial.suggest_int('latent_dim', 3, 200),
        'hidden_dim': trial.suggest_int('hidden_dim', 4, 256),
        'g_lr': trial.suggest_float('g_lr', 1e-5, 1e-1, log=True),
        'd_lr': trial.suggest_float('d_lr', 1e-5, 1e-1, log=True),
        'beta1': trial.suggest_float('beta1', 0.001, 0.9),
        'beta2': trial.suggest_float('beta2', 0.001, 0.999),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128]),
        'epochs': 10000,
        'patience': trial.suggest_int('patience', 50, 100),
        'n_critic': trial.suggest_int('n_critic', 1, 10),
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    generator, _, _ = train_cgan(X_train_scaled, y_train_scaled, params)
    
    X_synthetic, y_synthetic = generate_synthetic_data(generator, X_train_scaled.shape[0], params['latent_dim'], y_train_scaled, params['device'])
    
    X_train_combined = np.vstack((X_train_scaled, X_synthetic))
    y_train_combined = np.concatenate((y_train_scaled, y_synthetic))
    
    model = RandomForestRegressor(random_state=42)
    mse, _ = evaluate_model(model, X_train_combined, y_train_combined, X_test_scaled, y_test_scaled, scaler_y)
    
    return mse

# Call original data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y, df, X_preprocessed, y, X_train, X_test, y_train, y_test, preprocessed_feature_names, categorical_feature_names, categorical_cols = data_prep1()

# cGAN optimisation
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000)

best_params = study.best_params
best_params.update({
    'epochs': 10000,
    'patience': 200,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
})

# Train the cGAN
generator, g_losses, d_losses = train_cgan(X_train_scaled, y_train_scaled, best_params)

# Visualize cGAN losses
plt.figure(figsize=(10, 5))
plt.plot(g_losses, label='Generator Loss')
plt.plot(d_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GAN Training Losses')
plt.legend()
plt.show()

# Generate synthetic data
num_synthetic_samples = X_train_scaled.shape[0]
X_synthetic, y_synthetic = generate_synthetic_data(generator, num_synthetic_samples, best_params['latent_dim'], y_train_scaled, best_params['device'])

# Combine original and synthetic data
X_train_combined = np.vstack((X_train_scaled, X_synthetic))
y_train_combined = np.concatenate((y_train_scaled, y_synthetic))

# Distribution comparison
def plot_distribution_comparison(original_data, synthetic_data, feature_name):
    plt.figure(figsize=(10, 6))
    plt.hist(original_data, bins=50, alpha=0.5, label='Original')
    plt.hist(synthetic_data, bins=50, alpha=0.5, label='Synthetic')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.title(f'Distribution Comparison: {feature_name}')
    plt.legend()
    plt.close()

# Compare distributions for each feature
for i in range(X_train_scaled.shape[1]):
    plot_distribution_comparison(X_train_scaled[:, i], X_synthetic[:, i], f'Feature {i+1}')

# Compare distribution of target variable
plot_distribution_comparison(y_train_scaled, y_synthetic, 'Target Variable')

# Correlation analysis
def plot_correlation_heatmap(data, title):
    corr = np.corrcoef(data.T)
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.close()

# Plot correlation heatmaps
plot_correlation_heatmap(np.column_stack((X_train_scaled, y_train_scaled.reshape(-1, 1))), 'Correlation Heatmap: Original Data')
plot_correlation_heatmap(np.column_stack((X_synthetic, y_synthetic.reshape(-1, 1))), 'Correlation Heatmap: Synthetic Data')

# Print best hyperparameters
print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

# Define function to re-call combined data
def data_prep():
    X = X_preprocessed
    X_train = X_train_combined
    X_test = X_test_scaled
    y_train = scaler_y.inverse_transform(y_train_combined.reshape(-1, 1)).flatten()
    y_test = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    return df, X, y, X_train, X_test, y_train, y_test, preprocessed_feature_names, categorical_feature_names, categorical_cols


df, X, y, X_train, X_test, y_train, y_test, preprocessed_feature_names, categorical_features, categorical_cols = data_prep()
