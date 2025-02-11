import numpy as np
import matplotlib.pyplot as plt


def load_data():
    data = np.loadtxt("data.csv", delimiter=',', skiprows=1)
    mileage = data[:, 0]
    price = data[:, 1]
    mileage_mean, mileage_std = np.mean(mileage), np.std(mileage)
    price_mean, price_std = np.mean(price), np.std(price)
    mileage = (mileage - mileage_mean) / mileage_std
    price = (price - price_mean) / price_std
    return mileage, price, mileage_mean, mileage_std, price_mean, price_std


def estimatePrice(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def gradientDescent(mileage, price, theta0, theta1, learningRate, iterations):
    m = len(mileage)
    for _ in range(iterations):
        predictions = estimatePrice(mileage, theta0, theta1)

        d_theta0 = (1/m) * np.sum(predictions - price)
        d_theta1 = (1/m) * np.sum((predictions - price) * mileage)

        theta0 = theta0 - learningRate * d_theta0
        theta1 = theta1 - learningRate * d_theta1

    return theta0, theta1

def save_model(theta0, theta1, mileage_mean, mileage_std, price_mean, price_std):
    np.savetxt("model.csv", [theta0, theta1, mileage_mean, mileage_std, price_mean, price_std], delimiter=",")


def evaluate_model(mileage, price, theta0, theta1, mileage_std, price_mean, price_std):
    predictions = estimatePrice(mileage, theta0, theta1)  # Predict with normalized mileage

    predictions_denorm = predictions * price_std + price_mean
    real_prices = price * price_std + price_mean  # Denormalize actual prices

    mae = np.mean(np.abs(predictions_denorm - real_prices))
    mse = np.mean((predictions_denorm - real_prices) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((predictions_denorm - real_prices) ** 2) / np.sum((real_prices - np.mean(real_prices)) ** 2))

    print("\nModel Evaluation (Denormalized Values):")
    print(f"Mean Absolute Error (MAE): {mae:.2f} €")
    print(f"Mean Squared Error (MSE): {mse:.2f} €²")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} €")
    print(f"R² Score: {r2:.4f}")

    return mae, mse, rmse, r2


if __name__ == "__main__":
    mileage, price, mileage_mean, mileage_std, price_mean, price_std = load_data()
    theta0, theta1 = 0, 0
    learningRate = 0.1
    iterations = 1000
    theta0, theta1 = gradientDescent(mileage, price, theta0, theta1, learningRate, iterations)
    save_model(theta0, theta1, mileage_mean, mileage_std, price_mean, price_std)
    print(f"Modèle sauvegardé: theta0 = {theta0}, theta1 = {theta1}")

    # Denormalize mileage for plotting
    mileage_denorm = mileage * mileage_std + mileage_mean
    price_denorm = price * price_std + price_mean
    predictions_denorm = estimatePrice(mileage, theta0, theta1) * price_std + price_mean

    plt.scatter(mileage_denorm, price_denorm, color='blue', label="Données")
    plt.plot(mileage_denorm, predictions_denorm, color='red', label="Régression linéaire")
    plt.xlabel("Kilométrage")
    plt.ylabel("Prix")
    plt.title("Prix en fonction du kilométrage")
    plt.legend()
    plt.show()

    # Evaluate model on real-world prices
    evaluate_model(mileage, price, theta0, theta1, mileage_std, price_mean, price_std)
