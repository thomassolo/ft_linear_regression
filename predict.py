import numpy as np


def load_model():
    return np.loadtxt("model.csv", delimiter=",")


def estimatePrice(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def normalize(value, mean, std):
    return (value - mean) / std


def denormalize(value, mean, std):
    return value * std + mean


if __name__ == "__main__":
    theta0, theta1, mileage_mean, mileage_std, price_mean, price_std = load_model()

    try:
        mileage = float(input("Entrez un kilométrage: "))
        mileage_normalized = normalize(mileage, mileage_mean, mileage_std)
        predicted_price_normalized = estimatePrice(mileage_normalized, theta0, theta1)
        predicted_price = denormalize(predicted_price_normalized, price_mean, price_std)
        print(f"Le prix prédit pour un véhicule de {mileage} km est de {predicted_price:.2f} €.")
    except ValueError:
        print("Vous devez entrer un nombre.")