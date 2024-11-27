import numpy as np
import scipy.spatial.distance as dist
from matplotlib import pyplot as plt

def kmeans(X, max_iter=1000):
    k = 10
    n, d = X.shape
    assert max_iter > 0

    indices= np.random.choice(n, k, replace=False)  
    U = X[indices]
    for i in range(max_iter):
        D = dist.cdist(X, U)
        Y = cluster_assignments(D)
        old_U = U
        U = np.linalg.inv((Y.T @ Y + 1e-8 * np.eye(k)))@ Y.T @ X
        if np.allclose(old_U, U):
            break
    W = X - Y@U
    obj_val = 1/(2*n) * np.linalg.norm(W, 'fro') ** 2 
    return Y, U, obj_val

def cluster_assignments(D):
    max_indices = np.argmin(D, axis=1)

    Y = np.zeros_like(D)
    for index, max_index in enumerate(max_indices):
        Y[index][max_index] = 1
    return Y

def repeatKmeans(X, n_runs=100):
    best_obj_val = float('inf')
    best_y = None
    best_u = None
    for r in range(n_runs):
        print(r)
        Y, U, obj_val = kmeans(X)
        # TODO: Compare obj_val with best_obj_val. If it is lower,
        # then record the current Y, U and update best_obj_val
        if obj_val < best_obj_val:
            best_obj_val = obj_val
            best_y = Y
            best_u = U

    return best_y, best_u

def writeData(data, file_path="output.txt"): 
    # Writing the 2D list to a text file
    with open(file_path, "w") as file:
        for row in data:
            # Convert each row to a string with elements separated by spaces
            file.write(" ".join(map(str, row)) + "\n")   

def readfiles():
    # Reading the file back into a 2D list
    with open("best_y.txt", "r") as file:
        best_y = [list(map(float, line.split())) for line in file]
    
    with open("best_u.txt", "r") as file:
        best_u = [list(map(float, line.split())) for line in file]

    return np.array(best_y), np.array(best_u)

def prepareData(file_name):
    train_data = np.loadtxt(file_name, delimiter=',')
    y = train_data[:, 0]
    X = train_data[:, 1:] / 255.
    X1 = X[:, :784] #first image
    X2 = X[:, 784:1568] #second image
    X3 = X[:, 1568:] #third image

    return y, X, X1, X2, X3

def plotImg(x):
    img = x.reshape((84, 28))
    plt.imshow(img, cmap='gray')
    plt.show()
    return

def cluster_assignments(D):
    max_indices = np.argmin(D, axis=1)

    Y = np.zeros_like(D)
    for index, max_index in enumerate(max_indices):
        Y[index][max_index] = 1
    return Y

def validate(cluster_predictions, U, X1, X2, X3, Y):
    first_image_assignment = dist.cdist(X1, U)
    image_1_assignment = cluster_assignments(first_image_assignment)

    second_image_assignment = dist.cdist(X2, U)
    image_2_assignment = cluster_assignments(second_image_assignment)

    third_image_assignment = dist.cdist(X3, U)
    image_3_assignment = cluster_assignments(third_image_assignment)

    total = Y.shape[0]
    correct = 0

    for index, value in enumerate(image_1_assignment):
        prediction = value.argmin()
        if cluster_predictions[prediction] % 2 == 0:
            img_prediction = image_3_assignment[index].argmin()
            actual_prediction = cluster_predictions[img_prediction]
        else:
            img_prediction = image_2_assignment[index].argmin()
            actual_prediction = cluster_predictions[img_prediction]

        if actual_prediction == Y[index]:
            correct += 1

    print("accuracy: ", correct/total)

if __name__ == "__main__":
    y, X, X1, X2, X3 = prepareData('A4data/A4train.csv')

    val_Y, val_X, val_X1, val_X2, val_X3 = prepareData('A4data/A4val.csv')

    training_data = np.concatenate([X2, X3])

    best_y, best_u = repeatKmeans(training_data)

    writeData(best_y, 'best_y.txt')
    writeData(best_u, 'best_u.txt')
    
    # best_y, best_u = readfiles()

    cluster_maps = np.zeros((10, 10))

    for index, y_val in enumerate(best_y):
        index = index % 748
        label = y[index]
        cluster_maps[y_val.argmax()][int(label)] += 1

    cluster_predictions = []

    for map in cluster_maps:
        predicted_label = map.argmax()
        cluster_predictions.append(predicted_label)

    print(cluster_predictions)

    validate(cluster_predictions, best_u, val_X1, val_X2, val_X3, val_Y)