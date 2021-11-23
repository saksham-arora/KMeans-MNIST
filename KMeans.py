import numpy as np
import copy
import matplotlib.pyplot as plt
import numpy as np
import random

training_data = [line.split() for line in open('mnist_raw_data.txt').readlines()]
training_labels=[int(line.strip('\n')) for line in open('mnist_labels.txt','r').readlines()]
train_data= np.array(training_data, dtype=np.uint8)

class KMeans:

    def __init__(self, n_clusters=10, max_iter=500):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.loss_per_iteration = []

    def initialize_centroids(self):
#         np.random.seed(np.random.randint(0, 100000))
        
        random.choice(training_labels)
        self.centroids = []
        for i in range(self.n_clusters):
            rand_index = np.random.choice(range(len(self.fit_data)))
            self.centroids.append(self.fit_data[rand_index])

    def initialize_clusters(self):
        self.clusters = {'data': {i: [] for i in range(self.n_clusters)}}
        self.clusters['labels'] = {i: [] for i in range(self.n_clusters)}

    def form_cluster(self, fit_data, fit_labels):
        self.fit_data = fit_data
        self.fit_labels = fit_labels
        self.predicted_labels = [None for _ in range(self.fit_data.shape[0])]
        self.initialize_centroids()
        self.iterations = 0
        old_centroids = [np.zeros(shape=(fit_data.shape[1],))
                         for _ in range(self.n_clusters)]
        while self.convergence_check(self.iterations, old_centroids, self.centroids):
            old_centroids = copy.deepcopy(self.centroids)
            self.initialize_clusters()
            for j, sample in enumerate(self.fit_data):
                min_dist = float('inf')
                for i, centroid in enumerate(self.centroids):
                    dist = np.linalg.norm(sample-centroid)
                    if dist < min_dist:
                        min_dist = dist
                        self.predicted_labels[j] = i
                if self.predicted_labels[j] is not None:
                    self.clusters['data'][self.predicted_labels[j]].append(
                        sample)
                    self.clusters['labels'][self.predicted_labels[j]].append(
                        self.fit_labels[j])
            self.reshape_cluster()
            self.update_centroids()
            print("\nEpoch: ", self.iterations, '==> Difference between centroids: ', self.centroids_dist)
            self.iterations += 1
        incorrect_instances=self.calculate_accuracy()
        return self.iterations,incorrect_instances

    def update_centroids(self):
        for i in range(self.n_clusters):
            cluster = self.clusters['data'][i]
            if cluster == []:
                self.centroids[i] = self.fit_data[np.random.choice(
                    range(len(self.fit_data)))]
            else:
                self.centroids[i] = np.mean(
                    np.vstack((self.centroids[i], cluster)), axis=0)

    def reshape_cluster(self):
        for id, mat in list(self.clusters['data'].items()):
            self.clusters['data'][id] = np.array(mat)

    def convergence_check(self, iterations, centroids, updated_centroids):
        if iterations > self.max_iter:
            return False
        self.centroids_dist = np.linalg.norm(
            np.array(updated_centroids)-np.array(centroids))
        if self.centroids_dist <= 1e-10:
            print("Converged! ==> ", self.centroids_dist)
            return False
        return True

    def calculate_accuracy(self):
        self.clusters_labels = []
        self.clusters_info = []
        self.clusters_accuracy = []
        for clust, labels in list(self.clusters['labels'].items()):
            if isinstance(labels[0], (np.ndarray)):
                labels = [l[0] for l in labels]
            occur = 0
            max_label = max(set(labels), key=labels.count)
            self.clusters_labels.append(max_label)
            for label in labels:
                if label == max_label:
                    occur += 1
            acc = occur/len(list(labels))
            self.clusters_info.append(
                [max_label, occur, len(list(labels)), acc])
            self.clusters_accuracy.append(acc)
            self.accuracy = sum(self.clusters_accuracy)/self.n_clusters
        self.labels_ = []
        for i in range(len(self.predicted_labels)):
            self.labels_.append(self.clusters_labels[self.predicted_labels[i]])
        print('Accuracy:', self.accuracy)
        return self.clusters_info[0][2] - self.clusters_info[0][1]
    def reshape_to_plot(self, data):
        return data.reshape(data.shape[0], 28, 28).astype("uint8")
    
    def plot_imgs(self, in_data, n, random=False):
        data = np.array([d for d in in_data])
        data = self.reshape_to_plot(data)
        x1 = min(n//2, 5)
        if x1 == 0:
            x1 = 1
        y1 = (n//x1)
        x = min(x1, y1)
        y = max(x1, y1)
        fig, ax = plt.subplots(x, y, figsize=(5, 5))
        i = 0
        for j in range(x):
            for k in range(y):
                if random:
                    i = np.random.choice(range(len(data)))
                ax[j][k].set_axis_off()
                ax[j][k].imshow(data[i:i+1][0])
                i += 1
        plt.show()

    def plot_img(self, data):
        assert data.shape == (28*28,)
        data = data.reshape(1, 28, 28).astype('uint8')
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data[0])
        plt.show()
