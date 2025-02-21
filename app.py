import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import plotly.express as px
import os

# Helper function to initialize a Neural Network
def init_nn(activation='tanh', alpha=0.001, l1_size=8, l2_size=4, l3_size=0, l4_size=0, lr=0.001):
    h_layer_sizes = (l1_size,)
    
    if l2_size != 0 and l3_size == 0 and l4_size == 0:
        h_layer_sizes = (l1_size, l2_size)
    elif l2_size != 0 and l3_size != 0 and l4_size == 0:
        h_layer_sizes = (l1_size, l2_size, l3_size)
    elif l2_size != 0 and l3_size != 0 and l4_size != 0:
        h_layer_sizes = (l1_size, l2_size, l3_size, l4_size)
            
    clf = MLPClassifier(activation=activation, solver='adam', alpha=alpha, hidden_layer_sizes=h_layer_sizes,
                    learning_rate="constant", learning_rate_init=lr,
                    random_state=42, max_iter=1, warm_start=True)
    return clf

# Helper function for gradient descent (used in Linear Regression)
def gradient_descent(X, y, lr=0.001, epoch=300, lmd=0.01, reg='L2'):
    b1, b0 = 0.0, 0.0  # parameters
    b1_list, b0_list = [], []
    log, mse = [], []  # lists to store learning process
    
    for i in range(epoch):
        sumyhat = 0
        sumxyhat = 0

        reg_grad = 0

        if reg == 'L1':
            reg_grad = lmd
        elif reg == 'L2':
            reg_grad = 2 * b1 * lmd

        for j in range(len(X)):
            sumyhat += b0 + b1 * X[j] + reg_grad - y[j]
            sumxyhat += (b0 + b1 * X[j] + reg_grad - y[j]) * (X[j])
            
        # CALCULATE AND UPDATE b1 AND b0
        b1 -= lr * (1 / len(X)) * sumxyhat
        b0 -= lr * (1 / len(X)) * sumyhat
        b1_list.append(b1)
        b0_list.append(b0)

        log.append((b1, b0))
        reg_term = 0

        if reg == 'L1':
            reg_term = lmd * np.sum(b1)
        elif reg == 'L2':
            reg_term = lmd * np.sum(b1 * b1)

        mse.append(mean_squared_error(y, (b1 * X + b0)) + reg_term)        

    return b1, b0, log, mse, b1_list, b0_list

# Helper function to initialize Logistic Regression
def init_lr(alpha=0.001, lr=0.001, reg_type="L2"):
    clf = SGDClassifier(loss="log", penalty=reg_type.lower(), alpha=alpha, learning_rate="constant",
                    eta0=lr, max_iter=1, warm_start=True)
    return clf

# Helper function to initialize Linear SVM
def init_lsvm(alpha=0.001, lr=0.001, reg_type="L2"):
    clf = SGDClassifier(loss="hinge", penalty=reg_type.lower(), alpha=alpha, learning_rate="constant",
                    eta0=lr, max_iter=1, warm_start=True)
    return clf

# Helper function to perform KMeans clustering
def kmeans_cluster_centers(X, xx, yy, n_clusters=2, iterations=100):
    n_centroids = []
    labels = []
    Z_l = []
    centroids = None
    for i in range(iterations):
        clus = KMeans(
            max_iter=1,
            n_init=1,
            init=(centroids if centroids is not None else 'k-means++'),
            n_clusters=n_clusters,
            random_state=1)
        clus.fit(X)
        centroids = clus.cluster_centers_
        n_centroids.append(centroids)
        labels.append(clus.fit_predict(X))
        Z_l.append(clus.predict(np.c_[xx.ravel(), yy.ravel()]))
    return n_centroids, labels, Z_l

# Helper function to initialize Gaussian Naive Bayes
def init_gnb():
    clf = GaussianNB()
    return clf

# Helper function to initialize Decision Tree
def init_dtree(criterion='gini'):
    clf = DecisionTreeClassifier(criterion=criterion)
    return clf

# Helper function to initialize PCA
def init_pca():
    pca = PCA()
    return pca

# Helper function to initialize SVM
def init_svm(C=1, kernel='rbf', degree=3):
    clf = SVC(C=C, kernel=kernel, degree=degree)
    return clf

# Helper function to load Logistic Regression datasets
def init_lr_dataset(dataset="Uniform"):
    if dataset == "Uniform":
        df = pd.read_csv(os.path.join("datasets", "classification_linear_uniform.csv"))
        df.loc[df["y"] == -1, "y"] = 0  
    elif dataset == "XOR":
        df = pd.read_csv(os.path.join("datasets", "classification_nonlinear_xor.csv"))
        df.loc[df["y"] == -1, "y"] = 0
    
    feature_cols = ['x1', 'x2']
    X = df[feature_cols].to_numpy()
    y = df['y'].to_numpy()
    
    return X, y, df

# Helper function to load Linear Regression datasets
def init_linreg_dataset(dataset="Linear"):
    if dataset == "Linear":
        df = pd.read_csv(os.path.join("datasets", "regression_linear_line.csv"))
    elif dataset == "Square Root":
        df = pd.read_csv(os.path.join("datasets", "regression_linear_square_root.csv"))    
    
    X = np.array(df['x'])
    y = np.array(df['y'])
    
    return X, y, df

# Helper function to load KMeans datasets
def init_kmeans_dataset(dataset='4 Cluster'):
    if dataset == '4 Cluster':
        df = pd.read_csv(os.path.join("datasets", "clustering_4clusters.csv"))
    elif dataset == 'Uniform':
        df = pd.read_csv(os.path.join("datasets", "classification_linear_uniform.csv"))
    elif dataset == 'XOR':
        df = pd.read_csv(os.path.join("datasets", "classification_nonlinear_xor.csv"))
    elif dataset == 'With Outliers':
        df = pd.read_csv(os.path.join("datasets", "classification_clustering_outliers.csv"))
        df["x1"] = df["x"]
        df["x2"] = df["y"]
    
    feature_cols = ['x1', 'x2']
    X = df[feature_cols].to_numpy()
    
    return X, df

# Helper function to load Gaussian Naive Bayes datasets
def init_gnb_dataset(dataset="Independent"):
    if dataset == "Independent":
        df = pd.read_csv(os.path.join("datasets", "classification_naive_bayes_bernoulli_independent.csv"))
        df.loc[df["y"] == -1, "y"] = 0 
    elif dataset == "Dependent":
        df = pd.read_csv(os.path.join("datasets", "classification_dependent_feature.csv"))
        df.loc[df["y"] == -1, "y"] = 0 
    
    feature_cols = ['x1', 'x2']
    X = df[feature_cols].to_numpy()
    y = df['y'].to_numpy()
    
    return X, y, df

# Helper function to load PCA datasets
def init_pca_dataset(dataset="Uniform"):
    if dataset == "Uniform":
        df = pd.read_csv(os.path.join("datasets", "classification_linear_uniform.csv"))
        df.loc[df["y"] == -1, "y"] = 0  
    elif dataset == "XOR":
        df = pd.read_csv(os.path.join("datasets", "classification_nonlinear_xor.csv"))
        df.loc[df["y"] == -1, "y"] = 0
    elif dataset == "Circular":
        df = pd.read_csv(os.path.join("datasets", "classification_circle.csv"))
        df.loc[df["y"] == -1, "y"] = 0
    
    feature_cols = ['x1', 'x2']
    X = df[feature_cols].to_numpy()
    y = df['y'].to_numpy()
    
    return X, y, df

# Helper function to plot Gaussian Naive Bayes results
def plot_gnb(dataset='Independent'):
    plt.clf()
    X, y, df = init_gnb_dataset(dataset=dataset)
    xmin, xmax = df["x1"].min(), df["x1"].max() 
    ymin, ymax = df["x2"].min(), df["x2"].max()
    
    clf = init_gnb()
    clf.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)

    plt.imshow(
         Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        origin="lower",
        cmap=plt.cm.PuOr_r,
        alpha=0.7
    )

    contours = plt.contour(xx, yy, Z, linewidths=1, colors="white", alpha=0.1)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    sns.scatterplot(x="x1", y="x2", hue="y", data=df)
    plt.legend(loc='lower right')
    st.pyplot(plt)

# Helper function to plot PCA results
def plot_pca(dataset='Uniform'):
    plt.clf()
    X, y, df = init_pca_dataset(dataset=dataset)
    
    pca = init_pca()
    components = pca.fit_transform(X)
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(2),
        color=df["y"]
    )
    fig.update_traces(diagonal_visible=False)
    st.plotly_chart(fig)

# Main Streamlit App Code
def main():
    st.title("Machine Learning Visualization App")

    # Sidebar for algorithm selection and parameters
    with st.sidebar:
        st.header("Algorithm Selection")
        algorithms = ['Linear Regression', 'Logistic Regression', 'Neural Network', 'KMeans', 'PCA', 'None']
        algo = st.selectbox("Select Algorithm", algorithms)

        if algo == "Linear Regression":
            st.header("Linear Regression Parameters")
            lr = st.slider("Learning Rate", 0.0001, 1.0, 0.001, 0.0001)
            reg_strength = st.slider("Regularization Strength", 0.0, 1.0, 0.001, 0.0001)
            reg_type = st.selectbox("Regularization Type", ['L1', 'L2', 'None'])
            dataset_linreg = st.selectbox("Dataset", ['Linear', 'Square Root'])
            epochs = st.slider("Epochs", 10, 1000, 300, 10)

        elif algo == "Logistic Regression":
            st.header("Logistic Regression Parameters")
            lr = st.slider("Learning Rate", 0.0001, 1.0, 0.001, 0.0001)
            reg_strength = st.slider("Regularization Strength", 0.0, 1.0, 0.001, 0.0001)
            reg_type = st.selectbox("Regularization Type", ['L1', 'L2', 'None'])
            dataset_lr = st.selectbox("Dataset", ['Uniform', 'XOR'])
            epochs = st.slider("Epochs", 10, 1000, 300, 10)

        elif algo == "Neural Network":
            st.header("Neural Network Parameters")
            activation = st.selectbox("Activation Function", ['logistic', 'tanh', 'relu'])
            lr = st.slider("Learning Rate", 0.0001, 1.0, 0.001, 0.0001)
            reg_strength = st.slider("Regularization Strength", 0.0, 1.0, 0.001, 0.0001)
            l1_size = st.selectbox("L1 Size", [1, 2, 4, 8])
            l2_size = st.selectbox("L2 Size", [0, 1, 2, 4, 8])
            l3_size = st.selectbox("L3 Size", [0, 1, 2, 4, 8])
            l4_size = st.selectbox("L4 Size", [0, 1, 2, 4, 8])
            dataset_nn = st.selectbox("Dataset", ['Circular', 'XOR'])
            epochs = st.slider("Epochs", 10, 1000, 300, 10)

        elif algo == "KMeans":
            st.header("KMeans Parameters")
            n_clusters = st.slider("Number of Clusters", 1, 7, 3, 1)
            dataset_kmeans = st.selectbox("Dataset", ['Uniform', 'XOR', 'With Outliers', '4 Cluster'])
            epochs = st.slider("Epochs", 10, 1000, 300, 10)

        elif algo == "PCA":
            st.header("PCA Parameters")
            dataset_pca = st.selectbox("Dataset", ['Uniform', 'XOR', 'Circular'])

    # Main content area
    if algo == "Linear Regression":
        if st.button("Run Linear Regression"):
            X, y, df = init_linreg_dataset(dataset=dataset_linreg)
            if X is not None and y is not None:
                w, b, _, _, w_list, b_list = gradient_descent(X=X, y=y, lr=lr, epoch=epochs, lmd=reg_strength, reg=reg_type)
                
                plt.figure()
                sns.scatterplot(x=X.flatten(), y=y)  # Ensure X is 1D for scatterplot
                y_graph = w_list[-1] * X + b_list[-1]
                plt.plot(X, y_graph, 'r', linewidth=2.5)
                plt.title("Final Linear Regression Fit")
                st.pyplot(plt)

    elif algo == "Logistic Regression":
        if st.button("Run Logistic Regression"):
            X, y, df = init_lr_dataset(dataset=dataset_lr)
            if X is not None and y is not None:
                clf = init_lr(alpha=reg_strength, lr=lr, reg_type=reg_type)
                clf.fit(X, y)
                
                xmin, xmax = df["x1"].min(), df["x1"].max() 
                ymin, ymax = df["x2"].min(), df["x2"].max()
                xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                plt.figure()
                plt.imshow(
                    Z,
                    interpolation="nearest",
                    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                    aspect="auto",
                    origin="lower",
                    alpha=0.7,
                    cmap=plt.cm.Paired
                )
                contours = plt.contour(xx, yy, Z, linewidths=2, colors="white")
                sns.scatterplot(x="x1", y="x2", hue="y", data=df)
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.title("Final Logistic Regression Decision Boundary")
                st.pyplot(plt)

    if algo == "Neural Network":
        if st.button("Run Neural Network"):
            X, y, df = init_lr_dataset(dataset=dataset_nn)
            if X is not None and y is not None and df is not None:
                clf = init_nn(activation=activation, alpha=reg_strength, l1_size=l1_size, l2_size=l2_size,
                            l3_size=l3_size, l4_size=l4_size, lr=lr)
                clf.fit(X, y)
                
                xmin, xmax = df["x1"].min(), df["x1"].max() 
                ymin, ymax = df["x2"].min(), df["x2"].max()
                xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                plt.figure()
                plt.imshow(
                    Z,
                    interpolation="nearest",
                    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                    aspect="auto",
                    origin="lower",
                    alpha=0.7,
                    cmap=plt.cm.Paired
                )
                contours = plt.contour(xx, yy, Z, linewidths=2, colors="white")
                sns.scatterplot(x="x1", y="x2", hue="y", data=df)
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
                plt.title("Final Neural Network Decision Boundary")
                st.pyplot(plt)

    elif algo == "KMeans":
        if st.button("Run KMeans"):
            X, df = init_kmeans_dataset(dataset=dataset_kmeans)
            if X is not None:
                xmin, xmax = df["x1"].min(), df["x1"].max() 
                ymin, ymax = df["x2"].min(), df["x2"].max()
                xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))
                
                n_centroids, labels, Z_l = kmeans_cluster_centers(X, xx, yy, n_clusters=n_clusters, iterations=epochs)
                
                plt.figure()
                Z = Z_l[-1].reshape(xx.shape)
                plt.imshow(
                    Z,
                    interpolation="none",
                    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                    cmap=plt.cm.Paired,
                    alpha=0.5,
                    aspect="auto",
                    origin="lower",
                )
                sns.scatterplot(x="x1", y="x2", hue=labels[-1], palette="deep", data=df)
                centroids = n_centroids[-1]
                plt.scatter(
                    centroids[:, 0], 
                    centroids[:, 1],
                    marker="x",
                    s=80,
                    linewidths=3,
                    color="black",
                    zorder=10,
                )
                plt.title("Final KMeans Clustering")
                st.pyplot(plt)

    elif algo == "PCA":
        if st.button("Run PCA"):
            plot_pca(dataset=dataset_pca)

if __name__ == "__main__":
    main()

