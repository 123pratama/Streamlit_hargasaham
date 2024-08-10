import pandas as pd
import numpy as np 
import streamlit as st
import pickle 
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Memuat Fungsi data dari file csv
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Fungsi untuk Clustering dan visualisasi dendrogram
def perform_clustering(x_train, n_clusters, linkage_method):
    agglomerative_cluster = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage=linkage_method)
    cluster_labels_train = agglomerative_cluster.fit_predict(x_train)

    silhouette_avg_train = silhouette_score(x_train, cluster_labels_train)

    fig, ax = plt.subplots(figsize=(10,7))
    ax.set_title("Dendrogram Harga Saham Bank BRI")
    dend = dendrogram(linkage(x_train, method=linkage_method), ax=ax)
    st.pyplot(fig)

    # Fungsi untuk menyimpan model_clustering
    with open('simpan_model.pkl', 'wb') as f:
        pickle.dump(agglomerative_cluster, f)

    return silhouette_avg_train, agglomerative_cluster, cluster_labels_train

# Fungsi untuk menentukan jumlah cluster terbaik menggunakan metode Silhouette
def find_best_cluster(X, linkage_method):
    silhouette_scores = []
    for n_clusters in range(2, 11):
        agglomerative_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=linkage_method)
        cluster_labels = agglomerative_cluster.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    jumlah_clusters_terbaik = np.argmax(silhouette_scores) + 2  # Karena range dimulai dari 2
    return jumlah_clusters_terbaik, silhouette_scores

# Fungsi untuk memuat simpan model dari file simpan_model.pkl
def load_simpan_model(file_name='simpan_model.pkl'):
    with open(file_name, 'rb') as f:
        simpan_model = pickle.load(f)
    return simpan_model 

# Aplikasi pengelompokan saham
def main():
    st.title("Pengelompokan Harga Saham Bank BRI Menggunakan Agglomerative Hierarchical Clustering")

    # Unggah file csv
    upload_file = st.file_uploader("Unggah file CSV", type=["csv"])
    if upload_file is not None:
        df = load_data(upload_file)

        st.write("Data Harga Saham:")
        st.write(df)

        # Pilih fitur
        fitur = st.multiselect("Pilih Fitur", df.columns.tolist())

        if fitur:
            x = df[fitur].values

            # Checkbox untuk memilih normalisasi dan skala
            use_normalizer = st.checkbox("Gunakan Normalizer")
            use_scaler = st.checkbox("Gunakan Standard Scaler")

            if use_normalizer:
                normalizer = Normalizer()
                x = normalizer.fit_transform(x)
            
            if use_scaler:
                scaler = StandardScaler()
                x = scaler.fit_transform(x)
            
            # Pembagian data menjadi data pelatihan dan pengujian set
            x_train, x_test = train_test_split(x, test_size=0.2, random_state=20)

            # Pilih parameter clustering
            linkage_method = st.selectbox("Pilih metode linkage", ["complete", "average", "single"])
            n_clusters = st.slider("Pilih Jumlah Cluster", 2, 10, 4)
            jumlah_clusters_terbaik, silhouette_scores = find_best_cluster(x_train, linkage_method)
            st.write(f"Jumlah Cluster Terbaik: {jumlah_clusters_terbaik}")
                
            # Plot silhouette scores
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.plot(range(2, 11), silhouette_scores, marker='o')
            ax.set_title('Silhouette Scores untuk Berbagai Jumlah Cluster')
            ax.set_xlabel('Jumlah Cluster')
            ax.set_ylabel('Silhouette Score')
            st.pyplot(fig)


            if st.button("Lakukan Clustering"):
                silhouette_avg_train, agglomerative_cluster, cluster_labels_train = perform_clustering(x_train, n_clusters, linkage_method)

                st.write("Silhouette Score (Pelatihan):", silhouette_avg_train)

                train_clusters, train_counts = np.unique(cluster_labels_train, return_counts=True)
                st.write("Jumlah Anggota dalam Setiap Cluster Pada Set Pelatihan:")
                train_clusters_summary = pd.DataFrame({'Cluster': train_clusters, 'Jumlah Anggota': train_counts})
                st.write(train_clusters_summary)

                fig, ax = plt.subplots(figsize=(10, 7))
                scatter = ax.scatter(x_train[:, 0], x_train[:, 1], c=cluster_labels_train, cmap='rainbow')
                ax.set_title('Hasil Clustering pada Set Pelatihan')
                ax.set_xlabel(fitur[0])
                ax.set_ylabel(fitur[1])
                st.pyplot(fig)

                # Tambahkan anotasi untuk setiap titik
                for i in range(len(x_train)):
                    ax.annotate(cluster_labels_train[i], (x_train[i, 0], x_train[i, 1]))

                st.pyplot(fig)

                X_train_df = pd.DataFrame(x_train, columns=fitur)
                X_train_df['Cluster'] = cluster_labels_train
                st.write("Hasil Clustering pada Set Pelatihan:")
                st.write(X_train_df)

                # Muat model yang disimpan
                cluster_model = load_simpan_model()

                cluster_labels_test = cluster_model.fit_predict(x_test)

                fig, ax = plt.subplots(figsize=(10, 7))
                scatter = ax.scatter(x_test[:, 0], x_test[:, 1], c=cluster_labels_test, cmap='rainbow')
                ax.set_title('Hasil Clustering pada Set Pengujian')
                ax.set_xlabel(fitur [0])
                ax.set_ylabel(fitur [1])
                st.pyplot(fig)
                
                # Tambahkan anotasi untuk setiap titik
                for i in range(len(x_test)):
                    ax.annotate(cluster_labels_test[i], (x_test[i, 0], x_test[i, 1]))

                st.pyplot(fig)

                silhouette_avg_test = silhouette_score(x_test, cluster_labels_test)
                st.write("Silhouette Score (Test):", silhouette_avg_test)

                test_clusters, test_counts = np.unique(cluster_labels_test, return_counts=True)
                st.write("Jumlah Anggota dalam Setiap Cluster pada Set Pengujian:")
                test_cluster_summary = pd.DataFrame({'Cluster': test_clusters, 'Jumlah Anggota': test_counts})
                st.write(test_cluster_summary)


                X_test_df = pd.DataFrame(x_test, columns=fitur)
                X_test_df['Cluster'] = cluster_labels_test
                st.write("Hasil Clustering pada Set Pengujian:")
                st.write(X_test_df)

                kombinasi_df = pd.concat([X_train_df, X_test_df], ignore_index=True)
                st.write("Hasil Clustering Gabungan:")
                st.write(kombinasi_df)


if __name__ == "__main__":
    main()
