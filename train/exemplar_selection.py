import math
from collections import Counter
import sys
import numpy as np
import torch.nn as nn
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.covariance import empirical_covariance
from sklearn.metrics import silhouette_score
from torch import from_numpy, no_grad
from torch.autograd import Variable
import os
from .vae import *
from .vae_train import *
from .vae_generate import *
from .vae_plotting import *
from .vae_evaluate import *
from .early_stopping_vae import *
import models.basic_model as models
from .exemplar_strategies import herding
from sklearn.model_selection import train_test_split
from collections import defaultdict
from utils import data_handler
from sklearn.metrics.pairwise import cosine_similarity
from .diff_model import Diff1D, UNet1D, NoiseScheduler, compare_training_and_generated_data #, train_diffusion_model




def get_embeddings(input_dim, instance, feature_extractor, device, exemplar_method):
    if len(instance.shape) > 1:
        with no_grad():
            x = Variable(from_numpy(instance)).to(device)
            if exemplar_method == 'fetril':
                features = feature_extractor.feature_extraction(x.reshape(1,6,128).float(),fetril=True)[0].data.cpu().numpy()
            elif exemplar_method == 'icarl':
                features = feature_extractor.feature_extraction(x.reshape(1,6,128).float(),fetril=False)[0].data.cpu().numpy()
    elif len(instance.shape) == 1:
        with no_grad():
            x = Variable(from_numpy(instance))
            feature = feature_extractor(x.view(-1, input_dim).float())[0].data.numpy()
            features = feature / np.linalg.norm(feature)

    return features

def create_train_val_loaders(data_dict, val_percentage=None, batchSize=32, normalized=True):
    all_data = []
    all_labels = []

    for class_label, data in data_dict.items():
        # Convert data to tensor
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.full((data.size(0),), class_label, dtype=torch.long)

        all_data.append(data)
        all_labels.append(labels)

    # Concatenate all data and labels
    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    
    if normalized==True:
        mean = all_data.mean(dim=[0, 2])  # Average across the batch and features dimensions
        std = all_data.std(dim=[0, 2])    # Std dev across the batch and features dimensions
    
        normalized_data = (all_data - mean[:, None]) / std[:, None]  

        dataset = TensorDataset(normalized_data, all_labels)
    
    else:
        dataset = TensorDataset(all_data, all_labels)

    # Determine batch size
    if batchSize == 'whole':
        batch_size = len(dataset)  # Set batch size to the total number of samples for a single batch
    else:
        batch_size = int(batchSize)  # Default batch size

    if val_percentage is not None:
        total_samples = len(dataset)
        val_size = int(val_percentage * total_samples)
        train_size = total_samples - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        # Create DataLoaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        if normalized == True:
            return train_loader, val_loader, mean, std
        else:
            return train_loader, val_loader
    else:
        # Create DataLoader for training only
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if normalized == True:
            return train_loader, mean, std
        else:
            return train_loader

def get_exemplars_and_normalize(itera, train_x, train_y, test_x, test_y, val_x, val_y, all_test_data, all_test_labels,
                                 all_val_data, all_val_labels, exemplar_method, holdout_size, seen_cls,
                                 model, exemp, num_new_classes, device):
    """
    Retrieves exemplars from past tasks and normalizes train, test, and validation data.
    """

    # Get exemplars from past tasks
    if exemplar_method != 'fetril':
        if itera == 0:
            train_xs, train_ys = exemp.get_exemplar_train(train_y, itera, exemplar_method, holdout_size)

        else:
            train_xs, valX, train_ys, valy = exemp.get_exemplar_train(
                train_y, itera, exemplar_method, holdout_size, all_train_class=seen_cls, model=model
            )
            all_val_data.extend(valX)
            all_val_labels.extend(valy)

            train_xs = train_xs.tolist()
            train_ys = train_ys.tolist()

        # Add current task training data
        train_xs.extend(train_x)
        train_ys.extend(train_y)

        train_xs_array = np.array(train_xs, dtype=np.float32)
        test_xs_array = np.array(all_test_data, dtype=np.float32)
        val_xs_array = np.array(all_val_data, dtype=np.float32)

        train_xs = np.transpose(train_xs_array, (0, 2, 1))
        test_xs = np.transpose(test_xs_array, (0, 2, 1))
        val_xs = np.transpose(val_xs_array, (0, 2, 1))

        train_scale, test_scale, val_scale = data_handler.scale_data(train_xs, test_xs, val_xs)

        train_xs = np.transpose(train_scale, (0, 2, 1))
        test_xs = np.transpose(test_scale, (0, 2, 1))
        val_xs = np.transpose(val_scale, (0, 2, 1))

        train_xs = [i for i in train_xs]
        all_test_data = [i for i in test_xs]
        all_val_data = [i for i in val_xs]

    else:  # FETRIL
        if itera == 0:
            train_xs, train_ys = exemp.get_exemplar_train(train_y, itera, exemplar_method, holdout_size)
            train_xs.extend(train_x)
            train_ys.extend(train_y)
            all_test_data.extend(test_x)
            all_test_labels.extend(test_y)
            train_xs_array  = np.array(train_xs, dtype=np.float32)
            test_xs_array = np.array(all_test_data, dtype=np.float32)
            val_xs_array = np.array(all_val_data, dtype=np.float32)
            train_scale_tran, test_scale_tran, val_scale_tran = train_xs_array, test_xs_array, val_xs_array
            train_xs = [i for i in train_scale_tran]
            all_test_data = [i for i in test_scale_tran]
            all_val_data = [i for i in val_scale_tran]
        else:
            for name, param in model.named_parameters():
                if "conv" in name or "bn" in name:  # Freeze conv and batch norm layers
                    param.requires_grad = False

            # Print the trainable parameters for verification
            trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
            # print("Trainable Parameters:", trainable_params)

            train_xs_array = np.array(train_x, dtype=np.float32)
            test_xs_array = np.array(test_x, dtype=np.float32)
            val_xs_array = np.array(val_x, dtype=np.float32)
            # train_scale_tran, test_scale_tran, val_scale_tran =  train_xs_array, test_xs_array, val_xs_array

            train_xs = np.transpose(train_xs_array, (0, 2, 1))
            test_xs = np.transpose(test_xs_array, (0, 2, 1))
            val_xs = np.transpose(val_xs_array, (0, 2, 1))
            train_scale, test_scale, val_scale = data_handler.scale_data(train_xs, test_xs, val_xs)
            train_scale_tran = np.transpose(train_scale,(0, 2, 1))
            test_scale_tran = np.transpose(test_scale, (0, 2, 1))
            val_scale_tran = np.transpose(val_scale, (0, 2, 1))

            fetril_trainX = np.concatenate((train_scale_tran, val_scale_tran), axis=0)
            fetril_trainy = np.concatenate((train_y, val_y))

            train_xs, val_xs, train_ys, valy = exemp.get_exemplar_train(fetril_trainy, itera, exemplar_method,
                                                                        holdout_size, train_x=fetril_trainX,
                                                                        model=model)

            all_val_data = [i for i in val_xs]
            all_val_labels = [i for i in valy]
            all_test_data.extend(test_scale_tran)
            all_test_labels.extend(test_y)

    return train_xs, train_ys, all_test_data, all_test_labels, all_val_data, all_val_labels

def inversed_map_classes_vae(mapped_array, original_values):
    reverse_mapping = {0: original_values[0], 1: original_values[1]}
    original_vae_class = [reverse_mapping[item] for item in mapped_array]
    return original_vae_class

class Exemplar:
    def __init__(self, method, exemplar_m, dataname, seed, max_size, total_classes, input_dim, reversed_label_map, reversed_orig_map, outfile,
                 device, vae_lat_sampling=None, latent_vec_filter= None, person = None, base_class=None, new_class=None, beta_start=0, beta_end=0, timesteps=0):
        self.val = {}
        self.seed = seed
        self.train = {}
        self.cur_cls = 0
        self.method = method
        self.exemplar_m = exemplar_m
        self.dataname = dataname
        self.max_size = max_size
        self.total_classes = total_classes
        self.virtual_mapping = reversed_label_map
        self.original_mapping = reversed_orig_map
        self.input_dim = input_dim
        self.outfile = outfile
        self.train_to_indices = {}  
        self.device = device
        self.vae_map_class = {}
        self.vae_model = None
        self.key_list = {}
        self.sample_size = 0
        self.latent_dim = {}
        self.vae_lat_sampling = vae_lat_sampling
        self.latent_vec_filter = latent_vec_filter
        self.size_by_class = {}
        self.mean_feature_vector = {}
        self.mean_vec = {}
        self.log_var = {}
        self.sample_z = {}
        self.min_max_vectors = {}
        self.vae_mean = {}
        self.vae_std = {}
        self.person = person
        self.base_class = base_class
        self.new_class = new_class
        self.vae_train_sample = {}
        self.num_layers = {}
        self.num_channels = {}
        self.feature_size_by_class= {}
        self.trained_class=[]
        self.ddgr_mean = {}
        self.ddgr_std = {}
        self.old_train = {}
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps


    @staticmethod
    def get_dict_by_class(features, labels):
        classwise_dict_of_features = {}
        for idx, label in enumerate(labels):
            if label not in classwise_dict_of_features:
                classwise_dict_of_features[label] = [features[idx]]
            else:
                classwise_dict_of_features[label].append(features[idx])
        return classwise_dict_of_features

    def extract_features_of_class(self, list_of_instances, feature_extractor, device):
        features = []

        for each in list_of_instances:
            feature = get_embeddings(self.input_dim, each, feature_extractor, device, self.exemplar_m)
            features.append(feature)

        return features

    def extract_features(self, model, train_dict, device, val_dict=None):
        train_dict_features = {key: self.extract_features_of_class(val, model, device) for key, val in
                               train_dict.items()}
        val_dict_features = {key: self.extract_features_of_class(val, model, device) for key, val in
                             val_dict.items()} if val_dict is not None else None
        return train_dict_features, val_dict_features

    def icarl_update(self, model, train_dict, val_dict=None):
        for old_cl, value in self.train.items():
            value = np.array(value)
            if len(value) < self.train_store_dict[old_cl]:
                self.train_store_dict[old_cl] = len(value)
            self.train[old_cl] = value[:self.train_store_dict[old_cl]]

        if val_dict is not None:
            for old_cl, value in self.val.items():
                value = np.array(value)
                self.val[old_cl] = value[:self.val_store_dict[old_cl]]

        train_dict_features, val_dict_features = self.extract_features(model, train_dict, self.device, val_dict)

        #Exemplar selections based on herding process
        selected_features = {}
        for label, features in train_dict_features.items():
            if len(features) < self.train_store_dict[label]:
                self.train_store_dict[label] = len(features)
            nearest_indices = herding.herding_selection(features, self.train_store_dict[label])
            self.train[label] = np.array(train_dict[label])[nearest_indices]

            features = np.array(features)
            if label not in selected_features:
                selected_features[label] = []

            selected_features[label].append(features[nearest_indices])
            if val_dict is not None:
                nearest_indices_val = herding.herding_selection(val_dict_features[label], self.val_store_dict[label])
                self.val[label] = np.array(val_dict[label])[nearest_indices_val]

        for class_key, features_list in selected_features.items():
            features_tensor = torch.stack([torch.tensor(features, dtype=torch.float32) for features in features_list]).squeeze(0)
            mean_feature_vector = torch.mean(features_tensor, dim=0)
            l2_norm = torch.norm(mean_feature_vector, p=2)
            normalized_mean_vector = mean_feature_vector / l2_norm + 1e-8
            self.mean_feature_vector[class_key] = normalized_mean_vector

    def random_update(self, train_dict, task_num, val_dict=None):
        for old_cl, value in self.train.items():
            value = np.array(value)
            if len(value) < self.train_store_dict[old_cl]:
                self.train_store_dict[old_cl] = len(value)
            self.train[old_cl] = value[np.random.choice(len(value), self.train_store_dict[old_cl], replace=False)]

        if val_dict is not None:
            for old_cl, value in self.val.items():
                value = np.array(value)
                self.val[old_cl] = value[np.random.choice(len(value), self.val_store_dict[old_cl], replace=False)]

        for new_cl, features in train_dict.items():
            value = np.array(features, dtype=np.float32)
            print(new_cl, len(value), self.train_store_dict[new_cl])
            if len(value) < self.train_store_dict[new_cl]:
                self.train_store_dict[new_cl] = len(value)
            size_s = self.train_store_dict[new_cl]
            random_indices = np.random.choice(len(value), self.train_store_dict[new_cl], replace=False)
            self.train[new_cl] = value[random_indices]
            self.train_to_indices[new_cl] = random_indices

        # Flatten the data and labels by stacking them
        data = np.concatenate([self.train[key] for key in self.train], axis=0)
        labels = np.concatenate([[key] * self.train[key].shape[0] for key in self.train], axis=0)

        # Save data and labels to .npy files
        real_sample_dir = './saved_real_samples/' + self.dataname + '/' + str(self.base_class) + '_' + str(
            self.new_class) + '/' + str(self.person) + '/' + str(self.max_size)
        os.makedirs(real_sample_dir, exist_ok=True)
        # Define binary file paths
        data_file_name = real_sample_dir + '/sdata.bin'
        label_file_name = real_sample_dir + '/slabel.bin'

        # Save data in raw binary format
        with open(data_file_name, 'wb') as f:
            f.write(data.tobytes())  # Convert NumPy array to raw bytes and write to file

        # Save labels in raw binary format
        with open(label_file_name, 'wb') as f:
            f.write(labels.tobytes())  # Convert labels to raw bytes and write to file
    
        if val_dict is not None:
            for new_cl, features in val_dict.items():
                value = np.array(features)
                self.val[new_cl] = value[np.random.choice(len(value), self.val_store_dict[new_cl], replace=False)]

    def fetril_update(self, model, train_dict):
        print('FeTrIL Update')
        train_dict_features, _ = self.extract_features(model, train_dict, self.device)
        for class_key, features_list in train_dict_features.items():
            print('class_key:', class_key)
            if class_key not in self.mean_feature_vector:
                # Calculate the mean of features along the batch dimension (dim=0)
                features_tensor = torch.stack(
                    [torch.tensor(features, dtype=torch.float32) for features in features_list]).squeeze(0)

                # Calculate the mean of these features along the batch dimension (dim=0)
                mean_feature_vector = torch.mean(features_tensor, dim=0)

                l2_norm = torch.norm(mean_feature_vector, p=2)
                normalized_mean_vector = mean_feature_vector / l2_norm
                # Store the mean feature vector in the new dictionary
                self.mean_feature_vector[class_key] = normalized_mean_vector
                print('self.mean_feature_vector[class_key] (', class_key, '):', self.mean_feature_vector[class_key])
                print('Shape of self.mean_feature_vector[class_key] (', class_key, '):',
                      len(self.mean_feature_vector[class_key]))
            else:
                print("Mean feature vector has already been computed.")

    def ddgr_update(self, train_dict, task_num, strategy):
        print('DDGR Update')

        self.old_train = train_dict
        print(f"Training Diffusion Model for Task {task_num}...")
        self.trained_class = list(train_dict.keys())
        # self.trained_class.extend(self.key_list)
        train_loader, val_loader, self.ddgr_mean[task_num], self.ddgr_std[task_num] = create_train_val_loaders(
            data_dict=train_dict, val_percentage=0.1, batchSize=16)

        # Initialize diffusion model
        print('self.cur_cls: ', self.cur_cls)
        print('self.timesteps: ', self.timesteps)
        print('self.beta_start, self.beta_end: ', self.beta_start, self.beta_end)

        noise_scheduler = NoiseScheduler(timesteps=self.timesteps, device=self.device, beta_start=self.beta_start,
                                         beta_end=self.beta_end)
        model_save_path = './saved_dmodel/' + self.dataname + '/' + str(self.person) + '/' + str(
            self.base_class) + '_' + str(self.new_class) + '/' + strategy + '/' + str(self.beta_start) + '_' + str(
            self.beta_end) + '/'
        saved_dmodel = model_save_path + '/dmodel.pt'
        if task_num == 0:
            # Initialize NoiseScheduler and Diffusion Model
            self.diffusion_model = Diff1D(noise_scheduler=noise_scheduler, device=self.device)
        else:
            self.diffusion_model = Diff1D(noise_scheduler=noise_scheduler, device=self.device,
                                          model_save_path=saved_dmodel)

        self.diffusion_model.train_diffusion(train_loader, val_loader, epochs=50, patience=10)
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(self.diffusion_model.model.state_dict(), saved_dmodel)
        print(f"Diffusion model saved at {model_save_path}.")
        num_params = sum(p.numel() for p in self.diffusion_model.model.parameters() if p.requires_grad)
        print(f"Number of parameters in UNet1D: {num_params}")

    def vae_update(self, train_dict, task_num, strategy):

        #here we only train and save the model. we are going to generate the sample in another function called get_exemplar_train()

        # create a class mapping of inputting classes from data for CL classifier to the VAE
        self.key_list[task_num] = sorted(list(train_dict.keys()))
        print('keys_list:', self.key_list[task_num])
        new_dict = {i: train_dict[self.key_list[task_num][i]] for i in range(len(list(train_dict.keys())))}

        train_loader, val_loader, self.vae_mean[task_num], self.vae_std[task_num] = create_train_val_loaders(new_dict, val_percentage=0.1, batchSize=16)
        self.vae_train_sample[task_num] = len(train_loader.dataset)
        print(f"Total number of samples for VAE training: {self.vae_train_sample[task_num]}")

        kl_coefficient = 0.001
        ce_coefficient = 1
        LEARNING_RATE = 0.0005
        NUM_EPOCHS = 100
        
        if task_num < len(self.new_class):
            if task_num == 0:
                vae_folder = './saved_vae/' + self.dataname + '/' + str(self.person) + '/' + str(self.base_class) + '_' + str(self.new_class) + '/' + strategy + '/' + self.vae_lat_sampling + '/' + self.latent_vec_filter 
                os.makedirs(vae_folder, exist_ok=True)
                saved_model = vae_folder + '/vae_decoder_0.pt'
                saved_classifier = vae_folder + '/vae_classifier_0.pt'
            else:
                vae_folder = './saved_vae/' + self.dataname + '/' + str(self.person) + '/' + str(self.base_class) + '_' + str(self.new_class) + '/' + strategy + '/' + self.vae_lat_sampling + '/' + self.latent_vec_filter 
                os.makedirs(vae_folder, exist_ok=True)
                saved_model = vae_folder + '/vae_decoder_' + str(task_num) + '.pt'
                saved_classifier = vae_folder + '/vae_classifier_' + str(task_num) + '.pt'

            self.latent_dim[task_num] = 64
            vae_model = VAE(latent_dim=self.latent_dim[task_num], output_dim=len(train_dict))
            vae_model.to(self.device)
            
            optimizer = torch.optim.Adam(vae_model.parameters(), lr=LEARNING_RATE)

            model, log_dict = train_vae(num_epochs=NUM_EPOCHS, model=vae_model,
                                        optimizer=optimizer, val_loader=val_loader, device=self.device,
                                        train_loader=train_loader,
                                        kl_coefficient=kl_coefficient,
                                        ce_coefficient=ce_coefficient, saved_model=saved_model,
                                        saved_classifier = saved_classifier)
            
            self.mean_vec[task_num], self.log_var[task_num], self.sample_z[task_num] = generate_latent_space(model, train_dict, self.vae_mean[task_num], self.vae_std[task_num], device=self.device) 

            if self.vae_lat_sampling == 'boundary_box':
                self.min_max_vectors[task_num] = {}

                for label, vectors in self.sample_z[task_num].items():

                    log_var_tensor = torch.from_numpy(self.log_var[task_num][label]).to(dtype=torch.float32, device=self.device)
                    vectors_tensor = torch.from_numpy(vectors).to(dtype=torch.float32, device=self.device)  
                    mean_log_var = torch.mean(log_var_tensor, dim=0)
                    std_devs = torch.exp(0.5 * log_var_tensor)

                    # Calculate the min and max values for each dimension
                    min_vals = torch.min(vectors_tensor, dim=0)[0]  
                    max_vals = torch.max(vectors_tensor, dim=0)[0]  
                    
                    self.min_max_vectors[task_num][label] = {
                        'min': min_vals,  
                        'max': max_vals,
                        'mean_log_var': mean_log_var,
                    }

            elif self.vae_lat_sampling == 'gmm':
                self.min_max_vectors[task_num] = {}

                for label, vectors in self.mean_vec[task_num].items():

                    log_var_tensor = torch.from_numpy(self.log_var[task_num][label]).to(dtype=torch.float32,
                                                                                        device=self.device)
                    normalized_latent_means = torch.from_numpy(vectors)  # convert and move to device in one step
                    mean_log_var = torch.mean(log_var_tensor, dim=0)

                    # Reduce dimensionality with PCA
                    pca = PCA(n_components=0.70)  # Retain 95% of variance
                    normalized_vectors_tensor = pca.fit_transform(normalized_latent_means)

                    best_score = -1
                    best_clusters = None

                    for n_clusters in range(2, 10):

                        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                        cluster_labels = kmeans.fit_predict(normalized_vectors_tensor)
                        score = silhouette_score(normalized_vectors_tensor, cluster_labels)
                        if score > best_score:
                            best_score = score
                            best_clusters = kmeans

                    # Initialize GMM with KMeans parameters
                    gmm = GaussianMixture(n_components=best_clusters.n_clusters, random_state=0)

                    # Use KMeans cluster centers as initial means
                    gmm.means_init = best_clusters.cluster_centers_

                    # Use KMeans cluster weights as initial weights
                    cluster_weights = np.bincount(best_clusters.labels_) / len(best_clusters.labels_)
                    gmm.weights_init = cluster_weights

                    # Calculate initial covariances based on KMeans assignments
                    covariances = []
                    for i in range(best_clusters.n_clusters):
                        points_in_cluster = normalized_vectors_tensor[best_clusters.labels_ == i]
                        covariances.append(empirical_covariance(points_in_cluster))
                    gmm.covariances_init = np.array(covariances)

                    # Fit GMM
                    gmm.fit(normalized_vectors_tensor)

                    gmm_folder = './saved_gmm/' + self.dataname + '/' + str(self.person) + '/' + str(
                        self.base_class) + '_' + str(
                        self.new_class) + '/' + strategy + '/' + self.vae_lat_sampling + '/' + self.latent_vec_filter
                    os.makedirs(gmm_folder, exist_ok=True)
                    saved_gmm = gmm_folder + '/gmm_model_' + str(task_num) + '.pkl'
                    joblib.dump(gmm, saved_gmm)

                    self.min_max_vectors[task_num][label] = {
                        'gmm': gmm,
                        'pca': pca
                    }

    def get_holdout_size_by_labels(self, count_of_labels, store_num, val=False):
        sorted_count_dict = sorted(count_of_labels, key=count_of_labels.get)
        dict_of_store_size = {}
        for label in sorted_count_dict:
            true_size = min(store_num, count_of_labels[label])
            dict_of_store_size[label] = true_size
        for old_cl in self.train:
            if val:
                dict_of_store_size[old_cl] = min(store_num, len(self.val[old_cl]))
            else:
                dict_of_store_size[old_cl] = min(store_num, len(self.train[old_cl]))

        return dict_of_store_size

    def update(self, strategy, cls_num, task_num, train=None, val=None, model = None):
        train_x, train_y = train
   
        self.cur_cls += cls_num

        if strategy == 'taskvae' or strategy == 'taskvae_ratio' or strategy == 'fetril' or strategy == 'ddgr':
            samples_per_class = self.sample_size
        else:
            samples_per_class = self.max_size / self.cur_cls

        train_percent = 0.9 if val is not None else 1.0
        val_percent = 1 - train_percent
        val_store_num = math.ceil(
            samples_per_class * val_percent)  
        train_store_num = math.floor(
            samples_per_class) - val_store_num

        train_y_counts = Counter(train_y)

        if strategy != 'taskvae' and strategy != 'taskvae_ratio' and strategy != 'fetril' and strategy != 'ddgr':
            self.train_store_dict = self.get_holdout_size_by_labels(train_y_counts, train_store_num)
        train_dict, val_dict = self.get_dict_by_class(train_x, train_y), None
        self.size_by_class =  {key: len(value) for key, value in train_dict.items()}
        print('self.size_by_class', self.size_by_class)

        new_labels = set(train_y)

        mapped_new_classes = [self.original_mapping[self.virtual_mapping[each]] for each in new_labels]
        mapped_old_classes = [self.original_mapping[self.virtual_mapping[each]] for each in self.train.keys()]

        if strategy != 'taskvae' and strategy != 'taskvae_ratio' and strategy != 'fetril' and strategy != 'ddgr':
            if strategy != 'icarl':
                exemplar_details = "\nNew classes: {}, Old classes: {}\nUpdated memory size for each old class: {} [Train size={}]" \
                                   "".format(
                    mapped_new_classes, mapped_old_classes, int(samples_per_class), self.train_store_dict)
            else:
                exemplar_details = "\nNew classes: {}, Old classes: {}\nUpdated memory size for each old class: {} [Train size={}]" \
                                   "".format(
                    mapped_new_classes, mapped_old_classes, int(samples_per_class), self.train_store_dict)

        elif strategy == 'fetril':
            if task_num == 0:
                self.feature_size_by_class[task_num] = None

            exemplar_details = "\nNew classes: {}, Old classes: {}\nUpdated memory size for each old class: {}" \
                               "".format(
                mapped_new_classes, mapped_old_classes, self.feature_size_by_class[task_num])
        else:
            exemplar_details = "\nNew classes: {}, Old classes: {}\nUpdated memory size for each old class: {} [Train size={}]" \
                               "".format(
                mapped_new_classes, mapped_old_classes, int(samples_per_class), self.sample_size)

        if val is not None:
            val_x, val_y = val
            assert len(set(val_y)) == len(set(train_y))
            val_dict = self.get_dict_by_class(val_x, val_y)
            val_y_counts = Counter(val_y)
            self.val_store_dict = self.get_holdout_size_by_labels(val_y_counts, val_store_num, val=True)
            exemplar_details += f", Val sizes: {self.val_store_dict}"

        exemplar_details += "\n"
        file = open(self.outfile, 'a')
        file.write(exemplar_details)
        file.close()
        print(exemplar_details)

        if strategy == 'icarl':
            self.icarl_update(model, train_dict, val_dict=val_dict)
        elif strategy == 'random':
            self.random_update(train_dict, task_num, val_dict=val_dict)
        elif strategy == 'taskvae' or strategy == 'taskvae_ratio':
            self.vae_update(train_dict, task_num, strategy)
        elif strategy == 'fetril':
            self.fetril_update(model, train_dict)
        elif strategy == 'ddgr':
            self.ddgr_update(train_dict, task_num, strategy)

        if val is not None:
            print(f' {self.cur_cls}, {len(list(self.val.keys()))}, class num: {cls_num},'
                  f' val_store_nums: {self.val_store_dict}, len: {len(list(self.val.values())[0])}')
            assert self.cur_cls == len(list(self.val.keys()))
            for key, value in self.val.items():
                assert len(self.val[key]) == self.val_store_dict[key], print(key, len(self.val[key]),
                                                                             self.val_store_dict[key])

        if strategy != 'taskvae' and strategy != 'taskvae_ratio' and strategy != 'fetril' and strategy != 'ddgr':
            print('self.cur_cls', self.cur_cls, 'list(self.train.keys())', list(self.train.keys()))
            total_size = 0
            for key, value in self.train.items():
                total_size += len(value)
                print(f"Total exemplar size: {total_size}")
                assert len(self.train[key]) == self.train_store_dict[key], print(key, len(self.train[key]),
                                                                                 self.train_store_dict[key])
                print(f"Class: {key}, No. of exemplars: {len(value)}")

    def generate_sample_by_tasks(self, saved_vae_model, saved_classifier, task_num, sample_size):

        vae_m = Decoder(latent_dim=self.latent_dim[task_num])
        vae_c = Classifier(latent_dim=self.latent_dim[task_num], output_dim=len(self.key_list[task_num]))

        vae_m.load_state_dict(torch.load(saved_vae_model))
        vae_m.to(self.device)
        vae_m.eval()

        vae_c.load_state_dict(torch.load(saved_classifier))
        vae_c.to(self.device)
        vae_c.eval()
        
        # Create a dictionary to hold the data grouped by label
        data_dict = {}

        while True:

            gen_data, gen_label = generate_sample(vae_m, vae_c, self.min_max_vectors[task_num], sample_size,
                                                  self.device, sample_strategy=self.vae_lat_sampling,
                                                  latent_vec_filter=self.latent_vec_filter)

            unique_values = np.unique(gen_label)
            print("Generated class(es) from VAE:", unique_values)

            for label, data in zip(gen_label, gen_data):
                if label not in data_dict:
                    data_dict[label] = []
                data_dict[label].append(data)
            if len(unique_values) == len(self.key_list[task_num]):
                all_labels_reached = all(len(data_list) >= sample_size for data_list in data_dict.values())
            else:
                all_labels_reached = False
            if all_labels_reached:
                break  

        for new_cl, X in data_dict.items():
            self.train[new_cl] = np.array(data_dict[new_cl])[:sample_size]

        for new_cl, _ in data_dict.items():
            print('self.train[new_cl] len:', len(self.train[new_cl]))


    def get_exemplar_train(self, train_y, task_num, strategy, vae_sample=1,val=True, train_x=None, model=None, all_train_class=None):
        if strategy == 'taskvae' or strategy == 'taskvae_ratio':
            print('vae is here')

            if task_num > 0:    
                class_frequency = {label: train_y.count(label) for label in set(train_y)}
                if strategy == 'taskvae':
                    self.sample_size = int(sum(class_frequency.values()) / len(class_frequency))
                elif strategy == 'taskvae_ratio':
                    self.sample_size = int((sum(class_frequency.values()) / len(class_frequency)) * vae_sample)
                print('self.sample_size:', self.sample_size)

                for task in range(0, task_num):
                    print('Generating data from Task', task)
                    saved_model = ('./saved_vae/' + self.dataname + '/' + str(self.person) + '/' +
                                   str(self.base_class) + '_' + str(self.new_class) + '/' + strategy + '/' +
                                   self.vae_lat_sampling + '/' + self.latent_vec_filter + '/vae_decoder_' + str(task) +
                                   '.pt')
                    saved_classifier = ('./saved_vae/' + self.dataname + '/' + str(self.person) + '/' +
                                       str(self.base_class) + '_' + str(self.new_class) + '/' + strategy + '/' +
                                       self.vae_lat_sampling + '/' + self.latent_vec_filter + '/vae_classifier_' +
                                       str(task) + '.pt')
                    self.generate_sample_by_tasks(saved_model, saved_classifier, task, self.sample_size)

            exemplar_train_x = []
            exemplar_train_y = []
            for key, value in self.train.items():
                for train_x in value:
                    exemplar_train_x.append(train_x)
                    exemplar_train_y.append(key)
            if task_num > 0:
                exemplar_train_x = np.array(exemplar_train_x)
                exemplar_train_y = np.array(exemplar_train_y)
                num_classes = len(np.unique(exemplar_train_y))

                test_size = int(0.1 * len(exemplar_train_y))
                if test_size < num_classes:
                    test_size = num_classes

                exemplar_trainX, exemplar_val_X, exemplar_trainy, exemplar_val_y = train_test_split(exemplar_train_x, exemplar_train_y, test_size=test_size, random_state=42, shuffle=True, stratify=exemplar_train_y)
                
                return exemplar_trainX, exemplar_val_X, exemplar_trainy, exemplar_val_y
            else:
                return exemplar_train_x, exemplar_train_y

        elif strategy == 'fetril':
            print('fetril is here')
            centroids_new = {}
            pseudo_features = {}
            closest_class_pair = {}
            if task_num > 0:
                train_new = self.get_dict_by_class(train_x, train_y)
                train_new_features, _ = self.extract_features(model, train_new, self.device)

                for class_new, features_new in train_new_features.items():
                    print('class_new:', class_new)
                    features_new = np.array(features_new)

                    if np.isinf(features_new).any():
                        print(f"Fixing Inf values for class {class_new}")
                        features_new = np.where(np.isinf(features_new), 1e-8, features_new)

                    # Handle near-zero values: Add small Gaussian noise
                    if np.all(np.abs(features_new) < 1e-8):  # Check if all values are very small or zero
                        print(f"Fixing near-zero data for class {class_new}")
                        features_new += np.random.normal(0, 1e-8, features_new.shape)

                    pseudo_features[class_new] = features_new

                    # Calculate the mean of features along the batch dimension (dim=0)
                    features_tensor = torch.stack(
                        [torch.tensor(features, dtype=torch.float32) for features in features_new]).squeeze(0)

                    # Calculate the mean of these features along the batch dimension (dim=0)
                    mean_feature_new = torch.mean(features_tensor, dim=0)

                    l2_norm = torch.norm(mean_feature_new, p=2)
                    normalized_feature_new = mean_feature_new / l2_norm
                    # Store the mean feature vector in the new dictionary
                    centroids_new[class_new] = normalized_feature_new

                # Convert centroids to arrays for cosine similarity calculation
                old_class_keys = list(self.mean_feature_vector.keys())
                new_class_keys = list(centroids_new.keys())

                centroids_old_array = np.array([self.mean_feature_vector[c] for c in old_class_keys])
                centroids_new_array = np.array([centroids_new[c] for c in new_class_keys])

                print('centroids_old_array: ', centroids_old_array)
                print('centroids_new_array: ', centroids_new_array)

                # Compute cosine similarity between old and new class centroids
                similarity_matrix = cosine_similarity(centroids_old_array, centroids_new_array)

                # Find the closest new class for each old class
                for i, old_class_key in enumerate(old_class_keys):
                    closest_new_class_idx = np.argmax(similarity_matrix[i])
                    closest_new_class_key = new_class_keys[closest_new_class_idx]

                    closest_class_pair[old_class_key] = closest_new_class_key

                    # Retrieve the features of the closest new class
                    features_new_class = train_new_features[closest_new_class_key]

                    # Compute pseudo-features using the FeTrIL formula
                    centroid_old = self.mean_feature_vector[old_class_key]
                    centroid_new = centroids_new_array[closest_new_class_idx]

                    # Ensure all inputs are NumPy arrays
                    features_new_class = np.array(features_new_class)
                    centroid_new = np.array(centroid_new)
                    centroid_old = np.array(centroid_old)

                    # Reshape centroids for broadcasting
                    centroid_new = centroid_new.reshape(1, -1)
                    centroid_old = centroid_old.reshape(1, -1)

                    pseudo_features_old_class = features_new_class - centroid_new + centroid_old  # Formula

                    # Store the pseudo-features for the old class
                    pseudo_features[old_class_key] = pseudo_features_old_class

                for all_classes in list(pseudo_features.keys()):
                    self.feature_size_by_class[task_num][all_classes] = len(pseudo_features[all_classes])

                for class_new, features_new in train_new_features.items():
                    if class_new not in self.mean_feature_vector:
                        self.mean_feature_vector[class_new] = centroids_new[class_new]
            else:
                self.feature_size_by_class = defaultdict(dict)

            exemplar_train_x = []
            exemplar_train_y = []

            if task_num == 0:
                for key, value in self.train.items():
                    for train_x in value:
                        exemplar_train_x.append(train_x)
                        exemplar_train_y.append(key)
            if task_num > 0:
                for key, value in pseudo_features.items():
                    for train_x in value:
                        exemplar_train_x.append(train_x)
                        exemplar_train_y.append(key)
                exemplar_train_x = np.array(exemplar_train_x)
                exemplar_train_y = np.array(exemplar_train_y)
                num_classes = len(np.unique(exemplar_train_y))

                test_size = int(0.1 * len(exemplar_train_y))
                if test_size < num_classes:
                    test_size = num_classes

                exemplar_trainX, exemplar_val_X, exemplar_trainy, exemplar_val_y = train_test_split(exemplar_train_x,
                                                                                                    exemplar_train_y,
                                                                                                    test_size=test_size,
                                                                                                    random_state=42,
                                                                                                    shuffle=True,
                                                                                                    stratify=exemplar_train_y)

                print('exemplar_train_x.shape:', exemplar_train_x.shape)
                print('exemplar_train_y.shape:', exemplar_train_y.shape)
                print('exemplar_trainX.shape:', exemplar_trainX.shape)
                print('exemplar_trainy.shape:', exemplar_trainy.shape)
                print('exemplar_val_X.shape:', exemplar_val_X.shape)
                print('exemplar_val_y.shape:', exemplar_val_y.shape)

                # Unique class labels and counts for validation set
                unique_classes, val_counts = np.unique(exemplar_val_y, return_counts=True)
                print("\nClass counts in validation set:")
                for class_label, count in zip(unique_classes, val_counts):
                    print(f"Class {class_label}: {count}")

                return exemplar_trainX, exemplar_val_X, exemplar_trainy, exemplar_val_y
            else:
                return exemplar_train_x, exemplar_train_y

        elif strategy == 'ddgr':
            print('DDGR is here')
            class_frequency = {label: train_y.count(label) for label in set(train_y)}

            self.sample_size = int(sum(class_frequency.values()) / len(class_frequency))

            if task_num > 0:
                # Generate data for old classes
                self.train = self.diffusion_model.guided_reverse_diffusion(model, self.trained_class, self.sample_size,
                                                                           self.ddgr_mean[task_num - 1],
                                                                           self.ddgr_std[task_num - 1])
                print('self.train.keys(): ', list(self.train.keys()))
                print('self.old_train.keys(): ', list(self.old_train.keys()))
                compare_training_and_generated_data(self.old_train, self.train)
                print(f"Generated data stored for classes: {list(self.train.keys())}")

            exemplar_train_x = []
            exemplar_train_y = []
            for key, value in self.train.items():
                for train_x in value:
                    exemplar_train_x.append(train_x)
                    exemplar_train_y.append(key)
            if task_num > 0:
                exemplar_train_x = np.array(exemplar_train_x)
                exemplar_train_y = np.array(exemplar_train_y)
                num_classes = len(np.unique(exemplar_train_y))

                test_size = int(0.1 * len(exemplar_train_y))
                if test_size < num_classes:
                    test_size = num_classes

                print('exemplar_train_x.shape:', exemplar_train_x.shape)
                print('exemplar_train_y.shape:', exemplar_train_y.shape)

                exemplar_trainX, exemplar_val_X, exemplar_trainy, exemplar_val_y = train_test_split(exemplar_train_x,
                                                                                                    exemplar_train_y,
                                                                                                    test_size=test_size,
                                                                                                    random_state=42,
                                                                                                    shuffle=True,
                                                                                                    stratify=exemplar_train_y)

                print('exemplar_train_x.shape:', exemplar_train_x.shape)
                print('exemplar_train_y.shape:', exemplar_train_y.shape)
                print('exemplar_trainX.shape:', exemplar_trainX.shape)
                print('exemplar_trainy.shape:', exemplar_trainy.shape)
                print('exemplar_val_X.shape:', exemplar_val_X.shape)
                print('exemplar_val_y.shape:', exemplar_val_y.shape)

                return exemplar_trainX, exemplar_val_X, exemplar_trainy, exemplar_val_y
            else:
                return exemplar_train_x, exemplar_train_y

        else:
            exemplar_train_x = []
            exemplar_train_y = []
            for key, value in self.train.items():
                for train_x in value:
                    exemplar_train_x.append(train_x)
                    exemplar_train_y.append(key)
            if task_num > 0:
                exemplar_train_x = np.array(exemplar_train_x)
                exemplar_train_y = np.array(exemplar_train_y)
                num_classes = len(np.unique(exemplar_train_y))
                test_size = int(0.1 * len(exemplar_train_y))
                if test_size < num_classes:
                    test_size = num_classes

                exemplar_trainX, exemplar_val_X, exemplar_trainy, exemplar_val_y = train_test_split(exemplar_train_x, exemplar_train_y, test_size=test_size, random_state=42, shuffle=True, stratify=exemplar_train_y)
                
                return exemplar_trainX, exemplar_val_X, exemplar_trainy, exemplar_val_y
            else:
                return exemplar_train_x, exemplar_train_y

    def get_exemplar_val(self, classes_to_exclude=[]):
        exemplar_val_x = []
        exemplar_val_y = []
        for key, value in self.val.items():
            if key not in classes_to_exclude:
                for val_x in value:
                    exemplar_val_x.append(val_x)
                    exemplar_val_y.append(key)
        return exemplar_val_x, exemplar_val_y

    def get_cur_cls(self):
        return self.cur_cls


