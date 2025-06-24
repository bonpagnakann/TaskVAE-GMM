import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
import math


class UNet1D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet1D, self).__init__()
        self.input_channels = input_channels
        
        # Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, input_channels * 2),
        )

        # Encoder (Down-sampling)
        self.enc1 = self.conv_block(input_channels + input_channels * 2, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)

        # Bottleneck
        self.bottleneck = self.conv_block(128, 256)
        #self.bottleneck = self.conv_block(32, 64)

        # Decoder (Up-sampling)
        self.up4 = self.upconv(256, 128)
        self.dec4 = self.conv_block(256, 128)

        self.up3 = self.upconv(128, 64)
        self.dec3 = self.conv_block(128, 64)

        self.up2 = self.upconv(64, 32)
        self.dec2 = self.conv_block(64, 32)

        self.up1 = self.upconv(32, 16)
        self.dec1 = self.conv_block(32, 16)
        
        # Final Output Layer
        self.final = nn.Conv1d(16, output_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_channels),
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, t):
        # Embed time and incorporate it into the input
        t_emb = self.time_embed(t.unsqueeze(1).float()).view(x.size(0), -1, 1)
        t_emb = t_emb.expand(-1, -1, x.size(2))
        t_emb = t_emb.to(x.device)

        x = torch.cat([x, t_emb], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool1d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool1d(e2, kernel_size=2))
        e4 = self.enc4(F.max_pool1d(e3, kernel_size=2))

        # Bottleneck
        b = self.bottleneck(F.max_pool1d(e4, kernel_size=2))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        # Final output
        out = self.final(d1)
        return out


class NoiseScheduler:
    def __init__(self, timesteps, device, beta_start, beta_end):
        self.timesteps = timesteps
        self.device = device

        # Linear schedule for beta values
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(self.device)

        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        

    def add_noise(self, x, noise, t):
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1).to(self.device)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1).to(self.device)
        return sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise


class Diff1D:
    def __init__(self, noise_scheduler, device, model_save_path=None):
        self.device = device
        self.noise_scheduler = noise_scheduler
        
        if model_save_path:
            self.model = UNet1D(input_channels=6, output_channels=6).to(self.device)
            self.model.load_state_dict(torch.load(model_save_path))
        else:
            self.model = UNet1D(input_channels=6, output_channels=6).to(self.device)

    def guided_reverse_diffusion(self, cl_classifier, y_labels, num_samples, mean, std):

        batch_size = min(50, num_samples)
        generated_data = {label: [] for label in y_labels}
        class_counts = {label: 0 for label in y_labels}
        cl_classifier.eval()
        self.model.eval()
        print('self.sample_size: ', num_samples)
        print('class_counts:', class_counts)
        print('y_labels:', y_labels)
        for label in y_labels:
            print('generated_data[', label, ']:', len(generated_data[label]))

        while any(count < num_samples for count in class_counts.values()):
            for label in y_labels:
                remaining = num_samples - class_counts[label]
                if remaining <= 0:
                    continue

                current_batch_size = min(batch_size, remaining)
                y = torch.full((current_batch_size,), label, device=self.device, dtype=torch.long)

                # Start with Gaussian noise
                x_N = torch.randn((current_batch_size, 6, 128), device=self.device)

                for n in reversed(range(self.noise_scheduler.timesteps)):
                    z = torch.randn_like(x_N) if n > 0 else torch.zeros_like(x_N)
                    alpha_t = self.noise_scheduler.alpha[n]
                    alpha_bar_t = self.noise_scheduler.alpha_bar[n]
                    sigma_t = torch.sqrt(1 - alpha_t)

                    # Predict noise for the batch
                    eps_theta = self.model(x_N, torch.tensor([n] * current_batch_size, device=self.device))

                    mu = (x_N / torch.sqrt(alpha_t)) - (((1 - alpha_t) / (torch.sqrt(alpha_t)*torch.sqrt(1 - alpha_bar_t))) * eps_theta)
                    mu.requires_grad_()

                    # Classifier guidance
                    classifier_logits = cl_classifier(mu)
                    classifier_logits = torch.nn.functional.softmax(classifier_logits, dim=-1)
                    classifier_loss = F.cross_entropy(classifier_logits, y)

                    gradients = torch.autograd.grad(classifier_loss, mu, retain_graph=True)[0]
                    torch.nn.utils.clip_grad_norm_(gradients, max_norm=1.0)
                    guidance_scale = 1

                    x_N = mu + ((sigma_t ** 2)*gradients*guidance_scale) + (sigma_t * z)

                # Add the generated samples to the corresponding class
                # **Denormalize the Generated Samples**
                denormalized_data = x_N
                generated_data[label].extend(denormalized_data.detach().cpu().numpy())
                class_counts[label] += current_batch_size

            # Print count of generated samples for the current class
            print(f"Generated samples by classes: {class_counts}")

        return generated_data


    def train_diffusion(self, data_loader, val_loader=None, epochs=10, patience=10):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0

        print('Training Starts.....')
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Training Phase
            for batch_idx, (x, _) in enumerate(data_loader):
                x = x.to(self.device)
                # Add noise to the input
                t = torch.randint(0, self.noise_scheduler.timesteps, (x.size(0),), device=self.device)
                noise = torch.randn_like(x).to(self.device)
                noisy_x = self.noise_scheduler.add_noise(x, noise, t)

                # Predict noise and compute loss
                optimizer.zero_grad()
                predicted_noise = self.model(noisy_x, t)
                loss = F.mse_loss(predicted_noise, noise)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"NaN detected in gradients of layer: {name}")
                    if torch.isnan(param).any():
                        print(f"NaN detected in weights of layer: {name}")

                optimizer.step()

                epoch_loss += loss.item()

            # Store train loss for visualization
            train_losses.append(epoch_loss / len(data_loader))
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss / len(data_loader):.6f}")

            # Validation Phase
            self.model.eval()
            val_loss = 0.0
            

            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(self.device)
                    t = torch.randint(0, self.noise_scheduler.timesteps, (x.size(0),), device=self.device)
                    noise = torch.randn_like(x).to(self.device)
                    noisy_x = self.noise_scheduler.add_noise(x, noise, t)

                    predicted_noise = self.model(noisy_x, t)
                    loss = F.mse_loss(predicted_noise, noise)
                    val_loss += loss.item()

            val_losses.append(val_loss / len(val_loader))
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss / len(val_loader):.6f}")

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model
                torch.save(self.model.state_dict(), "best_diffusion_model.pth")
                print("Model improved and saved.")
            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        self.visualize_training(train_losses, val_losses)

    def visualize_training(self, train_losses, val_losses):

        # Ensure the directory exists
        save_dir = "./dmodel_train"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "train_val_loss.png")

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", marker="o")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", marker="o")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig(save_path)
        plt.close()

        print(f"Training and validation loss plot saved to {save_path}")

def compare_training_and_generated_data(training_data, generated_data):
    all_classes = set(training_data.keys()).union(set(generated_data.keys()))

    for cls in all_classes:
        train_data_cls = np.array(training_data.get(cls, []))
        gen_data_cls = np.array(generated_data.get(cls, []))

        if train_data_cls.size == 0 or gen_data_cls.size == 0:
            print(f"Class {cls}: Data missing in one of the datasets.")
            continue

        # Compute statistics
        train_mean = np.mean(train_data_cls, axis=(0, 2))
        train_std = np.std(train_data_cls, axis=(0, 2))
        gen_mean = np.mean(gen_data_cls, axis=(0, 2))
        gen_std = np.std(gen_data_cls, axis=(0, 2))

        num_channels = train_mean.shape[0]
        print(f"Class {cls}:")
        for channel in range(num_channels):
            print(f"  Channel {channel + 1}:")
            print(f"    Training Mean: {train_mean[channel]:.4f}, Training Std: {train_std[channel]:.4f}")
            print(f"    Generated Mean: {gen_mean[channel]:.4f}, Generated Std: {gen_std[channel]:.4f}")
        print("-")