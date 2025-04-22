import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# === Custom Dataset Class ===
class FamilyFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.genetic_factors = []
        self.external_factors = []
        self.variety_factors = []

        for continent in os.listdir(root_dir):
            continent_path = os.path.join(root_dir, continent)
            for country in os.listdir(continent_path):
                country_path = os.path.join(continent_path, country)
                for family in os.listdir(country_path):
                    family_path = os.path.join(country_path, family)
                    for image_file in os.listdir(family_path):
                        image_path = os.path.join(family_path, image_file)
                        self.image_paths.append(image_path)

                        # Parse filename to extract factors
                        factors = self._parse_filename(image_file)
                        self.genetic_factors.append({
                            "gender": factors["gender"],
                            "skin_color": factors["skin_color"]
                        })
                        self.external_factors.append({
                            "age": factors["age"],
                            "emotion": factors["emotion"],
                            "glasses": factors["glasses"],
                            "moustache": factors["moustache"]
                        })
                        self.variety_factors.append(torch.randint(0, 7, (1,)).item())

    def _parse_filename(self, filename):
        attributes = filename[:8]
        return {
            "image_id": attributes[0:2],
            "gender": int(attributes[2]),
            "skin_color": int(attributes[3]),
            "age": int(attributes[4]),
            "emotion": int(attributes[5]),
            "glasses": int(attributes[6]),
            "moustache": int(attributes[7])
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        genetic_factors = self.genetic_factors[idx]
        external_factors = self.external_factors[idx]
        variety_factor = self.variety_factors[idx]

        genetic_tensor = torch.tensor([genetic_factors["gender"], genetic_factors["skin_color"]]).float()
        external_tensor = torch.tensor([external_factors["age"], external_factors["emotion"], external_factors["glasses"], external_factors["moustache"]]).float()

        image_id = self._parse_filename(os.path.basename(img_path))["image_id"]

        return image, genetic_tensor, external_tensor, variety_factor, image_id

    def apply_variety_factor(self, image, variety_factor):
        brightness_factor = 1.0 + (variety_factor - 3) * 0.1
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        contrast_factor = 1.0 + (variety_factor - 3) * 0.05
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)
        rotation_angle = (variety_factor - 3) * 5
        image = image.rotate(rotation_angle, expand=True)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return image

# === Gradient Penalty Function ===
def gradient_penalty(discriminator, real_images, fake_images):
    batch_size = real_images.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1).to(real_images.device)  # Random tensor for interpolation
    interpolated_images = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated_images.requires_grad_(True)

    # Compute discriminator output
    disc_output = discriminator(interpolated_images)[0]  # Only need the adversarial output

    # Compute gradients
    gradients = torch.autograd.grad(outputs=disc_output, inputs=interpolated_images,
                                     grad_outputs=torch.ones_like(disc_output),
                                     create_graph=True, retain_graph=True)[0]

    # Compute gradient penalty
    gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)  # L2 norm
    return ((gradient_norm - 1) ** 2).mean()  # Penalty term

# === Orthogonality Loss ===
class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()

    def forward(self, x, y):
        x_reshaped = x.view(x.size(0), x.size(1), -1)
        y_reshaped = y.view(y.size(0), y.size(1), -1)

        # Compute Gram matrices
        gram_x = torch.bmm(x_reshaped, x_reshaped.transpose(1, 2))
        gram_y = torch.bmm(y_reshaped, y_reshaped.transpose(1, 2))

        # Normalize Gram matrices
        gram_x = gram_x / (x.size(2) * x.size(3))
        gram_y = gram_y / (y.size(2) * y.size(3))

        # Compute orthogonality loss
        loss = torch.mean((gram_x @ gram_y) ** 2)
        return loss

# === Attention Mechanism ===
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        attention = torch.bmm(query, key)
        attention = nn.functional.softmax(attention, dim=-1)

        value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        out = self.gamma * out + x
        return out

# === Model Architectures ===
class ParentEncoder(nn.Module):
    def __init__(self):
        super(ParentEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(1024 * 4 * 4, 256)
        )

    def forward(self, x):
        return self.model(x)

class ParentGenerator(nn.Module):
    def __init__(self):
        super(ParentGenerator, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(256 + 2 + 4, 1024 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (1024, 4, 4)),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            AttentionBlock(512),  # Add Attention
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            AttentionBlock(256),  # Add Attention
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z, external_factors, genetic_factors):
        z_combined = torch.cat((z, external_factors, genetic_factors), dim=1)
        return self.decoder(z_combined)

class ChildEncoder(nn.Module):
    def __init__(self):
        super(ChildEncoder, self).__init__()
        self.vgg = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # VGG Block 1
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # VGG Block 2
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # VGG Block 3
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 480)  # Project to 480 dimensions
        )

    def forward(self, x):
        features = self.vgg(x)
        return self.mlp(features)

class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

class ChildGenerator(nn.Module):
    def __init__(self):
        super(ChildGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(487, 128 * 16 * 16),  # Adjusted to match the concatenated input size
            PixelNorm(),
            nn.PReLU(),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),  # Block 1
            PixelNorm(),
            nn.PReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Block 2
            PixelNorm(),
            nn.PReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),  # Block 3
            nn.Tanh()  # Output layer
        )

    def forward(self, z, external_factors, genetic_factors, variety_factor):
        variety_factor = variety_factor.view(variety_factor.size(0), 1)
        z_combined = torch.cat((z, external_factors, genetic_factors, variety_factor), dim=1)
        return self.model(z_combined)

class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.adv_branch = nn.Linear(512 * 8 * 8, 1)
        self.class_branch = nn.Sequential(
            nn.Linear(512 * 8 * 8, 4),  # Output 4 factors
            nn.Sigmoid()  # For binary outputs
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        adv_output = self.adv_branch(features)
        class_output = self.class_branch(features)
        return adv_output, class_output

class GeneticDiscriminator(nn.Module):
    def __init__(self):
        super(GeneticDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1)
        )

    def forward(self, x):
        return self.model(x)

class MappingFunctionT(nn.Module):
    def __init__(self):
        super(MappingFunctionT, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(4, 128),  # Input: gf_x, gm_x
            nn.ReLU()
        )
        self.branches = nn.ModuleList([nn.Linear(128, 1) for _ in range(2)])  # Adjusted to predict 2 genetic factors

    def forward(self, gf_x, gm_x):
        combined = torch.cat((gf_x, gm_x), dim=1)
        x = self.head(combined)
        predictions = [branch(x) for branch in self.branches]
        return torch.cat(predictions, dim=1)  # Concatenate predictions

# === Loss Functions ===
def l1_loss(x, y):
    return nn.L1Loss()(x, y)

def classification_loss(external_factors, predictions):
    return -torch.mean(external_factors * torch.log(predictions) + (1 - external_factors) * torch.log(1 - predictions))

def adversarial_loss(discriminator, real_images, fake_images):
    # Get the real and fake outputs
    real_output = discriminator(real_images)
    fake_output = discriminator(fake_images)

    # Create target tensors with the correct shape
    real_target = torch.ones(real_images.size(0), 1).to(real_images.device)  # Single value for each real image
    fake_target = torch.zeros(fake_images.size(0), 1).to(real_images.device)  # Single value for each fake image

    # Compute the losses
    real_loss = nn.BCEWithLogitsLoss()(real_output.view(-1, 1), real_target)  # Reshape if necessary
    fake_loss = nn.BCEWithLogitsLoss()(fake_output.view(-1, 1), fake_target)  # Reshape if necessary

    return real_loss + fake_loss

# === Training Function ===
def train_model(train_loader, parent_encoder, parent_generator, child_encoder, child_generator,
                discriminator, genetic_discriminator, mapping_function, num_epochs=10,
                save_dir='model_weights', generated_images_dir='generated_images'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directory for generated images
    os.makedirs(generated_images_dir, exist_ok=True)

    # Adjusted learning rates
    lr_parent_child = 0.0001
    lr_discriminator = 0.00005  # Lower learning rate for discriminator
    optimizer_parent = optim.Adam(list(parent_encoder.parameters()) + list(parent_generator.parameters()), lr=lr_parent_child)
    optimizer_child = optim.Adam(list(child_encoder.parameters()) + list(child_generator.parameters()), lr=lr_parent_child)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr_discriminator)
    optimizer_genetic_discriminator = optim.Adam(genetic_discriminator.parameters(), lr=lr_parent_child)
    optimizer_mapping = optim.Adam(mapping_function.parameters(), lr=lr_parent_child)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for iteration, (images, genetic_factors, external_factors, variety_factors, image_ids) in enumerate(train_loader):
            images = images.to(device)
            genetic_factors = genetic_factors.to(device)
            external_factors = external_factors.to(device)
            variety_factors = variety_factors.to(device)
            for i in range(images.size(0)):
                image = images[i:i+1]  # Get a single image for processing
                image_id = image_ids[i]  # Get the corresponding image ID

                if image_id in ["01", "02"]:  # Parent domain
                    optimizer_parent.zero_grad()
                    encoded_parent = parent_encoder(image)
                    generated_parent = parent_generator(encoded_parent, external_factors[i:i+1], genetic_factors[i:i+1])

                    loss_parent = l1_loss(generated_parent, image)  # L1 reconstruction loss
                    loss_parent.backward()
                    optimizer_parent.step()
                else:  # Children domain
                    optimizer_child.zero_grad()
                    encoded_child = child_encoder(image)
                    generated_child = child_generator(encoded_child, external_factors[i:i+1], genetic_factors[i:i+1], variety_factors[i:i+1])

                    loss_child = l1_loss(generated_child, image)
                    loss_child.backward()
                    optimizer_child.step()

                    # Save generated child images every 10 iterations with unique filenames
                    if iteration % 100 == 0:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        save_path = os.path.join(generated_images_dir, f'generated_child_{image_id}_{timestamp}.png')
                        utils.save_image(generated_child.detach(), save_path)

                    gf_x = genetic_factors[i:i+1]  # Genetic factors of the parent
                    gm_x = genetic_factors[i:i+1]  # Assuming same for demonstration

                    predicted_genetic_factors = mapping_function(gf_x, gm_x)

                    # Compute mapping loss
                    mapping_loss = l1_loss(predicted_genetic_factors, genetic_factors[i:i+1])
                    mapping_loss.backward()
                    optimizer_mapping.step()

            # === Discriminator ===
            optimizer_discriminator.zero_grad()
            real_adv, real_class = discriminator(images)
            fake_adv, fake_class = discriminator(generated_child.detach())

            # Compute adversarial loss
            adv_loss = nn.BCEWithLogitsLoss()(real_adv, torch.ones_like(real_adv)) + \
                       nn.BCEWithLogitsLoss()(fake_adv, torch.zeros_like(fake_adv))

            # Compute classification loss with label smoothing
            smoothed_labels = external_factors * 0.9 + 0.1 * (1 - external_factors)  # Label smoothing
            class_loss = nn.BCEWithLogitsLoss()(real_class, smoothed_labels)

            # Compute gradient penalty
            gp = gradient_penalty(discriminator, images, generated_child.detach())

            total_loss = adv_loss + class_loss + 10 * gp  # Scale the gradient penalty
            total_loss.backward()
            optimizer_discriminator.step()

            # === Genetic Discriminator ===
            optimizer_genetic_discriminator.zero_grad()
            loss_genetic_discriminator = adversarial_loss(genetic_discriminator, images, generated_parent.detach())
            loss_genetic_discriminator.backward()
            optimizer_genetic_discriminator.step()

            # Print epoch, iteration, and loss values
            print(f"Epoch [{epoch + 1}/{num_epochs}], Iteration [{iteration + 1}/{len(train_loader)}], "
                  f"Loss Parent: {loss_parent.item() if 'loss_parent' in locals() else 'N/A'}, "
                  f"Loss Child: {loss_child.item() if 'loss_child' in locals() else 'N/A'}, "
                  f"Loss Mapping: {mapping_loss.item() if 'mapping_loss' in locals() else 'N/A'}, "
                  f"Loss Discriminator: {total_loss.item() if 'total_loss' in locals() else 'N/A'}, "
                  f"Loss Genetic Discriminator: {loss_genetic_discriminator.item() if 'loss_genetic_discriminator' in locals() else 'N/A'}")

        # Save model weights after each epoch
        os.makedirs(save_dir, exist_ok=True)
        torch.save(parent_encoder.state_dict(), os.path.join(save_dir, f'parent_encoder_epoch_{epoch + 1}.pth'))
        torch.save(parent_generator.state_dict(), os.path.join(save_dir, f'parent_generator_epoch_{epoch + 1}.pth'))
        torch.save(child_encoder.state_dict(), os.path.join(save_dir, f'child_encoder_epoch_{epoch + 1}.pth'))
        torch.save(child_generator.state_dict(), os.path.join(save_dir, f'child_generator_epoch_{epoch + 1}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(save_dir, f'discriminator_epoch_{epoch + 1}.pth'))
        torch.save(genetic_discriminator.state_dict(), os.path.join(save_dir, f'genetic_discriminator_epoch_{epoch + 1}.pth'))
        torch.save(mapping_function.state_dict(), os.path.join(save_dir, f'mapping_function_epoch_{epoch + 1}.pth'))

    print("Training complete. Models saved.")

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Mount Google Drive (if using Colab)
    from google.colab import drive
    drive.mount('/content/drive')

    # Set the root directory for the dataset
    root_dir = '/content/drive/MyDrive/sample_train2'  # Update this path accordingly

    # Define transformations (if needed)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Create the dataset and data loader
    dataset = FamilyFaceDataset(root_dir=root_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize models
    parent_encoder = ParentEncoder().to(device)
    parent_generator = ParentGenerator().to(device)
    child_encoder = ChildEncoder().to(device)
    child_generator = ChildGenerator().to(device)
    discriminator = Discriminator().to(device)
    genetic_discriminator = GeneticDiscriminator().to(device)
    mapping_function = MappingFunctionT().to(device)

    # Start training
    num_epochs = 50
    save_dir = '/content/drive/My Drive/model_weights'
    generated_images_dir = '/content/drive/My Drive/generated2_images'
    train_model(train_loader, parent_encoder, parent_generator, child_encoder, child_generator,
                discriminator, genetic_discriminator, mapping_function, num_epochs, save_dir,
                generated_images_dir)

if __name__ == "__main__":
    main()
-----------------------
Testing code

import os
import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import sqrtm
import lpips  # Import the LPIPS library
from google.colab import drive

# === Custom Dataset Class for Testing ===
class FamilyFaceTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.image_ids = []

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for image_file in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_file)
                    self.image_paths.append(image_path)
                    self.image_ids.append(folder)  # Store folder name as image ID

    def _parse_filename(self, filename):
        attributes = filename[:8]
        return {
            "image_id": attributes[0:2],
            "gender": int(attributes[2]),
            "skin_color": int(attributes[3]),
            "age": int(attributes[4]),
            "emotion": int(attributes[5]),
            "glasses": int(attributes[6]),
            "moustache": int(attributes[7])
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        if self.transform:
            image = self.transform(image)

        image_id = self._parse_filename(os.path.basename(img_path))["image_id"]
        return image, image_id

# Function to retrieve the directory structure of the input test data
def retrieve_directory_structure(input_root_dir):
    structure = {}

    for root, dirs, files in os.walk(input_root_dir):
        relative_path = os.path.relpath(root, input_root_dir)
        structure[relative_path] = {
            'directories': dirs,
            'files': files
        }

    return structure

def load_images_from_folder(folder):
    """Load images from a specific folder."""
    images = []
    image_ids = []
    for filename in os.listdir(folder):
        if filename.endswith('.png'):  # Adjust this according to your image format
            img_path = os.path.join(folder, filename)
            try:
                images.append(Image.open(img_path).convert('RGB'))  # Load and convert to RGB
                image_ids.append(filename)  # Store the filename
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return images, image_ids

def extract_features(model, image_tensor):
    """Extract features from the model's second-to-last layer."""
    with torch.no_grad():
        features = model(image_tensor)
    return features

def compute_cosine_similarity(features_a, features_b):
    """Compute cosine similarity between two sets of features."""
    return cosine_similarity(features_a, features_b)

def calculate_fid(real_images, generated_images, device):
    """Calculate the Fréchet Inception Distance (FID) between two sets of images."""
    # Load the Inception model
    inception_model = models.inception_v3(weights='DEFAULT', transform_input=False).to(device)
    inception_model.eval()

    # Preprocess images and extract features
    def get_features(images):
        with torch.no_grad():
            # Transform images to tensors and resize
            tensor_images = [transforms.Resize((299, 299))(transforms.ToTensor()(img)).unsqueeze(0).to(device) for img in images]
            features = [inception_model(img).detach().cpu().numpy() for img in tensor_images]
            return np.concatenate(features, axis=0)

    # Extract features for real and generated images
    real_features = get_features(real_images)
    generated_features = get_features(generated_images)

    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)

    # Calculate the FID
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_value = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid_value

def calculate_lpips(real_images, generated_images, device):
    """Calculate LPIPS score between two sets of images."""
    lpips_model = lpips.LPIPS(net='alex').to(device)  # Use the AlexNet model
    lpips_scores = []

    for real, generated in zip(real_images, generated_images):
        real_tensor = transforms.ToTensor()(real).unsqueeze(0).to(device)
        generated_tensor = transforms.ToTensor()(generated).unsqueeze(0).to(device)

        # Calculate LPIPS score
        score = lpips_model(real_tensor, generated_tensor)
        lpips_scores.append(score.item())

    avg_lpips = np.mean(lpips_scores)
    return avg_lpips

def test_model(parent_encoder, parent_generator, child_encoder, child_generator, device, input_root_dir, output_root_dir):
    # Set models to evaluation mode
    parent_encoder.eval()
    parent_generator.eval()
    child_encoder.eval()
    child_generator.eval()

    cosine_similarities = []
    child_encodings = []  # List to store only child encodings
    generated_images = []  # Store generated images for FID calculation
    real_images = []  # Load real child images for FID calculation

    # Iterate through each family folder
    for family_folder in os.listdir(input_root_dir):
        family_folder_path = os.path.join(input_root_dir, family_folder)

        if os.path.isdir(family_folder_path):  # Check if it's a directory
            images, image_ids = load_images_from_folder(family_folder_path)
            print(f"Loaded {len(images)} images from {family_folder_path}")

            # Process each image in the family folder
            for idx, image in enumerate(images):
                if image is None:
                    continue  # Skip if there was an error loading the image

                image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension

                # Check if the image is a child based on the filename
                if not image_ids[idx].startswith(('01', '02')):  # If not a parent ID
                    # Encode and generate child image
                    encoded_child = child_encoder(image_tensor)
                    generated_child = child_generator(encoded_child, torch.zeros(1, 4).to(device), torch.zeros(1, 2).to(device), torch.zeros(1, 1).to(device))
                    generated_image = transforms.ToPILImage()(generated_child.squeeze(0).cpu())

                    # Save the generated child image
                    output_family_folder = os.path.join(output_root_dir, family_folder)
                    os.makedirs(output_family_folder, exist_ok=True)
                    generated_image.save(os.path.join(output_family_folder, f'{image_ids[idx].split(".")[0]}_child_generated.png'))
                    print(f"Generated child image for ID: {image_ids[idx]} in {output_family_folder}")

                    # Store the encoding of the child image
                    encoded_child_np = encoded_child.detach().cpu().numpy()
                    child_encodings.append(encoded_child_np.flatten())  # Flatten the tensor

                    # Store generated images for FID calculation
                    generated_images.append(generated_image)

                else:
                    # Store real child images for FID calculation (if available)
                    real_images.append(image)

    # Compute cosine similarities for child images
    if len(child_encodings) >= 2:
        child_encodings = np.array(child_encodings)  # Convert to NumPy array
        cosine_similarities = compute_cosine_similarity(child_encodings[:-1], child_encodings[1:])
        avg_cosine_similarity = np.mean(cosine_similarities)
        print(f"Average Cosine Similarity for child images: {avg_cosine_similarity}")
    else:
        print("Not enough child images for cosine similarity calculation.")

    # Calculate FID if sufficient images are available
    if len(real_images) > 0 and len(generated_images) > 0:
        fid_value = calculate_fid(real_images, generated_images, device)
        print(f"Fréchet Inception Distance (FID): {fid_value}")

        # Calculate LPIPS if sufficient images are available
        avg_lpips = calculate_lpips(real_images, generated_images, device)
        print(f"Average LPIPS Score: {avg_lpips}")
    else:
        print("Not enough images for FID calculation.")

# Main function to execute the process
def main():
    # Mount Google Drive
    drive.mount('/content/drive')

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set the root directory for the test dataset
    input_root_dir = '/content/drive/MyDrive/testdata'  # Update this path accordingly
    output_root_dir = '/content/drive/MyDrive/generated_images1402'

    # Retrieve the directory structure of the input test data
    directory_structure = retrieve_directory_structure(input_root_dir)
    print(f"Input directory structure: {directory_structure}")

    # Define transformations (if needed)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Create the test dataset and data loader
    test_dataset = FamilyFaceTestDataset(root_dir=input_root_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize models
    parent_encoder = ParentEncoder().to(device)
    parent_generator = ParentGenerator().to(device)
    child_encoder = ChildEncoder().to(device)
    child_generator = ChildGenerator().to(device)

    # Load the trained model weights
    try:
        parent_encoder.load_state_dict(torch.load('/content/drive/MyDrive/model_weights/parent_encoder_epoch_40.pth', weights_only=True))
        parent_generator.load_state_dict(torch.load('/content/drive/MyDrive/model_weights/parent_generator_epoch_40.pth', weights_only=True))
        child_encoder.load_state_dict(torch.load('/content/drive/MyDrive/model_weights/child_encoder_epoch_40.pth', weights_only=True))
        child_generator.load_state_dict(torch.load('/content/drive/MyDrive/model_weights/child_generator_epoch_40.pth', weights_only=True))
    except FileNotFoundError as e:
        print(f"Model weights not found: {e}")
        return

    # Run the testing function
    test_model(parent_encoder, parent_generator, child_encoder, child_generator, device, input_root_dir, output_root_dir)

# Execute the main function
if __name__ == "__main__":
    main()
