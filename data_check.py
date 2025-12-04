#data_check.py
import os
import cv2
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

def get_image_paths(dataset_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_paths = []

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for root, _, files in os.walk(class_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in image_extensions:
                        image_paths.append((os.path.join(root, file), class_name))
    return image_paths

def check_image_readability(image_paths):
    unreadable = []
    for path, _ in image_paths:
        try:
            img = Image.open(path)
            img.verify()
        except Exception:
            unreadable.append(path)
    return unreadable

def analyze_image_dimensions(image_paths):
    sizes = Counter()
    for path, _ in image_paths:
        try:
            img = Image.open(path)
            sizes[img.size] += 1
        except:
            continue
    return sizes

def count_images_per_class(image_paths):
    class_counter = Counter([label for _, label in image_paths])
    return class_counter

def plot_class_distribution(class_counts):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.barh(classes, counts, color='skyblue')
    plt.title("Image Count per Class")
    plt.xlabel("Number of Images")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.show()

def plot_sample_images(image_paths, class_counts, samples_per_class=2):
    plt.figure(figsize=(12, 8))
    plotted = 0
    shown = set()
    
    for path, label in image_paths:
        if class_counts[label] >= samples_per_class and shown.count(label) < samples_per_class:
            try:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(len(class_counts), samples_per_class, plotted + 1)
                plt.imshow(img)
                plt.title(label)
                plt.axis('off')
                plotted += 1
                shown.add(label)
                if plotted >= len(class_counts) * samples_per_class:
                    break
            except:
                continue
    plt.tight_layout()
    plt.show()

# Dataset path
dataset_path = r"D:\projects\thermal_induction_motor\prescriptive maintenance\IR-Motor-bmp"

# Step 1: Get all image paths
image_paths = get_image_paths(dataset_path)

# Step 2: Count per class
class_counts = count_images_per_class(image_paths)
print("\n Class Distribution:")
for cls, count in class_counts.items():
    print(f"Class '{cls}': {count} images")

# Step 3: Image size check
size_counts = analyze_image_dimensions(image_paths)
print("\nImage Size Distribution:")
for size, count in size_counts.items():
    print(f"{size}: {count} images")

# Step 4: File corruption check
unreadable = check_image_readability(image_paths)
print(f"\n⚠️ Corrupt/Unreadable Images: {len(unreadable)}")
if unreadable:
    print("List of corrupt images:")
    for img in unreadable[:5]:
        print(img)

# Step 5: Plot distribution
plot_class_distribution(class_counts)

# Step 6: Sample images per class
# plot_sample_images(image_paths, class_counts)  # Uncomment if you want visual output
