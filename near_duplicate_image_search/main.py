"""
AI Image Search Algorithm Demo
Uses CLIP model for image embeddings and cosine similarity for search
"""

import os
import pickle
import warnings
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# For this demo, we'll simulate CLIP embeddings since installing transformers might be heavy
# In production, you'd use: from transformers import CLIPProcessor, CLIPModel


class ImageSearchEngine:
    def __init__(self):
        self.image_paths = []
        self.image_embeddings = []
        self.image_database = {}

    def create_demo_dataset(self):
        """
        Creates a lightweight demo dataset with sample images
        For demo purposes, we'll create synthetic embeddings that represent different image types
        """
        # Demo images with their simulated features (normally extracted by CNN/CLIP)
        demo_images = {
            "cat_1.jpg": {
                "url": "https://placekitten.com/300/200",
                "features": np.array(
                    [0.8, 0.2, 0.1, 0.9, 0.3, 0.1, 0.7, 0.2]
                ),  # Cat-like features
                "tags": ["animal", "cat", "pet", "cute"],
            },
            "dog_1.jpg": {
                "url": "https://placedog.net/300/200",
                "features": np.array(
                    [0.7, 0.3, 0.2, 0.8, 0.4, 0.1, 0.6, 0.3]
                ),  # Dog-like features
                "tags": ["animal", "dog", "pet", "loyal"],
            },
            "landscape_1.jpg": {
                "url": "https://picsum.photos/300/200?nature",
                "features": np.array(
                    [0.1, 0.9, 0.8, 0.1, 0.2, 0.9, 0.1, 0.8]
                ),  # Nature features
                "tags": ["nature", "landscape", "outdoor", "scenic"],
            },
            "city_1.jpg": {
                "url": "https://picsum.photos/300/200?city",
                "features": np.array(
                    [0.2, 0.1, 0.9, 0.2, 0.8, 0.2, 0.9, 0.1]
                ),  # Urban features
                "tags": ["city", "urban", "buildings", "architecture"],
            },
            "flower_1.jpg": {
                "url": "https://picsum.photos/300/200?flower",
                "features": np.array(
                    [0.3, 0.8, 0.6, 0.3, 0.2, 0.7, 0.2, 0.9]
                ),  # Flower features
                "tags": ["flower", "nature", "colorful", "garden"],
            },
            "car_1.jpg": {
                "url": "https://picsum.photos/300/200?car",
                "features": np.array(
                    [0.9, 0.1, 0.3, 0.2, 0.9, 0.1, 0.8, 0.2]
                ),  # Vehicle features
                "tags": ["car", "vehicle", "transport", "modern"],
            },
        }

        self.image_database = demo_images
        self.image_paths = list(demo_images.keys())
        self.image_embeddings = np.array(
            [img_data["features"] for img_data in demo_images.values()]
        )

        print(f"✓ Created demo dataset with {len(self.image_paths)} images")
        return demo_images

    def extract_features_real(self, image_path: str) -> np.ndarray:
        """
        Real feature extraction using a pre-trained model
        This is a simplified version - in production you'd use CLIP or ResNet
        """
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            img = img.resize((224, 224))  # Standard size for most models

            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0

            # Simulate feature extraction (in real implementation, pass through CNN)
            # This creates a simple feature vector based on color channels and basic stats
            features = []
            for channel in range(3):  # RGB channels
                channel_data = img_array[:, :, channel]
                features.extend(
                    [
                        np.mean(channel_data),  # Average intensity
                        np.std(channel_data),  # Color variation
                        np.max(channel_data),  # Brightest point
                        np.min(channel_data),  # Darkest point
                    ]
                )

            # Add some texture features (simplified)
            gray = np.mean(img_array, axis=2)
            features.extend(
                [
                    np.mean(np.gradient(gray)[0]),  # Horizontal edges
                    np.mean(np.gradient(gray)[1]),  # Vertical edges
                ]
            )

            return np.array(features)

        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return np.random.rand(14)  # Return random features as fallback

    def build_index(self, image_folder: str = None):
        """
        Build search index from images in a folder
        For demo, we'll use the synthetic dataset
        """
        if image_folder is None:
            # Use demo dataset
            self.create_demo_dataset()
        else:
            # Real implementation for actual image folder
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            self.image_paths = []
            features_list = []

            for filename in os.listdir(image_folder):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    img_path = os.path.join(image_folder, filename)
                    features = self.extract_features_real(img_path)
                    self.image_paths.append(img_path)
                    features_list.append(features)

            self.image_embeddings = np.array(features_list)
            print(f"✓ Built index with {len(self.image_paths)} images")

    def text_to_features(self, query_text: str) -> np.ndarray:
        """
        Convert text query to feature vector
        In production, this would use CLIP's text encoder
        """
        # Simple keyword-based feature mapping for demo
        keyword_features = {
            "cat": np.array([0.9, 0.1, 0.1, 0.9, 0.2, 0.1, 0.8, 0.1]),
            "dog": np.array([0.8, 0.2, 0.2, 0.8, 0.3, 0.1, 0.7, 0.2]),
            "animal": np.array([0.8, 0.2, 0.1, 0.9, 0.3, 0.1, 0.7, 0.2]),
            "pet": np.array([0.8, 0.2, 0.1, 0.9, 0.3, 0.1, 0.7, 0.2]),
            "nature": np.array([0.1, 0.9, 0.8, 0.1, 0.2, 0.9, 0.1, 0.8]),
            "landscape": np.array([0.1, 0.9, 0.8, 0.1, 0.2, 0.9, 0.1, 0.8]),
            "city": np.array([0.2, 0.1, 0.9, 0.2, 0.8, 0.2, 0.9, 0.1]),
            "urban": np.array([0.2, 0.1, 0.9, 0.2, 0.8, 0.2, 0.9, 0.1]),
            "flower": np.array([0.3, 0.8, 0.6, 0.3, 0.2, 0.7, 0.2, 0.9]),
            "car": np.array([0.9, 0.1, 0.3, 0.2, 0.9, 0.1, 0.8, 0.2]),
            "vehicle": np.array([0.9, 0.1, 0.3, 0.2, 0.9, 0.1, 0.8, 0.2]),
        }

        # Find matching keywords and average their features
        query_words = query_text.lower().split()
        matching_features = []

        for word in query_words:
            if word in keyword_features:
                matching_features.append(keyword_features[word])

        if matching_features:
            return np.mean(matching_features, axis=0)
        else:
            # Return neutral features for unknown queries
            return np.ones(8) * 0.5

    def search_by_text(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Search images using text query
        """
        if len(self.image_embeddings) == 0:
            print("No images in index. Please build index first.")
            return []

        # Convert text to features
        query_features = self.text_to_features(query).reshape(1, -1)

        # Calculate similarities
        similarities = cosine_similarity(query_features, self.image_embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            image_name = self.image_paths[idx]
            similarity_score = similarities[idx]
            results.append((image_name, similarity_score))

        return results

    def search_by_image(
        self, query_image_path: str, top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Search similar images using an image as query
        """
        if len(self.image_embeddings) == 0:
            print("No images in index. Please build index first.")
            return []

        # Extract features from query image
        query_features = self.extract_features_real(query_image_path).reshape(1, -1)

        # Calculate similarities (exclude the query image itself if it's in the database)
        similarities = cosine_similarity(query_features, self.image_embeddings)[0]

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            image_name = self.image_paths[idx]
            similarity_score = similarities[idx]
            # Skip if it's the same image
            if os.path.basename(query_image_path) != os.path.basename(image_name):
                results.append((image_name, similarity_score))

        return results[:top_k]

    def display_results(self, results: List[Tuple[str, float]], query: str):
        """
        Display search results with scores
        """
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 50)

        if not results:
            print("No results found.")
            return

        for i, (image_name, score) in enumerate(results, 1):
            print(f"{i}. {image_name}")
            print(f"   Similarity Score: {score:.3f}")

            # Display tags if available (for demo dataset)
            if image_name in self.image_database:
                tags = self.image_database[image_name]["tags"]
                print(f"   Tags: {', '.join(tags)}")

            print()

    def save_index(self, filepath: str):
        """
        Save the search index to disk
        """
        index_data = {
            "image_paths": self.image_paths,
            "image_embeddings": self.image_embeddings,
            "image_database": self.image_database,
        }

        with open(filepath, "wb") as f:
            pickle.dump(index_data, f)

        print(f"✓ Index saved to {filepath}")

    def load_index(self, filepath: str):
        """
        Load search index from disk
        """
        with open(filepath, "rb") as f:
            index_data = pickle.load(f)

        self.image_paths = index_data["image_paths"]
        self.image_embeddings = index_data["image_embeddings"]
        self.image_database = index_data.get("image_database", {})

        print(f"✓ Index loaded from {filepath}")


def demo_search_engine():
    """
    Demonstration of the image search engine
    """
    print("🚀 AI Image Search Engine Demo")
    print("=" * 40)

    # Initialize search engine
    engine = ImageSearchEngine()

    # Build index with demo dataset
    print("📚 Building search index...")
    engine.build_index()

    # Demo text-based searches
    test_queries = [
        "cat",
        "nature landscape",
        "city urban",
        "flower garden",
        "car vehicle",
    ]

    print("\n🔍 Testing Text-based Search:")
    print("=" * 40)

    for query in test_queries:
        results = engine.search_by_text(query, top_k=3)
        engine.display_results(results, query)

    # Show statistics
    print("📊 Search Engine Statistics:")
    print("=" * 30)
    print(f"Total images indexed: {len(engine.image_paths)}")
    print(f"Feature vector dimension: {engine.image_embeddings.shape[1]}")
    print(f"Index size: {engine.image_embeddings.nbytes / 1024:.2f} KB")

    # Save index demo
    print("\n💾 Saving search index...")
    engine.save_index("image_search_index.pkl")

    return engine


if __name__ == "__main__":
    # Run the demo
    search_engine = demo_search_engine()

    print("\n" + "=" * 50)
    print("🎯 Usage Examples:")
    print("=" * 50)
    print(
        """
# Initialize and build index
engine = ImageSearchEngine()
engine.build_index()  # Uses demo dataset

# Search by text
results = engine.search_by_text("cat", top_k=3)
engine.display_results(results, "cat")

# Search by image (if you have real images)
# results = engine.search_by_image("path/to/query/image.jpg", top_k=3)

# Save/load index
engine.save_index("my_index.pkl")
engine.load_index("my_index.pkl")

# For real images, use:
# engine.build_index("path/to/your/image/folder")
    """
    )

    print("\n🔧 To use with real images:")
    print(
        "1. Install required packages: pip install pillow numpy scikit-learn matplotlib"
    )
    print("2. For production: pip install transformers torch (to use CLIP)")
    print("3. Replace demo dataset with engine.build_index('your_image_folder')")
