import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict
import time


class ImageSearchEngine:
    """Moteur de recherche d'images basé sur la similarité visuelle"""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        self.features_db = None
        self.image_paths = None
        self.labels = None
        
    def load_dataset(self, split: str = "train") -> pd.DataFrame:
        """Charge le dataset avec les features pré-extraites"""
        csv_path = self.data_path / f"{split}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset {csv_path} not found")
        
        # Charge sans headers car première ligne = données
        df = pd.read_csv(csv_path, header=None)
        print(f"Dataset {split}: {len(df)} images, {df.shape[1]-2} features")
        
        return df
    
    def build_index(self, split: str = "train"):
        """Construit l'index de recherche"""
        df = self.load_dataset(split)
        
        # Structure: [chemin_image, label, feature1, feature2, ...]
        self.image_paths = df.iloc[:, 0].values
        self.labels = df.iloc[:, 1].values
        feature_data = df.iloc[:, 2:].values
        
        # Conversion en float32 pour optimiser
        self.features_db = feature_data.astype(np.float32)
        
        categories = np.unique(self.labels)
        print(f"Index construit: {len(self.image_paths)} images")
        print(f"Catégories: {categories}")
        print(f"Features: {self.features_db.shape[1]} dimensions")
        
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Calcule la similarité cosinus entre vecteurs"""
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        
        a_normalized = a / (a_norm + 1e-8)
        b_normalized = b / (b_norm + 1e-8)
        
        return np.dot(a_normalized, b_normalized.T)
    
    def search_similar_images(self, query_image_path: str, k: int = 5) -> List[Tuple[str, float, str]]:
        """Recherche d'images similaires par similarité cosinus"""
        if self.features_db is None:
            raise ValueError("Index non construit. Appelez build_index() d'abord.")
        
        # Trouve l'index de l'image query
        query_idx = None
        for i, path in enumerate(self.image_paths):
            if path == query_image_path:
                query_idx = i
                break
        
        if query_idx is None:
            raise ValueError(f"Image {query_image_path} non trouvée dans la base")
        
        query_features = self.features_db[query_idx:query_idx+1]
        
        # Calcule similarités avec toute la base
        similarities = self.cosine_similarity(query_features, self.features_db)[0]
        
        # Récupère le top-k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append((
                self.image_paths[idx],
                float(similarities[idx]),
                self.labels[idx]
            ))
        
        return results
    
    def evaluate_search_quality(self, num_queries: int = 50) -> Dict[str, float]:
        """Évalue la qualité de recherche sur un échantillon"""
        if len(self.image_paths) < num_queries:
            num_queries = len(self.image_paths)
        
        query_indices = np.random.choice(len(self.image_paths), num_queries, replace=False)
        
        correct_top1 = 0
        same_category_total = 0
        
        for query_idx in query_indices:
            query_path = self.image_paths[query_idx]
            true_label = self.labels[query_idx]
            
            # Recherche des similaires
            results = self.search_similar_images(query_path, k=10)
            
            # Compte catégories identiques (excluant la query elle-même)
            same_category_count = 0
            for path, score, label in results[1:]:  # Skip query (index 0)
                if label == true_label:
                    same_category_count += 1
            
            same_category_total += same_category_count
            
            # Vérifie si top-1 (après query) est bonne catégorie
            if len(results) > 1 and results[1][2] == true_label:
                correct_top1 += 1
        
        return {
            'category_accuracy_top1': correct_top1 / num_queries,
            'avg_same_category_in_top9': same_category_total / (num_queries * 9),
            'total_queries': num_queries
        }
    
    def benchmark_speed(self, num_queries: int = 20) -> Dict[str, float]:
        """Benchmark de vitesse de recherche"""
        if len(self.image_paths) < num_queries:
            num_queries = len(self.image_paths)
        
        query_indices = np.random.choice(len(self.image_paths), num_queries, replace=False)
        
        start_time = time.time()
        
        for query_idx in query_indices:
            query_path = self.image_paths[query_idx]
            self.search_similar_images(query_path, k=5)
        
        total_time = time.time() - start_time
        
        return {
            'total_time': total_time,
            'avg_time_per_query': total_time / num_queries,
            'queries_per_second': num_queries / total_time
        }


class FeatureExtractor:
    """Extracteur de features pour images compressées"""
    
    def extract_histogram_features(self, image: Image.Image, bins: int = 64) -> np.ndarray:
        """Extrait histogramme de luminance"""
        if image.mode != 'L':
            image = image.convert('L')
        
        hist, _ = np.histogram(np.array(image).flatten(), bins=bins, range=(0, 255))
        hist = hist.astype(np.float32)
        return hist / (np.sum(hist) + 1e-8)  # Normalisation
    
    def extract_statistical_features(self, image: Image.Image) -> np.ndarray:
        """Extrait features statistiques"""
        if image.mode != 'L':
            image = image.convert('L')
        array = np.array(image, dtype=np.float32)
        
        features = [
            np.mean(array),
            np.std(array),
            np.min(array),
            np.max(array),
            np.median(array),
            np.percentile(array, 25),
            np.percentile(array, 75),
        ]
        
        return np.array(features, dtype=np.float32)
    
    def extract_features(self, image: Image.Image) -> np.ndarray:
        """Extrait features combinées (histogramme + stats)"""
        hist = self.extract_histogram_features(image, bins=32)
        stats = self.extract_statistical_features(image)
        return np.concatenate([hist, stats])


class CompressedImageSearchEngine:
    """Moteur de recherche pour images compressées"""
    
    def __init__(self, features_db: np.ndarray, image_paths: List[str], labels: List[str]):
        self.features_db = features_db
        self.image_paths = image_paths
        self.labels = labels
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Similarité cosinus optimisée"""
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        
        a_normalized = a / (a_norm + 1e-8)
        b_normalized = b / (b_norm + 1e-8)
        
        return np.dot(a_normalized, b_normalized.T)
    
    def search_similar_images(self, query_idx: int, k: int = 10) -> List[Tuple[str, float, str]]:
        """Recherche par index de query"""
        query_features = self.features_db[query_idx:query_idx+1]
        similarities = self.cosine_similarity(query_features, self.features_db)[0]
        top_indices = np.argsort(similarities)[::-1][:k]
        
        return [(self.image_paths[idx], similarities[idx], self.labels[idx]) for idx in top_indices]
    
    def evaluate_quality(self, num_queries: int = 20) -> Dict[str, float]:
        """Évalue qualité de recherche sur images compressées"""
        if len(self.image_paths) < num_queries:
            num_queries = len(self.image_paths)
        
        query_indices = np.random.choice(len(self.image_paths), num_queries, replace=False)
        
        correct_top1 = 0
        same_category_total = 0
        
        for query_idx in query_indices:
            true_label = self.labels[query_idx]
            results = self.search_similar_images(query_idx, k=10)
            
            # Compte même catégorie (sans la query)
            same_category_count = sum(1 for _, _, label in results[1:] if label == true_label)
            same_category_total += same_category_count
            
            # Top-1 accuracy (après query)
            if len(results) > 1 and results[1][2] == true_label:
                correct_top1 += 1
        
        return {
            'category_accuracy_top1': correct_top1 / num_queries,
            'avg_same_category_in_top9': same_category_total / (num_queries * 9),
            'total_queries': num_queries
        } 
