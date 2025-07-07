import os
import io
import math
import time
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image, ImageOps
import pickle


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim_simple(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculate simplified SSIM"""
    mu1, mu2 = np.mean(original), np.mean(reconstructed)
    sigma1, sigma2 = np.var(original), np.var(reconstructed)
    sigma12 = np.mean((original - mu1) * (reconstructed - mu2))
    
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
    return float(np.clip(ssim, 0, 1))


class CompressionMethod:
    def __init__(self, name: str):
        self.name = name
    
    def compress(self, image: Image.Image) -> bytes:
        raise NotImplementedError
    
    def decompress(self, data: bytes) -> Image.Image:
        raise NotImplementedError




class PNGCompression(CompressionMethod):
    """PNG - Compression sans perte"""
    def __init__(self):
        super().__init__("PNG_Lossless")
    
    def compress(self, image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', optimize=True)
        return buffer.getvalue()
    
    def decompress(self, data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data))


class JPEGCompression(CompressionMethod):
    """JPEG - Compression avec perte"""
    def __init__(self, quality: int = 85):
        super().__init__(f"JPEG_Q{quality}")
        self.quality = quality
    
    def compress(self, image: Image.Image) -> bytes:
        if image.mode not in ['RGB', 'L']:
            image = image.convert('RGB')
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=self.quality, optimize=True)
        return buffer.getvalue()
    
    def decompress(self, data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data))




class HaarCompression(CompressionMethod):
    """Haar Wavelets"""
    def __init__(self, tolerance: float = 5.0):
        super().__init__(f"Haar_T{tolerance}")
        self.tolerance = tolerance
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        image = ImageOps.grayscale(image)
        dim = max(image.size)
        new_dim = 2 ** int(math.ceil(math.log(dim, 2)))
        return ImageOps.pad(image, (new_dim, new_dim))
    
    def get_haar_step(self, i: int, k: int) -> np.ndarray:
        transform = np.zeros((2 ** k, 2 ** k))
        for j in range(2 ** (k - i - 1)):
            transform[2 * j, j] = 1 / 2
            transform[2 * j + 1, j] = 1 / 2
        offset = 2 ** (k - i - 1)
        for j in range(2 ** (k - i - 1)):
            transform[2 * j, offset + j] = 1 / 2
            transform[2 * j + 1, offset + j] = -1 / 2
        for j in range(2 ** (k - i), 2 ** k):
            transform[j, j] = 1
        return transform
    
    def get_haar_transform(self, k: int) -> np.ndarray:
        transform = np.eye(2 ** k)
        for i in range(k):
            transform = transform @ self.get_haar_step(i, k)
        return transform
    
    def haar_encode(self, a: np.ndarray) -> np.ndarray:
        k = int(math.log2(len(a)))
        row_encoder = self.get_haar_transform(k)
        return row_encoder.T @ a @ row_encoder
    
    def haar_decode(self, a: np.ndarray) -> np.ndarray:
        k = int(math.log2(len(a)))
        row_decoder = np.linalg.inv(self.get_haar_transform(k))
        return row_decoder.T @ a @ row_decoder
    
    def truncate_values(self, a: np.ndarray, tolerance: float) -> np.ndarray:
        return np.where(np.abs(a) < tolerance, 0, a)
    
    def compress(self, image: Image.Image) -> bytes:
        processed_img = self.preprocess_image(image)
        array = np.array(processed_img, dtype=np.float64)
        encoded = self.haar_encode(array)
        truncated = self.truncate_values(encoded, self.tolerance)
        
        data = {'array': truncated, 'original_size': image.size}
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def decompress(self, data: bytes) -> Image.Image:
        data_dict = pickle.loads(data)
        array = data_dict['array']
        
        decoded = self.haar_decode(array)
        decoded = np.clip(decoded, 0, 255).astype(np.uint8)
        img = Image.fromarray(decoded, mode='L')
        
        if 'original_size' in data_dict:
            img = img.resize(data_dict['original_size'], Image.LANCZOS)
        return img


class DCTCompression(CompressionMethod):
    """DCT - Transformée cosinus discrète"""
    def __init__(self, quality_factor: float = 50.0):
        super().__init__(f"DCT_Q{quality_factor}")
        self.quality_factor = quality_factor
    
    def dct2d(self, block: np.ndarray) -> np.ndarray:
        """DCT 2D simple sur un bloc 8x8"""
        N = block.shape[0]
        dct_block = np.zeros_like(block, dtype=np.float64)
        
        for u in range(N):
            for v in range(N):
                cu = 1/math.sqrt(2) if u == 0 else 1
                cv = 1/math.sqrt(2) if v == 0 else 1
                
                sum_val = 0
                for x in range(N):
                    for y in range(N):
                        sum_val += block[x, y] * math.cos((2*x + 1) * u * math.pi / (2*N)) * math.cos((2*y + 1) * v * math.pi / (2*N))
                
                dct_block[u, v] = (2/N) * cu * cv * sum_val
        
        return dct_block
    
    def idct2d(self, dct_block: np.ndarray) -> np.ndarray:
        """IDCT 2D inverse"""
        N = dct_block.shape[0]
        block = np.zeros_like(dct_block, dtype=np.float64)
        
        for x in range(N):
            for y in range(N):
                sum_val = 0
                for u in range(N):
                    for v in range(N):
                        cu = 1/math.sqrt(2) if u == 0 else 1
                        cv = 1/math.sqrt(2) if v == 0 else 1
                        sum_val += cu * cv * dct_block[u, v] * math.cos((2*x + 1) * u * math.pi / (2*N)) * math.cos((2*y + 1) * v * math.pi / (2*N))
                
                block[x, y] = (2/N) * sum_val
        
        return block
    
    def quantize(self, dct_block: np.ndarray) -> np.ndarray:
        """Quantification basée sur la qualité"""
        quantization_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float64)
        
        scale = 50.0 / self.quality_factor if self.quality_factor < 50 else 2 - self.quality_factor/50.0
        quantization_table = quantization_table * scale
        quantization_table = np.maximum(quantization_table, 1)
        
        return np.round(dct_block / quantization_table) * quantization_table
    
    def compress(self, image: Image.Image) -> bytes:
        # Convertir en niveaux de gris
        if image.mode != 'L':
            image = image.convert('L')
        
        array = np.array(image, dtype=np.float64) - 128  # Centrer autour de 0
        
        # Padding pour avoir des dimensions multiples de 8
        h, w = array.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            array = np.pad(array, ((0, pad_h), (0, pad_w)), mode='constant')
        
        compressed_blocks = []
        
        # Traiter par blocs 8x8
        for i in range(0, array.shape[0], 8):
            for j in range(0, array.shape[1], 8):
                block = array[i:i+8, j:j+8]
                dct_block = self.dct2d(block)
                quantized_block = self.quantize(dct_block)
                compressed_blocks.append(quantized_block)
        
        data = {
            'blocks': compressed_blocks,
            'original_shape': (h, w),
            'padded_shape': array.shape
        }
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def decompress(self, data: bytes) -> Image.Image:
        data_dict = pickle.loads(data)
        blocks = data_dict['blocks']
        original_shape = data_dict['original_shape']
        padded_shape = data_dict['padded_shape']
        
        # Reconstruire l'image
        reconstructed = np.zeros(padded_shape, dtype=np.float64)
        block_idx = 0
        
        for i in range(0, padded_shape[0], 8):
            for j in range(0, padded_shape[1], 8):
                quantized_block = blocks[block_idx]
                dct_block = self.quantize(quantized_block)  # Re-quantifier
                block = self.idct2d(dct_block)
                reconstructed[i:i+8, j:j+8] = block
                block_idx += 1
        
        # Recadrer à la taille originale et recentrer
        reconstructed = reconstructed[:original_shape[0], :original_shape[1]] + 128
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        return Image.fromarray(reconstructed, mode='L')




class CompressionEvaluator:
    """Évaluateur de méthodes de compression"""
    
    def __init__(self):
        self.methods = [
            PNGCompression(),
            JPEGCompression(quality=85),
            HaarCompression(tolerance=5.0),
            DCTCompression(quality_factor=50)
        ]
    
    def evaluate_single_image(self, image_path: str) -> Dict[str, Any]:
        """Évalue toutes les méthodes sur une image"""
        original_image = Image.open(image_path)
        original_size = os.path.getsize(image_path)
        
        results = {'image_path': image_path, 'original_size': original_size, 'methods': {}}
        
        for method in self.methods:
            try:
                start_time = time.time()
                compressed_data = method.compress(original_image)
                compression_time = time.time() - start_time
                
                start_time = time.time()
                reconstructed_image = method.decompress(compressed_data)
                decompression_time = time.time() - start_time
                
                compressed_size = len(compressed_data)
                compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
                
                # Calcul métriques qualité
                orig_array = np.array(original_image.convert('L'), dtype=np.float64)
                recon_array = np.array(reconstructed_image.convert('L'), dtype=np.float64)
                
                if orig_array.shape != recon_array.shape:
                    recon_image_resized = reconstructed_image.resize(original_image.size, Image.LANCZOS)
                    recon_array = np.array(recon_image_resized.convert('L'), dtype=np.float64)
                
                psnr = calculate_psnr(orig_array, recon_array)
                ssim = calculate_ssim_simple(orig_array, recon_array)
                
                results['methods'][method.name] = {
                    'compressed_size': compressed_size,
                    'compression_ratio': compression_ratio,
                    'compression_time': compression_time,
                    'decompression_time': decompression_time,
                    'psnr': psnr,
                    'ssim': ssim
                }
                
            except Exception as e:
                results['methods'][method.name] = {'error': str(e)}
        
        return results 
