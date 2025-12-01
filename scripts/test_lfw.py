#!/usr/bin/env python3
"""
Test Marcus on the LFW dataset.

Enrols a subset of LFW identities, then tests recognition accuracy.
"""

import sys
import os
from pathlib import Path

# Suppress warnings
os.environ["ONNXRUNTIME_LOGGING_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

import random
from PIL import Image
import numpy as np
from collections import defaultdict


def load_lfw_dataset(lfw_path: str, min_images: int = 5, max_identities: int = 100):
    """Load LFW dataset, filtering for people with enough images."""
    
    lfw_dir = Path(lfw_path)
    identity_images = defaultdict(list)
    
    print(f"Scanning {lfw_path}...")
    
    for person_dir in sorted(lfw_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        
        images = list(person_dir.glob("*.jpg"))
        if len(images) >= min_images:
            identity_images[person_dir.name] = [str(p) for p in images]
    
    # Limit to max_identities
    identities = list(identity_images.keys())[:max_identities]
    filtered = {k: identity_images[k] for k in identities}
    
    total_images = sum(len(v) for v in filtered.values())
    print(f"Found {len(filtered)} identities with {total_images} images (min {min_images} per person)")
    
    return filtered


def test_marcus_on_lfw(
    lfw_path: str = "/Users/selin/Desktop/PROJECTS/Computer Vision/Marcus/lfw/lfw-deepfunneled/lfw-deepfunneled",
    num_identities: int = 20,
    enrol_images: int = 2,
    test_images: int = 2
):
    """
    Test Marcus recognition on LFW.
    
    For each identity:
    - Enrol first N images
    - Test with remaining images
    - Check if correct identity is in top-K results
    """
    
    print("\n" + "=" * 60)
    print("MARCUS LFW RECOGNITION TEST")
    print("=" * 60)
    
    # Load dataset
    identity_images = load_lfw_dataset(lfw_path, min_images=enrol_images + test_images, max_identities=num_identities)
    
    if len(identity_images) == 0:
        print("ERROR: No identities found with enough images.")
        return
    
    # Initialise pipeline ONCE
    print("\nInitialising Marcus pipeline...")
    from marcus_core.pipeline import FacialPipeline
    from marcus_core.config import SystemConfig
    
    config = SystemConfig()
    pipeline = FacialPipeline(config)
    
    # Warm up the models
    print("Warming up models...")
    pipeline.warmup()
    
    # Enrol identities
    print(f"\n--- Enrolling {len(identity_images)} identities ({enrol_images} images each) ---")
    
    enrolled_count = 0
    for name, images in identity_images.items():
        enrol_set = images[:enrol_images]
        
        for img_path in enrol_set:
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                result = pipeline.enroll(image=img, name=name, source="lfw")
                if result:
                    enrolled_count += 1
                    print(f"  Enrolled: {name}")
            except Exception as e:
                print(f"  Error enrolling {name}: {e}")
        
    print(f"\nTotal enrolled: {enrolled_count} images")
    
    # Test recognition
    print(f"\n--- Testing Recognition ---")
    
    # Debug: Check what's in the vector store
    stats = pipeline.matcher.get_statistics()
    print(f"DEBUG: {stats}")
    print(f"DEBUG: Vector store count: {pipeline.matcher.vector_store.count()}")
    print(f"DEBUG: FAISS index ntotal: {pipeline.matcher.vector_store._index.ntotal}")
    print(f"DEBUG: Is trained: {pipeline.matcher.vector_store._is_trained}")
    
    # Debug: test a manual search 
    test_img = np.array(Image.open(list(identity_images.values())[0][0]).convert("RGB"))
    test_results = pipeline.detect_and_embed(test_img)
    if test_results and test_results[0].embedding is not None:
        emb = test_results[0].embedding
        print(f"DEBUG: Test embedding shape: {emb.shape}, norm: {np.linalg.norm(emb):.4f}")
        
        # Direct FAISS search
        import faiss
        query = emb.reshape(1, -1).astype(np.float32)
        query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-10)
        distances, indices = pipeline.matcher.vector_store._index.search(query, 5)
        print(f"DEBUG: Direct FAISS search - distances: {distances[0]}, indices: {indices[0]}")
        
        raw_results = pipeline.matcher.vector_store.search(emb, top_k=5, threshold=0.0)
        print(f"DEBUG: Raw vector search returned {len(raw_results)} results")
        if raw_results:
            for r in raw_results[:3]:
                print(f"DEBUG:   - {r.metadata.get('name')}: score={r.score:.4f}")
    
    correct_top1 = 0
    correct_top5 = 0
    total_tests = 0
    results_detail = []
    
    for name, images in identity_images.items():
        test_set = images[enrol_images:enrol_images + test_images]
        
        for img_path in test_set:
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                results = pipeline.search(img, top_k=5, threshold=0.0)
                
                if results and len(results) > 0 and results[0].matches:
                    matches = results[0].matches
                    top1_name = matches[0].identity.name if matches else None
                    top5_names = [m.identity.name for m in matches[:5]]
                    top1_score = matches[0].score if matches else 0
                    
                    is_top1 = top1_name == name
                    is_top5 = name in top5_names
                    
                    if is_top1:
                        correct_top1 += 1
                    if is_top5:
                        correct_top5 += 1
                    
                    status = "CORRECT" if is_top1 else "WRONG"
                    print(f"  {name}: {top1_name} ({top1_score:.2%}) - {status}")
                    
                    results_detail.append({
                        "true": name,
                        "predicted": top1_name,
                        "score": top1_score,
                        "correct": is_top1
                    })
                else:
                    print(f"  {name}: No matches found")
                    results_detail.append({
                        "true": name,
                        "predicted": None,
                        "score": 0,
                        "correct": False
                    })
                
                total_tests += 1
                
            except Exception as e:
                print(f"  Error testing {name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    top1_acc = (correct_top1 / total_tests * 100) if total_tests > 0 else 0
    top5_acc = (correct_top5 / total_tests * 100) if total_tests > 0 else 0
    
    print(f"Total tests:     {total_tests}")
    print(f"Top-1 Accuracy:  {correct_top1}/{total_tests} ({top1_acc:.1f}%)")
    print(f"Top-5 Accuracy:  {correct_top5}/{total_tests} ({top5_acc:.1f}%)")
    
    print("=" * 60)
    
    return {
        "total_tests": total_tests,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc
    }


if __name__ == "__main__":
    test_marcus_on_lfw(
        num_identities=20,  # Test with 20 people
        enrol_images=2,     # Enrol 2 images per person
        test_images=2       # Test with 2 images per person
    )
