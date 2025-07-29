import time
import torch
import json
from typing import Dict
import numpy as np
from transformers import LongformerTokenizer, LongformerForMaskedLM


def create_longformer_sample_data(tokenizer, batch_size: int = 1, seq_len: int = 1024):
    """Longformer용 샘플 데이터 생성"""
    # 무작위 토큰 ID 생성 (vocab 범위 내에서)
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(1, vocab_size - 1, (batch_size, seq_len))
    
    # 특수 토큰 설정
    input_ids[:, 0] = tokenizer.cls_token_id  # [CLS]
    input_ids[:, -1] = tokenizer.sep_token_id  # [SEP]
    
    # Attention mask (모든 토큰 유효)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Global attention mask (CLS 토큰과 SEP 토큰에 global attention)
    global_attention_mask = torch.zeros(batch_size, seq_len)
    global_attention_mask[:, 0] = 1  # CLS 토큰
    global_attention_mask[:, -1] = 1  # SEP 토큰
    
    # MLM labels (15% 마스킹)
    labels = input_ids.clone()
    mask_prob = 0.15
    mask_indices = torch.bernoulli(torch.full(labels.shape, mask_prob)).bool()
    mask_indices[:, 0] = False  # [CLS] 마스킹 안함
    mask_indices[:, -1] = False  # [SEP] 마스킹 안함
    
    # 마스킹된 위치는 -100 (무시), 나머지는 원본 토큰
    labels[~mask_indices] = -100
    
    # 입력에서 마스킹된 위치를 [MASK] 토큰으로 교체
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_indices] = tokenizer.mask_token_id
    
    return {
        'input_ids': masked_input_ids,
        'attention_mask': attention_mask,
        'global_attention_mask': global_attention_mask,
        'labels': labels
    }


def benchmark_longformer_model(model, tokenizer, config: Dict):
    """Longformer 모델 벤치마킹 수행"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = {}
    
    # Longformer에 적합한 시퀀스 길이로 테스트 (긴 시퀀스 강조)
    test_configs = [
        {'batch_size': 1, 'seq_len': 1024},
        {'batch_size': 1, 'seq_len': 4096},
        {'batch_size': 4, 'seq_len': 1024},
        {'batch_size': 4, 'seq_len': 4096},
        {'batch_size': 16, 'seq_len': 1024},
        {'batch_size': 16, 'seq_len': 4096},
    ]
    
    for test_config in test_configs:
        batch_size = test_config['batch_size']
        seq_len = test_config['seq_len']
        
        print(f"Testing batch_size={batch_size}, seq_len={seq_len}")
        
        try:
            # Longformer 샘플 데이터 생성
            sample_data = create_longformer_sample_data(tokenizer, batch_size, seq_len)
            
            # GPU로 이동
            sample_data = {k: v.to(device) for k, v in sample_data.items()}
            
            # Warmup (3회)
            print("  Warming up...", end="")
            with torch.no_grad():
                for _ in range(3):
                    _ = model(**sample_data)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            print(" done")

            # 실제 측정 (10회)
            times = []
            memory_usage = []
            
            for i in range(10):
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # 메모리 사용량 측정 시작
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    start_memory = torch.cuda.memory_allocated()
                
                # 시간 측정 시작
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model(**sample_data)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                # 시간 측정 종료
                end_time = time.time()
                inference_time = end_time - start_time
                times.append(inference_time)
                
                # 메모리 사용량 측정 종료 (GB 단위)
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated()
                    memory_usage.append(peak_memory / 1024**3)  # GB 단위
                
                if (i + 1) % 2 == 0:  # 2회마다 출력 (더 간결하게)
                    print(f"  Run {i+1}: {inference_time:.4f}s", end="")
                    if torch.cuda.is_available():
                        print(f", Memory: {peak_memory/1024**3:.2f}GB")
                    else:
                        print()
            
            # 통계 계산
            config_key = f"batch_{batch_size}_seq_{seq_len}"
            results[config_key] = {
                'batch_size': batch_size,
                'seq_len': seq_len,
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'median_time': np.median(times),
                'times': times,
                'tokens_per_second': (batch_size * seq_len) / np.mean(times),
                'efficiency_ratio': seq_len / np.mean(times),  # 시퀀스 길이당 처리 속도
            }
            
            if memory_usage:
                results[config_key]['mean_memory_gb'] = np.mean(memory_usage)
                results[config_key]['peak_memory_gb'] = np.max(memory_usage)
            
            print(f"  Average: {np.mean(times):.4f}s ± {np.std(times):.4f}s")
            print(f"  Median: {np.median(times):.4f}s")
            print(f"  Tokens/sec: {(batch_size * seq_len) / np.mean(times):.1f}")
            print(f"  Efficiency: {seq_len / np.mean(times):.1f} (tokens/sec per seq_len)")
            if memory_usage:
                print(f"  Memory: {np.mean(memory_usage):.2f}GB (avg), {np.max(memory_usage):.2f}GB (peak)")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            results[config_key] = {'error': str(e)}
            continue
    
    return results


def main():
    print("=== Longformer Model Benchmark ===")
    
    # Longformer 모델 설정
    model_name = "allenai/longformer-base-4096"
    
    print(f"Model: {model_name}")
    print("Loading Longformer model and tokenizer...")
    
    try:
        # Longformer 모델과 토크나이저 로드
        tokenizer = LongformerTokenizer.from_pretrained(model_name)
        model = LongformerForMaskedLM.from_pretrained(model_name)

        model.config.attention_window = [64] * len(model.config.attention_window)
        
        # 모델 정보 출력
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # 디바이스 정보
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Target device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Longformer 특성 출력
        print(f"Max position embeddings: {model.config.max_position_embeddings}")
        print(f"Attention window: {model.config.attention_window}")
        print(f"Model hidden size: {model.config.hidden_size}")
        print()
        
        
    except Exception as e:
        print(f"Error loading Longformer model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 벤치마킹 수행
    print("Starting Longformer benchmark...")
    results = benchmark_longformer_model(model, tokenizer, {})
    
    # 결과 저장
    output_file = "longformer_benchmark_results.json"
    with open(output_file, 'w') as f:
        # numpy arrays를 list로 변환하여 JSON 직렬화 가능하게 만들기
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict) and 'times' in value:
                serializable_results[key] = {**value}
                serializable_results[key]['times'] = value['times'].tolist() if hasattr(value['times'], 'tolist') else value['times']
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # 요약 출력
    print("\n=== Longformer Benchmark Summary ===")
    for config_name, result in results.items():
        if 'error' in result:
            print(f"{config_name}: ERROR - {result['error']}")
        else:
            mem_info = ""
            if 'mean_memory_gb' in result and 'peak_memory_gb' in result:
                mem_info = f", Memory: {result['mean_memory_gb']:.2f}GB (avg), {result['peak_memory_gb']:.2f}GB (peak)"
            print(f"{config_name}: {result['mean_time']:.4f}s avg, {result['tokens_per_second']:.1f} tokens/sec{mem_info}")

if __name__ == "__main__":
    main()