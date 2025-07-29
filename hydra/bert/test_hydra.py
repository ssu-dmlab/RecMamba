import time
import torch
import json
from typing import Dict
import numpy as np
import os
import sys

print("Testing mamba_ssm import...")
try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    print("✓ mamba_ssm import successful")
except ImportError as e:
    print(f"✗ mamba_ssm import failed: {e}")

from src.bert_layers import BertForMaskedLM
from src.configuration_bert import BertConfig as HydraBertConfig
from transformers import BertTokenizer


def create_hydra_sample_data(tokenizer, batch_size: int = 1, seq_len: int = 1024):
    """Hydra BERT용 샘플 데이터 생성"""
    # 무작위 토큰 ID 생성 (vocab 범위 내에서)
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(1, vocab_size - 1, (batch_size, seq_len))
    
    # 특수 토큰 설정
    input_ids[:, 0] = tokenizer.cls_token_id  # [CLS]
    input_ids[:, -1] = tokenizer.sep_token_id  # [SEP]
    
    # Attention mask (모든 토큰 유효)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Token type IDs (모두 0으로 설정)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
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
        'token_type_ids': token_type_ids,
        'labels': labels
    }


def benchmark_hydra_model(model, tokenizer, config: Dict):
    """Hydra BERT 모델 벤치마킹 수행"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = {}
    
    # Hydra BERT에 적합한 시퀀스 길이로 테스트
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
            # Hydra BERT 샘플 데이터 생성
            sample_data = create_hydra_sample_data(tokenizer, batch_size, seq_len)
            
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
    print("=== Hydra BERT Model Benchmark ===")
    
    # Hydra BERT 모델 설정
    hydra_config_params = {
        "num_hidden_layers": 23,
        "max_position_embeddings": 1024,
        "use_position_embeddings": False,
        "hidden_size": 768,
    }
    
    print("Loading Hydra BERT model and tokenizer...")
    
    try:
        # Hydra BERT 설정 생성 (matrix_mixer_type: hydra)
        hydra_config = HydraBertConfig.from_pretrained(
            "bert-base-uncased", **hydra_config_params
        )
        # print(f"Hydra BERT Config: {hydra_config}")
        
        # Padding for divisibility by 8
        if hydra_config.vocab_size % 8 != 0:
            hydra_config.vocab_size += 8 - (hydra_config.vocab_size % 8)
        
        # 토크나이저 로드
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Hydra BERT 모델 생성
        model = BertForMaskedLM(hydra_config)
        
        # 체크포인트 로드 시도
        checkpoint_path = "../../hydra_bert_23layers.pt"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            model = BertForMaskedLM.from_composer(
                pretrained_checkpoint=checkpoint_path, 
                config=hydra_config
            )
        else:
            print("No checkpoint found, using randomly initialized model")
        
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
        
        # Hydra BERT 특성 출력
        print(f"Num hidden layers: {hydra_config.num_hidden_layers}")
        print(f"Max position embeddings: {hydra_config.max_position_embeddings}")
        print(f"Use position embeddings: {hydra_config.use_position_embeddings}")
        print(f"Hidden size: {hydra_config.hidden_size}")
        print(f"Vocab size: {hydra_config.vocab_size}")
        print()
        
    except Exception as e:
        print(f"Error loading Hydra BERT model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 벤치마킹 수행
    print("Starting Hydra BERT benchmark...")
    results = benchmark_hydra_model(model, tokenizer, {})
    
    # 결과 저장
    output_file = "hydra_bert_benchmark_results.json"
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
    print("\n=== Hydra BERT Benchmark Summary ===")
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