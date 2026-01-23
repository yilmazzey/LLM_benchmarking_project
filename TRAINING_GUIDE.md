# 🚀 LLaMA Fine-tuning Quick Start Guide

## 📋 Yeni Default Ayarlar

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| **Epochs** | 2 | Training epoch sayısı |
| **Batch Size** | 8 | Per-device batch size |
| **Gradient Accumulation** | 8 | Gradient accumulation steps |
| **Effective Batch Size** | 64 | (8 × 8) Gerçek batch size |
| **Eval Split** | 0.1 | 10% validation split |

## 🎯 Hızlı Başlangıç

### 1️⃣ Lokal Bilgisayarda (Dataset Hazırlama)

```bash
# Preprocessed dataset oluştur
python data_process.py --input ./veri.json --output ./processed_dataset

# Servera yükle
scp -r processed_dataset/ your-server:~/LLM_benchmarking_project/
```

### 2️⃣ Serverde (Otomatik Training)

```bash
# Script'i çalıştır (default ayarlarla)
./setup_and_train.sh

# veya özel ayarlarla
./setup_and_train.sh --models "1b 3b" --epochs 3 --batch-size 4
```

## 📝 Kullanım Örnekleri

### Default Ayarlarla (Önerilen)
```bash
./setup_and_train.sh
```
**Çalıştırılan komutlar:**
```bash
python llama-ft.py --model 1b --epochs 2 --batch_size 8 --gradient_accumulation 8 --eval_split 0.1 --use_processed --processed_path ./processed_dataset
python llama-ft.py --model 3b --epochs 2 --batch_size 8 --gradient_accumulation 8 --eval_split 0.1 --use_processed --processed_path ./processed_dataset
python llama-ft.py --model 8b --epochs 2 --batch_size 8 --gradient_accumulation 8 --eval_split 0.1 --use_processed --processed_path ./processed_dataset
```

### Sadece Belirli Modeller
```bash
./setup_and_train.sh --models "1b 3b"
```

### Özel Training Parametreleri
```bash
# Daha küçük batch size, daha fazla accumulation
./setup_and_train.sh --batch-size 4 --grad-accum 16

# Daha fazla epoch
./setup_and_train.sh --epochs 5

# Validation olmadan
./setup_and_train.sh --eval-split 0.0
```

### Tek Model, Özel Ayarlar
```bash
./setup_and_train.sh --models "8b" --epochs 3 --batch-size 4
```

## 🖥️ Tmux Yönetimi

### Session Kontrolü
```bash
# Training session'a bağlan
tmux attach -t llama-training

# Session'dan ayrıl (training devam eder!)
# Tuş kombinasyonu: Ctrl+b then d

# Tmux session listesi
tmux ls

# Training'i durdur ve temizle
./setup_and_train.sh --cleanup
```

### Pencere Gezintisi
```bash
Ctrl+b then 0    # Status/Control panel
Ctrl+b then 1    # 1B model training + log
Ctrl+b then 2    # 3B model training + log
Ctrl+b then 3    # 8B model training + log
Ctrl+b then 4    # GPU monitoring (nvidia-smi)
Ctrl+b then 5    # Combined logs (tüm modeller)

Ctrl+b then d    # Detach (ayrıl)
Ctrl+c           # Stop tail (log takibini durdur)
```

## 📊 Training Monitoring

### Log Dosyaları
```bash
# Canlı log takibi
tail -f train_1b.log
tail -f train_3b.log
tail -f train_8b.log

# Tüm loglara birden bak
tail -f train_*.log
```

### GPU Monitoring
```bash
# Tmux window 4'te otomatik çalışıyor
# veya manuel:
watch -n 2 nvidia-smi
```

### Training Process'leri
```bash
# Çalışan training'leri kontrol et
ps aux | grep llama-ft.py

# Process'leri öldür (veya cleanup kullan)
pkill -f llama-ft.py
```

## 🎛️ Tüm Parametreler

```bash
./setup_and_train.sh [OPTIONS]

Options:
  --clone                 Repo'yu sil ve yeniden clone et
  --skip-deps             Dependency kurulumunu atla
  --models "1b 3b 8b"     Train edilecek modeller (default: "1b 3b 8b")
  --epochs N              Epoch sayısı (default: 2)
  --batch-size N          Per-device batch size (default: 8)
  --grad-accum N          Gradient accumulation (default: 8)
  --eval-split FLOAT      Validation split ratio (default: 0.1)
  --resume                Mevcut tmux session'a bağlan
  --cleanup               Tüm training'i durdur ve temizle
  --help                  Yardım mesajı
```

## 📈 Performans Tahminleri

### Memory Kullanımı (H100 80GB)
| Model | Batch Size 1 | Batch Size 8 | Parallel (3 model) |
|-------|--------------|--------------|-------------------|
| 1B    | ~10GB        | ~15GB        | ✅ Rahat çalışır   |
| 3B    | ~12GB        | ~18GB        | ✅ Rahat çalışır   |
| 8B    | ~15GB        | ~22GB        | ✅ Rahat çalışır   |
| **Total** | ~37GB    | ~55GB        | **55GB / 80GB**   |

### Training Hızı
- **Batch Size 8**: Batch Size 1'e göre ~8x daha az step
- **Beklenen süre**: ~3-6 saat (model boyutuna göre)
- **Effective Batch Size 64**: Stabil ve hızlı convergence

### Validation Split (10%)
- **Training data**: %90 (~1350 sample)
- **Validation data**: %10 (~150 sample)
- **Avantajlar**:
  - Overfitting kontrolü
  - Real-time validation loss tracking
  - Best checkpoint selection

## 🔍 Troubleshooting

### Problem: processed_dataset bulunamadı
```bash
# Hata mesajı:
ERROR: processed_dataset/ directory not found!

# Çözüm:
python data_process.py --input ./veri.json --output ./processed_dataset
```

### Problem: GPU memory yetersiz
```bash
# Batch size'ı düşür:
./setup_and_train.sh --batch-size 4

# Veya sadece 1-2 model train et:
./setup_and_train.sh --models "1b"
```

### Problem: Tmux session zaten var
```bash
# Hata mesajı:
WARNING: Tmux session 'llama-training' already exists

# Çözüm 1: Mevcut session'a bağlan
./setup_and_train.sh --resume

# Çözüm 2: Eski session'ı temizle
./setup_and_train.sh --cleanup

# Çözüm 3: Manuel temizle
tmux kill-session -t llama-training
```

### Problem: Training crash oldu
```bash
# Log'ları kontrol et
tail -100 train_1b.log

# Training'i yeniden başlat
./setup_and_train.sh --cleanup
./setup_and_train.sh
```

## 💡 Pro Tips

1. **SSH Disconnect Sonrası**: `tmux attach -t llama-training` ile tekrar bağlan
2. **Log Takibi**: Tmux window 5'te tüm loglara aynı anda bak
3. **GPU Monitoring**: Tmux window 4'te nvidia-smi otomatik yenilenir
4. **Multiple Experiments**: Her experiment için farklı branch kullan
5. **Background Training**: Tmux session açık kalır, SSH disconnect olsa bile

## 📦 Checkpoint'ler

Training sonrası checkpoint'ler:
```
lora_adapters/
├── llama-3.2-1b-abstract-lora_ep2_20260123_143022/
│   ├── checkpoint-XXX/
│   └── final/  ← Final LoRA adapters
├── llama-3.2-3b-abstract-lora_ep2_20260123_143022/
│   ├── checkpoint-XXX/
│   └── final/
└── llama-3.1-8b-abstract-lora_ep2_20260123_143022/
    ├── checkpoint-XXX/
    └── final/
```

## 🎓 En İyi Pratikler

1. ✅ **Her zaman preprocessed dataset kullan**: `--use_processed` (script'te otomatik)
2. ✅ **Validation split kullan**: Overfitting kontrolü için (default: 0.1)
3. ✅ **Batch size 8**: Optimal hız/memory dengesi
4. ✅ **Effective batch 64**: Stabil training için ideal
5. ✅ **Tmux kullan**: SSH disconnect'te training devam eder
6. ✅ **Log'ları izle**: Real-time progress tracking
7. ✅ **Cleanup yap**: Test sonrası her zaman cleanup

## 🆘 Hızlı Komutlar

```bash
# Başlat
./setup_and_train.sh

# İzle
tmux attach -t llama-training

# Ayrıl
Ctrl+b then d

# Durdur
./setup_and_train.sh --cleanup

# Yardım
./setup_and_train.sh --help
```

---

**Happy Training! 🚀**
