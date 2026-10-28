# MRDT

여러 시점에서 찍은 물체 사진(context 1–5장)과 원하는 방향(azimuth, elevation)을 받아, 그 방향에서 본 **256×256 이미지 한 장**을 생성하는 diffusion transformer입니다. MVImgNet2.0 실물 데이터로 처음부터(scratch) 학습합니다.

- **생성기:** RDT 기반 transformer, 약 2.57억 parameter, depth 14, hidden 1024, mlp ratio 4. timestep과 목표 방향은 adaLN-single로 주입합니다.
- **조건:**
  - context 이미지의 SD-VAE latent를 target token과 같은 self-attention에 넣습니다(joint).
  - 같은 이미지의 frozen SigLIP feature는 cross-attention으로 넣습니다.
  - 선택적으로 class 이름 문장(T5-XXL)을 넣습니다.
- **학습:**
  - x0(clean latent) 예측, Min-SNR-5 가중치
  - condition dropout 10% + CFG
  - EMA, bf16 mixed precision
  - global batch 64, lr 1e-4 constant, 200k step
- **세 가지 설정(run):**

| run | 인자 | cross-attention이 보는 것 |
|---|---|---|
| null-text | `--cross-attn alternate --text-cond null` | 짝수 block: 학습되는 null token, 홀수 block: SigLIP |
| image-only | `--cross-attn image` | 모든 block: SigLIP |
| class-text | `--cross-attn alternate --text-cond class --text-bank <bank> --image-only-drop-prob 0.05 --text-only-drop-prob 0.05` | 짝수 block: class 문장(T5-XXL), 홀수 block: SigLIP |

## 1. 설치

검증한 환경은 Linux, Python 3.10, conda입니다.

```bash
conda create -n mrdt python=3.10 -y
conda activate mrdt
# Ampere 등(예: RTX 3090)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# Blackwell(예: RTX PRO 6000)은 대신: pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-mrdt.txt
```

Blackwell GPU 2장 DDP에서 torch 2.7.1에 포함된 NCCL 2.26.2가 `illegal memory access`로 실패한 적이 있습니다. 그런 경우 NCCL 2.27.7의 `libnccl.so.2`를 `LD_PRELOAD`로 지정하고 `NCCL_P2P_DISABLE=1`을 주면 동작했습니다.

## 2. 사전학습 weight (frozen)

학습과 평가에는 SigLIP과 SD-VAE가 필요합니다. 기본 위치는 `<repo>/pretrained`이고, 다른 곳에 두려면 `MRDT_MODEL_ROOT`를 지정합니다.

```bash
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir pretrained/siglip-so400m-patch14-384
huggingface-cli download stabilityai/sd-vae-ft-mse         --local-dir pretrained/sd-vae-ft-mse
huggingface-cli download google/t5-v1_1-xxl                 --local-dir pretrained/t5-v1_1-xxl   # class-text bank를 만들 때만 필요
export MRDT_MODEL_ROOT=$PWD/pretrained
```

## 3. 데이터 준비

MVImgNet2.0 part 1(archive 41개, 약 2.4TB, 309 class)을 사용합니다. 데이터는 이 저장소에 포함되지 않으며, [공식 저장소](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet2.0)에서 신청해 다운로드 링크와 비밀번호를 받아야 합니다.

### 3.1 다운로드

`scripts/mvimgnet_download.py`는 이어받기와 archive별 quickXorHash 검증을 합니다. 별도 환경이 필요합니다.

```bash
pip install requests quickxorhash pyppeteer      # 별도 conda 환경 권장
export MVIMGNET_PASSWORD=...                     # 신청 메일의 비밀번호
python scripts/mvimgnet_download.py --url <신청 메일의 공유 링크> --out data/MVImgNet2.0/raw --jobs 8
```

### 3.2 압축 해제와 검증

압축을 풀면서 모든 파일의 SHA256과 RGB/mask decode를 확인하고, 통과한 archive만 지웁니다. `--watch`를 주면 다운로드와 동시에 진행합니다.

```bash
python -m plan_b.mvimgnet.extract_verify_archives --raw data/MVImgNet2.0/raw --data data/MVImgNet2.0/prepared \
    --category-json <공식 저장소>/mvimgnet2_category.json --delete-archives --watch
```

### 3.3 catalog 생성

```bash
bash plan_b/mvimgnet/build_catalog.sh data/MVImgNet2.0/prepared data/MVImgNet2.0/catalog_full 24
export MRDT_CATALOG=$PWD/data/MVImgNet2.0/catalog_full/catalog
```

이 스크립트는 네 단계를 차례로 실행합니다.

1. 검증 기록 확정(`--finalize`)
2. camera 정리(`index_verified_full_bin15`)
3. 전체 mask 검사(`audit_full_masks`)
4. catalog 작성(`build_masked_catalog`)

결과 object 수는 train 165,711 / val 9,309 / test 9,258입니다.

### 3.4 view cache

학습할 때마다 원본 사진을 전처리하지 않도록 view를 미리 계산해 둡니다. 결과는 학습 때 계산하는 것과 정확히 같습니다. 약 137GB이고, 빠른 디스크(SSD)에 두는 것을 권장합니다.

```bash
python -m plan_b.mvimgnet.build_view_cache --catalog $MRDT_CATALOG --out /path/to/ssd/mrdt_cache
export MRDT_VIEW_CACHE=/path/to/ssd/mrdt_cache
```

### 3.5 VAE latent cache

view마다 SD-VAE latent를 fp32(TF32 off)로 한 번만 계산해 둡니다. 약 88GB이고, RTX 3090 4장에서 약 5시간 걸립니다. 다시 실행하면 끝난 object는 건너뜁니다.

```bash
bash scripts/build_latents.sh $MRDT_CATALOG $MRDT_VIEW_CACHE $MRDT_MODEL_ROOT/sd-vae-ft-mse /path/to/ssd/mrdt_latents 0,1,2,3
export MRDT_LATENT_CACHE=/path/to/ssd/mrdt_latents
```

### 3.6 class-text bank (class-text run만)

309개 class 이름을 `a photo of {a|an} {class 이름}` 문장으로 만들어 T5-XXL로 한 번만 encoding합니다. 이름 3개는 데이터를 직접 보고 바꿨습니다.

| class | 바꾼 이름 |
|---|---|
| 571 | artificial flower |
| 321 | liquid glue |
| 453 | glue stick |

```bash
python -m plan_b.mvimgnet.class_text --catalog $MRDT_CATALOG --t5 $MRDT_MODEL_ROOT/t5-v1_1-xxl \
    --out data/MVImgNet2.0/catalog_full/class_text_t5xxl.pt
```

## 4. 학습

`scripts/launch_run.sh`는 다음 일을 합니다.

1. repo를 `<run>.launch/code`에 복사하고 그 사본으로 학습합니다. 이후 작업 폴더를 고쳐도 실행 중인 학습에 영향이 없습니다.
2. 5k step마다 milestone 평가를 자동으로 돌리는 watcher를 같이 띄웁니다.

고정 설정은 bf16, latent cache, lr 1e-4 constant, warmup 1000, 200k step입니다.

```bash
# RTX 3090 4장: GPU당 16 = 8 × accum 2 (global batch 64), 학습 GPU당 약 13GB, 평가를 같이 돌리는 GPU0 약 20GB
bash scripts/launch_run.sh scratch runs/main_nulltext   0,1,2,3 16 2 0 --cross-attn alternate --text-cond null
bash scripts/launch_run.sh scratch runs/main_image      0,1,2,3 16 2 0 --cross-attn image
bash scripts/launch_run.sh scratch runs/main_classtext  0,1,2,3 16 2 0 --cross-attn alternate --text-cond class \
    --text-bank data/MVImgNet2.0/catalog_full/class_text_t5xxl.pt --image-only-drop-prob 0.05 --text-only-drop-prob 0.05
```

**GPU 구성 바꾸기**
- GPU 수, GPU당 batch, accum은 global batch가 64가 되도록 맞추면 됩니다. 예: GPU 2장 `32 1`, GPU 1장(80GB 이상) `64 1` + `MRDT_MEM_CAP=<GPU 용량>`.
- 매 step에서 쓰는 global sample 집합은 GPU 구성과 무관하게 같습니다.

**걸린 시간**
- RTX 3090 4장: 약 1.7 s/step(평가 포함), 200k step에 약 4일.
- RTX PRO 6000 1장: 약 1.3 s/step.

**중단과 재개**
- 중단: `touch runs/<run>/STOP`. 다음 step이 끝나면 checkpoint를 저장하고 멈춥니다.
- 재개: `scratch` 대신 `resume`을 쓰고 인자는 같게 줍니다.
- GPU 구성을 바꿔 재개하려면 `--reshard-resume`을 추가합니다(global batch는 같아야 함).

**저장 공간:** checkpoint는 하나에 약 4.07GB(model, EMA, AdamW)이고, milestone 40개면 run당 약 165GB입니다.

## 5. 평가

학습 중에는 `scripts/eval_milestones.sh`가 milestone마다 자동으로 평가합니다. 직접 평가하려면 다음을 실행합니다.

```bash
python -m plan_b.mvimgnet.eval_nvs --checkpoint runs/main_nulltext/checkpoints/step_200000.pt --catalog $MRDT_CATALOG \
    --view-cache $MRDT_VIEW_CACHE --split val --max-objects 1000 --contexts 1 3 5 --batch 8 --out runs/main_nulltext/eval/step_200000
```

**평가 방식 (Evaluation)**
- **평가 쌍:** val split에서 고정 순서로 object 1000개를 고르고, object마다 target view 1장과 context view를 고정 seed(2026)로 정합니다. target은 모든 k에서 같습니다.
- **k:** context 이미지 수입니다. view가 부족한 object를 빼서 k=3은 996쌍, k=5는 995쌍입니다.
- **생성:** EMA weight, DPM-Solver++ 20 step, CFG 3, 생성 seed 1234.
- **지표:**
  - PSNR, SSIM, LPIPS(AlexNet; VGG도 기록): 256×256 전체 이미지와 정답 mask의 전경 영역에서 따로 계산
  - FID, KID: 같은 쌍의 정답 target 이미지와 비교
- 결과는 `metrics.json`과 이미지별 CSV로 저장됩니다.

## 6. 샘플 생성

```bash
python mrdt_sample.py --checkpoint runs/main_nulltext/checkpoints/step_200000.pt --catalog $MRDT_CATALOG \
    --view-cache $MRDT_VIEW_CACHE --split val --index 0 --azimuth 30 --elevation 10 --output sample.png
```

- azimuth와 elevation은 object마다 정한 기준 view를 0°로 한 각도입니다.
- 기준 view의 화면 위쪽이 up 축입니다.
- `--context-ids`로 context 이미지를 직접 고를 수 있습니다(1–5장).

## 7. 결과

val 1000쌍 기준입니다. ↑는 클수록, ↓는 작을수록 좋습니다. k=5가 context를 모두 쓴 경우입니다.

| step | k | run | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | KID ↓ |
|---|---|---|---|---|---|---|---|
| 100k | 1 | null-text | 19.73 | 0.7753 | 0.2323 | 50.13 | 0.0073 |
| 100k | 1 | image-only | 19.74 | 0.7756 | 0.2333 | 50.52 | 0.0075 |
| 100k | 1 | class-text | 19.76 | 0.7753 | 0.2324 | 51.35 | 0.0078 |
| 100k | 3 | null-text | 22.13 | 0.8111 | 0.1483 | 43.35 | 0.0053 |
| 100k | 3 | image-only | 22.01 | 0.8091 | 0.1523 | 43.23 | 0.0054 |
| 100k | 3 | class-text | 22.05 | 0.8096 | 0.1507 | 43.84 | 0.0057 |
| 100k | 5 | null-text | 23.14 | 0.8263 | 0.1227 | 41.41 | 0.0052 |
| 100k | 5 | image-only | 23.04 | 0.8245 | 0.1252 | 41.04 | 0.0050 |
| 100k | 5 | class-text | 23.04 | 0.8245 | 0.1249 | 41.80 | 0.0052 |
| 200k | 1 | null-text | 20.27 | 0.7844 | 0.2115 | 41.99 | 0.0041 |
| 200k | 1 | image-only | 20.22 | 0.7841 | 0.2113 | 42.29 | 0.0042 |
| 200k | 1 | class-text | TBD | TBD | TBD | TBD | TBD |
| 200k | 3 | null-text | 23.21 | 0.8293 | 0.1237 | 36.01 | 0.0031 |
| 200k | 3 | image-only | 22.99 | 0.8260 | 0.1281 | 35.78 | 0.0030 |
| 200k | 3 | class-text | TBD | TBD | TBD | TBD | TBD |
| 200k | 5 | null-text | 24.38 | 0.8467 | 0.1013 | 33.90 | 0.0029 |
| 200k | 5 | image-only | 24.09 | 0.8426 | 0.1054 | 33.79 | 0.0030 |
| 200k | 5 | class-text | TBD | TBD | TBD | TBD | TBD |

## 8. 재현 관련 메모

- `configs/`에는 위 세 run이 실제로 사용한 설정(`config_used.json`)이 있습니다. 서버 경로만 `<MRDT_CATALOG>` 같은 placeholder로 바꿨습니다.
- 설정에는 학습 소스 파일의 SHA256(`sources`, `model_sources`)과 데이터 identity(`data_identity`)가 기록되어 있습니다. resume할 때 둘 중 하나라도 다르면 trainer가 거부합니다.
- 실행 이력:
  - null-text와 class-text: RTX 3090 4장(4 × 16, accum 2)
  - image-only: RTX PRO 6000 2장(2 × 32)으로 시작해, step 90,498부터 GPU 1장(64, `--reshard-resume`)으로 이어 학습했습니다. global batch와 step마다의 sample 집합은 같고, noise와 dropout random 값만 다릅니다(`configs/imageonly.json`의 `reshard_history`).
- 이 공개본은 결과를 낸 코드와 계산이 같습니다. 학습 코드에서 바뀐 것은 SigLIP/VAE 기본 경로 한 줄뿐이라, 그 파일의 SHA256만 `configs/`의 값과 다릅니다.

## 9. 테스트

```bash
python -m pytest tests_plan_b plan_b/mvimgnet/test_epoch_sampling.py   # 데이터가 필요한 일부 테스트는 자동으로 건너뜀
python -m tests_plan_b.smoke_current                                   # 작은 모델로 CPU forward, backward, sampling
```

## 출처와 라이선스

- 생성기 backbone은 [RDT (thu-ml/RoboticsDiffusionTransformer)](https://github.com/thu-ml/RoboticsDiffusionTransformer)를 바탕으로 하고, MIT License(`LICENSE`)를 따릅니다.
- 이 저장소는 [LimHaksoo/MRDT](https://github.com/LimHaksoo/MRDT)의 공유 코드에서 출발했습니다.
- 데이터는 [MVImgNet2.0](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet2.0)의 이용 조건을 따릅니다.
