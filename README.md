````markdown
# Qwen-Video Whitebox VideoGEM

훈련 없이(Training-free) **VideoGEM (GEM의 비디오 확장)**을 **Qwen 2.0 기반 LLaVA-Video** 비전 타워에 **화이트박스 수준**으로 붙여,
- **q/k/v 가중치와 레이어 히든**을 직접 사용한 **self-self attention(q/q, k/k, v/v)**
- **정적/동적 레이어 가중**
- **프롬프트 분해(verb/object/action) + 좌표 가중 평균**
으로 **행동 히트맵과 위치**를 구합니다.

## 설치
```bash
pip install open-clip-torch==2.24.0 decord opencv-python matplotlib spacy
python -m spacy download en_core_web_sm
```

## 환경 변수(텍스트 인코더)

```bash
export GEM_TEXT_MODEL="ViT-L-14-336"
export GEM_TEXT_PRETRAINED="openai"
export GEM_DEVICE="cuda"
```

> Vision tower와 동일 계열(OpenCLIP/SigLIP) 텍스트 인코더를 매칭하세요.

## 실행

* 데이터셋 일괄:

```bash
bash qwen_gem/scripts/run_whitebox_gem.bash \
  --model_path "lmms-lab/LLaVA-Video-7B-Qwen2" \
  --data_path "DATAS/eval/VideoMME/formatted_dataset.json" \
  --video_root "DATAS/eval/VideoMME/videos/data/data" \
  --results_dir "qwen_gem/outputs/VideoMME"
```

* 단일 비디오:

```bash
bash qwen_gem/scripts/vis_one.bash \
  DATAS/eval/VideoMME/videos/data/data/xxxx.mp4 \
  "A photo of a person whisk eggs."
```

## 산출물

* `heatmap_{verb,object,action}_f{t}.png`: 프레임별 히트맵+peak 좌표
* `final_peaks.json`: 프레임별 peak 좌표 및 최종 좌표(verb:obj:action=1:1:3 가중 평균)
* `meta.json`: 그리드, CLS 유무, 레이어 가중(정적/동적/결합), 환경 기록

## 구현 노트

* **화이트박스**: `gem/vision_introspect.py`가 Qwen 비전타워에서

  * 블록 리스트, 각 블록의 **q/k/v projection Linear**(또는 qkv Linear)를 찾고,
  * 포워드 훅으로 **블록 출력 X_l**을 수집합니다.
* **동적 가중**: `Y_l ≈ X_l^CLS - X_{l-1}^CLS`로 residual 근사하여,
  `s_l = cos((x_CLS - Y_l), e_EOS)`로 레이어별 중요도를 구하고 마지막 D개에 softmax(-s·τ_d).
* **히트맵**: `cos(O_comb_patch, e_prompt)`를 패치 그리드로 reshape, 프레임 해상도로 업샘플.
* **좌표 결합**: verb/object/action 세 히트맵의 peak 좌표를 가중 평균.

## 주의/호환 이슈

* 비전타워에 CLS 토큰이 없으면 `has_cls=False`로 자동 처리(동적가중 비활성).
* 일부 구현체는 `attn.qkv` 대신 `attn.q_proj/k_proj/v_proj`를 사용 → 자동 분기.
* 텍스트-비전 임베딩 공간 미스매치는 성능에 영향. 가능하면 vision과 **동일 계열** OpenCLIP 텍스트 인코더 사용.
* OOM 시 `--max_frames`를 줄이거나 `GEM_DEVICE=cpu`로 검증 후 `cuda`로 전환.

````
