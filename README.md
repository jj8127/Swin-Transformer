# Swin-Transformer
# Swin Transformer: Shifted Window Transformer

---

## **1. 등장 배경**

- **ViT**는 고해상도 이미지의 경우 모든 패치 간 self-attention 계산으로 인해 연산량이 제곱에 비례하여 증가.
- **ViT**는 local 정보를 보지 못함.
- 영상 데이터에서 물체의 크기 변화(scale of visual entities)에 따른 인식 문제를 기존 CNN은 계층적 구조(feature pyramid)를 통해 해결.
- **ViT는 Backbone으로 활용 불가능**.

---

## **2. Swin Transformer가 문제를 해결한 방법**

1. **고해상도 입력 이미지 처리의 비효율성**: Window 기반의 Self-Attention 설계.
2. **Local 구조의 부족한 활용**: 계층적 피라미드 구조로 이미지 국소적 특성 문제 해결.

### **2.1 ViT와 Swin Transformer 연산량 비교**

#### **(1) ViT (Vision Transformer)**

\[ O_{\text{ViT}} = N^2 \cdot D \quad \text{(여기서 } N = \frac{H \cdot W}{P^2} \text{)} \]

- \( H, W \): 입력 이미지의 높이와 너비.
- \( P \): 패치 크기.
- \( D \): 임베딩 차원.
- \( N \): 패치의 총 개수.

#### **(2) Swin Transformer**

\[ O_{\text{Swin}} = H \cdot W \cdot M^2 \cdot D \]

- \( H, W \): 입력 이미지의 높이와 너비.
- \( M \): 윈도 크기.
- \( D \): 임베딩 차원.

#### **(3) 실제 연산량 비교**

- **ViT**: 약 \( 29.4 \times 10^6 \) FLOPs.
- **Swin Transformer**: 약 \( 1.47 \times 10^6 \) FLOPs.

비교 결과:
- ViT의 연산량은 패치 수의 제곱(\( N^2 \))에 비례해 이미지 크기가 커질수록 급격히 증가.
- Swin Transformer의 연산량은 입력 이미지 크기(\( H \cdot W \))에 선형적으로 비례하며, 윈도 크기(\( M^2 \))의 영향을 받음.
- Swin Transformer가 ViT보다 약 20배 이상 효율적임.

---

## **3. ViT와 Swin Transformer 구조 비교**

### **Swin Transformer의 특징**

Swin Transformer는 ViT의 단점을 보완하기 위해 **윈도(window)** 단위의 로컬 Attention을 적용하고, 이를 점진적으로 확장하여 전역적 특성을 학습한다.

#### **구조적 특징**

- **윈도 기반 처리 (W-MSA)**: 이미지를 고정 크기 \( M \times M \) 윈도로 나눈 뒤, 각 윈도 내에서만 Self-Attention 수행.
- **전환된 윈도(SW-MSA)**: 다음 레이어에서 윈도 위치를 이동(Sliding)하여 이전 윈도 간의 상호작용을 학습.
- **다단계 구조**: 해상도를 점진적으로 축소하여 전역적 특징 학습.
- **MLP와 Norm Layer**: ViT와 동일하게 사용.
- **연산량**: \( \Omega(W-MSA) = 4hwC^2 + 2M^2hwC \), 여기서 \( M \)은 윈도의 크기. 연산량은 입력 크기 \( hw \)와 윈도 크기 \( M^2 \)에 선형적으로 비례.

#### **장점**

- **효율적인 연산**: 윈도 기반 처리로 연산량 크게 감소.
- **로컬-전역 학습**: 윈도 내 로컬 특징과 전환된 윈도를 통한 전역 관계 학습.

| 항목             | Vision Transformer (ViT)               | Swin Transformer                            |
|------------------|----------------------------------------|---------------------------------------------|
| **Attention 범위** | 전역 (Global)                         | 로컬 (윈도 내부) + 전환된 윈도 (전역 효과)    |
| **연산량**        | \( O(N^2) \) (패치 수의 제곱)         | \( O(hwM^2) \) (윈도 크기 선형)             |
| **로컬 정보 학습** | 부족                                  | 강함                                        |
| **적용 가능성**    | 주로 분류(Classification)             | 분류(Classification), 검출(Detection), 세분화(Segmentation) |

---

## **4. Swin Transformer 구조**

### **구성 단계**

1. **Patch Partition**
   - 입력 이미지 (\( H \times W \times 3 \))를 \( 4 \times 4 \) 크기의 패치로 나눔.
   - 각 패치를 하나의 벡터로 표현.
   - 패치 수는 \( \frac{H}{4} \times \frac{W}{4} \)가 되며, 각 패치의 벡터 차원은 48 (\( 4 \times 4 \times 3 \)).

2. **Linear Embedding**
   - 패치를 선형 변환(Dense layer)을 통해 차원을 \( C \)로 변환.
   - 변환 후 크기: \( \frac{H}{4} \times \frac{W}{4} \times C \).

3. **Swin Transformer Blocks**
   - **Stage 1**: 입력 크기 \( \frac{H}{4} \times \frac{W}{4} \times C \), 블록 2회 반복.
   - **Patch Merging**: 해상도를 절반으로 줄이고 채널 수를 \( 2C \)로 증가.
   - **Stage 2**: 입력 크기 \( \frac{H}{8} \times \frac{W}{8} \times 2C \), 블록 2회 반복.
   - **Patch Merging**: 해상도를 절반으로 줄이고 채널 수를 \( 4C \)로 증가.
   - **Stage 3, Stage 4**: 동일한 패턴으로 진행하며 채널 수 및 해상도 조정.

---

## **5. Shift Window Transformer**

- Swin Transformer는 윈도 내부 패치 간 Self-Attention만 수행하여 연산량을 감소시킴.
- 윈도 경계에서의 Attention 부족 문제를 해결하기 위해 **Shift Window** 개념 도입.

### **문제점 및 해결**

1. **문제점**
   - Window 경계끼리의 Attention이 불가능.
2. **Shift Window 개념 도입**
   - 윈도를 Shift하여 경계 간 상호작용 수행.
   - Shift 후 연산량 증가 문제 해결을 위해 **Cyclic Shifting** 사용.
   - Masking으로 연산량 효율 관리 및 상호작용 유지.
   - Reverse Cyclic Shift로 원래 이미지 복원.

---

# Swin Transformer

## 6. Relative Position Bias

기존 ViT에서 사용하던 Absolute Bias는 성능 저하를 나타냈다. 따라서, Swin Transformer에서는 Relative Position Bias를 활용하여 Attention을 계산(scaled dot product 수행)한다.

### Swin Transformer Attention 계산 수식

1. Self-Attention

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+B\right)V
$$

- \( Q \): Query
- \( K \): Key
- \( V \): Value
- \( d_k \): Key의 차원 수
- \( B \): Relative Position Bias

### Relative Position Bias 계산 예시
![Relative Position Bias 예시](relative_position_bias_example.png)

---

## 7. Swin Transformer 구조

![Swin Transformer 구조](swin_transformer_structure.png)

### ① Patch Partition

- 입력 이미지 $(H \times W \times 3)$를 $4 \times 4$ 크기의 패치로 나눈다.  
- 각 패치를 하나의 벡터로 표현한다.  
- 패치 수는 $\frac{H}{4} \times \frac{W}{4}$가 되며, 각 패치의 벡터 차원은 $48 \, (4 \times 4 \times 3)$이다.

---

### ② Linear Embedding

- 패치를 선형 변환(Dense layer)을 통해 차원을 $C$로 변환한다.  
  (예: $192 \rightarrow 96$로 감소)  
- 변환 후 각 패치의 크기는 $\frac{H}{4} \times \frac{W}{4} \times C$가 된다.

---

### ③ Stage 1: Swin Transformer Block

- **입력 크기**: $\frac{H}{4} \times \frac{W}{4} \times C$  
- Swin Transformer Block이 **2번 반복**됨.  
- 블록 내에서 **W-MSA**와 **Shifted Window Attention (SW-MSA)**를 통해 로컬 및 글로벌 정보를 학습.  
- **결과**: 출력 크기는 $\frac{H}{4} \times \frac{W}{4} \times C$로 유지.

---

### ④ Patch Merging

- Stage 1의 출력을 받아 해상도를 절반으로 줄이고, 채널 수를 $2C$로 증가시킨다.  
- **출력 크기**: $\frac{H}{8} \times \frac{W}{8} \times 2C$

---

### ⑤ Stage 2: Swin Transformer Block

- **입력 크기**: $\frac{H}{8} \times \frac{W}{8} \times 2C$  
- Swin Transformer Block이 **2번 반복**됨.  
- 블록 내에서 **W-MSA**와 **Shifted Window Attention (SW-MSA)**를 통해 로컬 및 글로벌 정보를 학습.  
- **결과**: 출력 크기는 $\frac{H}{8} \times \frac{W}{8} \times 2C$로 유지.

---

### ⑥ Patch Merging

- Stage 2의 출력을 받아 해상도를 절반으로 줄이고, 채널 수를 $4C$로 증가시킨다.  
- **출력 크기**: $\frac{H}{16} \times \frac{W}{16} \times 4C$

---

### ⑦ Stage 3: Swin Transformer Block

- **입력 크기**: $\frac{H}{16} \times \frac{W}{16} \times 4C$
- Swin Transformer Block이 **2번 반복**됨.  
  - **학습 방식**:
    - **W-MSA** (Window-based Multi-Head Self Attention)
    - **Shifted Window Attention (SW-MSA)**
  - **목표**: 로컬 및 글로벌 정보를 학습
- **결과**: 출력 크기는 $\frac{H}{16} \times \frac{W}{16} \times 4C$로 유지.

---

### ⑧ Patch Merging

- **입력**: Stage 3의 출력
- **변경 사항**:
  - 해상도를 절반으로 줄임
  - 채널 수를 $8C$로 증가시킴
- **출력 크기**: $\frac{H}{32} \times \frac{W}{32} \times 8C$

---

### ⑨ Stage 4: Swin Transformer Block

- **입력 크기**: $\frac{H}{32} \times \frac{W}{32} \times 8C$
- Swin Transformer Block이 **2번 반복**됨.  
  - **학습 방식**:
    - **W-MSA** (Window-based Multi-Head Self Attention)
    - **Shifted Window Attention (SW-MSA)**
  - **목표**: 로컬 및 글로벌 정보를 학습.

