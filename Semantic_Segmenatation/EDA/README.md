# 탐색적 데이터 분석 (EDA)

## 데이터 분포(Data Distribution)

#### class 별 annotation 개수 분포

![sample]()

: class 별 annotation 개수는 Paper, Plastic Bag이 9311, 7643개로 비교적 많았고, Battery, Clothing이 63, 177개로 비교적 적은 수를 보였다.

#### image 별 annotation 개수 분포

![sample]()

: image 별 annotation 개수는 1~12개 범위의 image가 전체의 77.4%를 차지했다. annotation이 가장 많은 이미지는 70개, 가장 적은 image는 0개의 annotation을 가졌다.

#### 라벨링 규칙(Labeling Rule)
- 플라스틱 병에 붙어있는 비닐은 플라스틱 병과 함께 Plastic으로 분류한다.
- Plastic Bag 내부의 쓰레기는 Plastic Bag으로 분류한다.
- Plastic Bag 외부의 쓰레기는 따로 분류한다.
- 병뚜껑은 따로 Metal로 분류하지 않는다.
- 종이 박스에 붙은 쓰레기는 분류 규칙이 일관되지 않았다.