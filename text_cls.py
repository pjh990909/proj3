

#STEP 1
from transformers import pipeline

#STEP 2
# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

#STEP 3
# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
# text = "현대제철, 지난해 영업익 3,313억원···전년比 67.7% 감소"
text = "샤오미의 폴더블 폰의 점유율이 삼성전자 보다 높아졌다." #비정형 데이터 문장을 보고 해석이 달라질 수 있음


#STEP 4
result = classifier(text)

#STEP 5
print(result)#text classification은 주어진 텍스트를 사전 정의된 카테고리나 클래스 중 하나에 할당하는 자연어 처리 작업