#STEP 1
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

#STEP 2
# classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
tokenizer = AutoTokenizer.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
model = AutoModelForTokenClassification.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
ner = pipeline("ner", model=model, tokenizer=tokenizer)

#STEP 3
# text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
text = "서울역으로 안내해줘"

#STEP 4
# result = classifier(text)
result = ner(text)

#STEP 5
print(result)#token classification은 각 단어를 나눠 클래스로 분류해줌