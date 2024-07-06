from fastapi import FastAPI, Form
#STEP 1
from sentence_transformers import SentenceTransformer

#STEP 2
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


app = FastAPI()


@app.post("/textclassification/")
async def textclassification(text1: str = Form(),text2: str = Form()):

    #STEP 3
    # sentence1 = "튀어"
    # sentence2 = "도망가"

    #STEP 4
    embedding1 = model.encode(text1)
    embedding2 = model.encode(text2)
    print(embedding1.shape)
    # [3, 384]

    #STEP 5
    similarities = model.similarity(embedding1, embedding2)
    print(similarities)
    # tensor([[1.0000, 0.6660, 0.1046],
    #         [0.6660, 1.0000, 0.1411],
    #         [0.1046, 0.1411, 1.0000]])

    return {"result": similarities.item()}