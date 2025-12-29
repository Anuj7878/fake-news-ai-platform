from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="distilbert-base-multilingual-cased",
    top_k=None
)

print("Fake News Detection Ready!")
print("Type 'exit' to quit\n")

while True:
    text = input("Enter news text: ")
    if text.lower() == "exit":
        break

    results = classifier(text)[0]

    # LABEL_0 = Real, LABEL_1 = Fake
    real_score = 0
    fake_score = 0

    for r in results:
        if r["label"] == "LABEL_0":
            real_score = r["score"]
        elif r["label"] == "LABEL_1":
            fake_score = r["score"]

    if fake_score > real_score:
        print(f"Result: FAKE NEWS ({fake_score*100:.2f}% confidence)")
    else:
        print(f"Result: REAL NEWS ({real_score*100:.2f}% confidence)")

    print("-" * 40)
