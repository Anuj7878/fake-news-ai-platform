from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

classifier = pipeline(
    "text-classification",
    model="distilbert-base-multilingual-cased",
    top_k=None
)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    note = None

    if request.method == "POST":
        text = request.form["news"]
        outputs = classifier(text)[0]

        real_score = 0
        fake_score = 0

        for o in outputs:
            if o["label"] == "LABEL_0":
                real_score = o["score"]
            elif o["label"] == "LABEL_1":
                fake_score = o["score"]

        if fake_score >= 0.60:
            result = "LIKELY FAKE NEWS"
            confidence = round(fake_score * 100, 2)
            note = "High risk – misinformation likely"
        elif fake_score >= 0.40:
            result = "UNCERTAIN"
            confidence = round(fake_score * 100, 2)
            note = "Needs manual verification"
        else:
            result = "LIKELY REAL NEWS"
            confidence = round(real_score * 100, 2)
            note = "Low risk – appears reliable"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        note=note
    )

if __name__ == "__main__":
    app.run(debug=True)
