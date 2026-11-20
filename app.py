import os
from flask import Flask, render_template, request, jsonify, send_file
from huggingface_hub import InferenceClient
from io import BytesIO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    story = (data.get("story") or "").strip()
    style = (data.get("style") or "3D cartoon").strip()
    language = (data.get("language") or "Bangla").strip()

    if len(story) < 8:
        return jsonify(error="একটু বিস্তারিত গল্প লিখুন (কমপক্ষে ৮ অক্ষর)।"), 400
    if len(story) > 1200:
        return jsonify(error="গল্পটি ১২০০ অক্ষরের মধ্যে রাখুন।"), 400

    token = os.getenv("HF_TOKEN")
    if not token:
        return jsonify(error="AI ভিডিও সার্ভিস এখনো যুক্ত হয়নি। মালিককে সার্ভারে HF_TOKEN সেট করতে হবে।"), 503

    prompt = (
        f"Create a short animated {style} cartoon video. "
        f"Story: {story}. Visual style: colorful, family-friendly, "
        f"clear character animation, cinematic composition. Language context: {language}."
    )

    try:
        client = InferenceClient(provider="fal-ai", api_key=token)
        video_bytes = client.text_to_video(
            prompt,
            model="Wan-AI/Wan2.2-TI2V-5B",
            num_frames=49
        )
        return send_file(
            BytesIO(video_bytes),
            mimetype="video/mp4",
            as_attachment=False,
            download_name="rubel-ai-cartoon.mp4"
        )
    except Exception as exc:
        # Do not expose credentials or internal server details to visitors.
        app.logger.exception("Video generation failed")
        return jsonify(error="ভিডিও তৈরি করা যায়নি। সার্ভিসের ক্রেডিট, মডেল অ্যাক্সেস ও সেটিংস পরীক্ষা করুন।"), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
