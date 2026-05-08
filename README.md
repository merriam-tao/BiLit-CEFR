# BiLit-CEFR Gradio Demo

A bilingual (English + Chinese) literary difficulty demo for CEFR-style labels (`A`, `B`, `C`).

## Demo Features

- Input box: `Enter a literary passage...`
- Input limit: up to 100 words/tokens (English or Chinese)
- Output includes:
  - Predicted CEFR difficulty
  - Detected language
  - Confidence score
  - Similar Chinese `C`-level passage (cross-lingual retrieval)
- Built-in test examples: 3 English + 3 Chinese passages

## Project Structure

- `app.py` - Gradio app entry point
- `requirements.txt` - Python dependencies
- `comparison_outputs/best_model.joblib` - selected best classifier artifact
- `en_test.jsonl`, `zh_test.jsonl` - test data used for examples and retrieval

## Deploy on Hugging Face Spaces

1. Create a new Space on Hugging Face.
2. Choose:
   - **SDK**: Gradio
   - **Hardware**: CPU Basic
3. Upload/push these files to the Space repo:
   - `app.py`
   - `requirements.txt`
   - `comparison_outputs/best_model.joblib`
   - `en_test.jsonl`
   - `zh_test.jsonl`
4. Commit and wait for build to finish.
5. Open the Space URL and test the demo.

## Demo Screenshot

<!-- Replace with an actual screenshot path or image link -->
![Demo Screenshot Placeholder](./assets/demo-screenshot-placeholder.png)

## Notes

- If model download is slow, set `HF_TOKEN` in Space secrets for better Hugging Face Hub rate limits.
- CPU Basic is sufficient for this demo, but first inference may be slightly slower due to model loading.

## Publish to GitHub

I cannot link your GitHub account from here. Follow **[GITHUB_PUBLISH.md](./GITHUB_PUBLISH.md)** on your machine: install Git, run `git init`, `git add`, `git commit`, then `gh repo create ...` or add `origin` and `git push`.
