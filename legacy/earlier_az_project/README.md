# BLEnD Azerbaijani Poster Project

This is the final, time-bounded implementation plan for the poster.

## 1. Download the official BLEnD repo
```bash
git clone https://github.com/nlee0212/BLEnD.git
```
Keep the repository as-is. We use the original `data/` directory, not `data_SemEval/`.

## 2. Install dependencies
For the experiment:
```bash
pip install openai pandas scipy statsmodels
```
For official BLEnD scoring, follow the official repository requirements and Azerbaijani stemmer setup. BLEnD's evaluator uses language-specific lemmatization/stemming for short-answer scoring.

## 3. Prepare the 300-question sample
From this project folder:
```bash
python prepare_sample.py
```
Set `BLEND_DIR` first if the repo is not at `../BLEnD`.

Outputs:
- `work/Azerbaijan_sample_300.csv`
- `work/Azerbaijan_300_data.json`
- `work/Azerbaijan_inst4_prompt.csv`

## 4. Run the two-language experiment
Set:
```bash
# Windows PowerShell example
$env:OPENAI_API_KEY="YOUR_KEY"
$env:MODEL_A="YOUR_MODEL_A"
$env:MODEL_B="YOUR_MODEL_B"
```
Then:
```bash
python run_inference.py
```
The script is resumable and writes one CSV per model/language.

## 5. Score using the official BLEnD evaluator
Run BLEnD's `evaluation/evaluate.py` for:
- country: `Azerbaijan`
- language: `Azerbaijani`
- language: `English`
- prompt: `inst-4`
- annotation file: the project-generated `Azerbaijan_300_data.json`
- response directory: this project's `work/responses`

Use the official scoring code rather than inventing a new correctness metric.

## 6. Analyze paired consistency
After the official evaluator creates the two `*_response_score.csv` files:
```bash
python analyze_pairs.py
```
Outputs are in `work/analysis/`.

## 7. What goes on the poster
Only four things are essential:
1. English vs. Azerbaijani accuracy bar chart.
2. Paired outcome chart (both correct / English-only / Azerbaijani-only / both wrong).
3. McNemar p-value + accuracy gap.
4. 3–5 clear discordant examples.

Everything else is optional.

## Important methodological choice
Use BLEnD's gold annotations as the correctness standard. Do not manually judge cultural correctness from personal intuition. For qualitative examples, only label errors when the evidence is clear.
