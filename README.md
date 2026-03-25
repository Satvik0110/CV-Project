# OCR Preprocessing Pipeline for Degraded Document Images

A research pipeline that improves [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) accuracy on degraded document images using adaptive image preprocessing. Built and evaluated on a synthetic paragraph dataset with controlled distortion conditions.

--- 

## Overview

Real-world scanned documents suffer from blur, skew, and contrast degradation. This project generates a synthetic paragraph dataset, applies controlled distortions at varying severities, and measures how much an adaptive preprocessing pipeline can recover OCR accuracy compared to a minimal baseline — evaluated using CER (Character Error Rate) and WER (Word Error Rate).

---

## Dataset

300 synthetic paragraph images are generated programmatically. Each image contains 3–6 lines of mixed word and number tokens rendered with randomised fonts (20–28pt) on a paper-textured near-white background. Ground truth is stored as normalised, space-separated lowercase text for fair metric comparison.

---

## Distortion Types

Three distortion types are applied at three severity levels, plus a combined condition (all three applied sequentially):

| Type     | Mild               | Medium             | Severe             |
|----------|--------------------|--------------------|--------------------|
| Blur     | Gaussian 5x5       | Gaussian 9x9       | Gaussian 15x15     |
| Skew     | 2 degrees          | 5 degrees          | 10 degrees         |
| Contrast | alpha=0.70 beta=18 | alpha=0.48 beta=12 | alpha=0.28 beta=6  |

---

## Preprocessing Pipeline

The adaptive pipeline applies the following steps in order:

1. **Grayscale conversion**
2. **Upscale by character height** — estimates character height via connected component analysis and scales the image so characters are approximately 32px tall. This is critical for paragraphs: scaling by total image height (as done for single-word images) would shrink a multi-line image rather than enlarge it.
3. **Quality gate** — if Laplacian variance > 350 and contrast std > 38, the image is already clean and skips the full pipeline, preventing any degradation of good input.
4. **Contrast enhancement** — 2nd–98th percentile linear stretch. Preferred over CLAHE because CLAHE amplifies local texture noise in the large uniform background regions typical of document images.
5. **Inversion check** — applied after contrast enhancement; a pre-contrast image with severe compression can have a misleading mean that triggers false inversion.
6. **Unsharp masking** — sigma=2.0, strength=2.5. A second pass (sigma=1.5, strength=1.5) is applied if Laplacian variance remains below 40 after the first.
7. **Morphological close** — horizontal kernel with width proportional to character height (`max(3, char_height // 10)`). Strictly horizontal to avoid merging adjacent text lines.
8. **Smart binarization** — Otsu if contrast std > 30, otherwise adaptive threshold with block size scaled to 2x character height. A fixed small block (e.g. 15px) on a paragraph image binarizes within individual characters rather than separating text from background.
9. **Deskew** — projection profile method: rotates the binary image through candidate angles and picks the angle that maximises row-sum variance. More robust than minAreaRect for paragraphs because variable line lengths bias the minimum bounding rectangle.
10. **Padding** — 15px white border added before passing to Tesseract.

The baseline pipeline applies only upscaling, inversion check, and padding — so all deltas are attributable purely to the adaptive steps.

---

## OCR Configuration

Tesseract is run with `--psm 6 --oem 3` (uniform block of text, LSTM engine). PSM 3 (fully automatic) was rejected because it runs orientation/script detection which is unreliable on short synthetic regions and adds approximately 30% runtime overhead with no accuracy benefit on clean synthetic paragraphs.

---

## Metrics

**Character Error Rate (CER)** — character-level Levenshtein distance divided by ground truth length.

**Word Error Rate (WER)** — word-token edit distance (substitutions + deletions + insertions) divided by reference word count, computed via dynamic programming. This correctly penalises word boundary errors, unlike character-level distance on the joined string.

Both predictions and ground truth pass through the same normalisation before any metric is computed: newlines and tabs are replaced with spaces, consecutive spaces are collapsed, and the string is lowercased and stripped.

---

## Results

Clean baseline: CER = 0.0021, WER = 0.0110

| Distortion | Base CER | Pre CER | Delta CER | Base WER | Pre WER | Delta WER | CER Improvement |
|------------|----------|---------|-----------|----------|---------|-----------|-----------------|
| Blur       | 0.1961   | 0.1531  | -0.0429   | 0.4465   | 0.2366  | -0.2099   | 21.9%           |
| Skew       | 0.1551   | 0.1221  | -0.0329   | 0.3730   | 0.1998  | -0.1733   | 21.2%           |
| Contrast   | 0.0179   | 0.0013  | -0.0166   | 0.0321   | 0.0055  | -0.0265   | 92.9%           |
| Combined   | 0.9817   | 0.3846  | -0.5970   | 1.0000   | 0.5601  | -0.4399   | 60.8%           |

Mean delta CER across all conditions: -0.2528  
Mean delta WER across all conditions: -0.2703

---

## Setup

```bash
# System dependencies
apt-get install -y tesseract-ocr fonts-liberation fonts-freefont-ttf \
                   fonts-urw-base35 fonts-dejavu

# Python dependencies
pip install pytesseract opencv-python python-Levenshtein \
            tqdm matplotlib Pillow gradio
```

Open `CVProj_v2.ipynb` in Google Colab and run all cells sequentially.