from surya.models import load_predictors

# Load all predictors
predictors = load_predictors(
    detection=True,
    recognition=True,
    layout=True
)

# -------- SURYA OCR + TABLE DETECTION --------
def surya_ocr(images):

    # Run OCR with layout detection
    recognitions = predictors["recognition"](
        images=images,
        task_names=["ocr_with_boxes"] * len(images),
        det_predictor=predictors["detection"]
    )

    # Layout detection (tables, paragraphs, figures)
    layouts = predictors["layout"](images)

    results = []

    for page_index, rec in enumerate(recognitions):

        page_result = {
            "text_lines": [],
            "tables": [],
            "boxes": []
        }

        # -------- Extract Text + Bounding Boxes --------
        for line in rec.text_lines:

            text_data = {
                "text": line.text,
                "bbox": line.bbox   # [x1, y1, x2, y2]
            }

            page_result["text_lines"].append(text_data)
            page_result["boxes"].append(line.bbox)

        # -------- Extract Tables from Layout --------
        for element in layouts[page_index].elements:

            if element.label == "table":

                table_data = {
                    "bbox": element.bbox
                }

                page_result["tables"].append(table_data)

        results.append(page_result)

    return results
