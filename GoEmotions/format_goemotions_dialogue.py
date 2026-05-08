from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd


RESPONSE_TEMPLATES = {
    "loss": [
        "That sounds really heavy, and it makes sense that this is hurting so much. You deserve some gentleness while you carry it.",
        "There is a lot of sadness in that, and it feels hard for a reason. I am holding this with you for a moment.",
        "That kind of loss can sit deeply and quietly at the same time. It is understandable that your heart feels worn down.",
        "This sounds painful in a tender way, and it is not small. Anyone in that place would need some care.",
        "There is a real ache behind that. You do not have to minimize how hard this feels.",
    ],
    "anger": [
        "That sounds genuinely frustrating, and your reaction makes sense. There is a lot of intensity in what you are carrying right now.",
        "You sound fed up, and it is understandable that this hit a nerve. That kind of strain can be exhausting.",
        "There is a lot of irritation in that, and it does not seem to have come from nowhere. Anyone pushed like that could feel wound up.",
        "That frustration feels very real. It makes sense that you would feel tense and unsettled by it.",
        "You sound angry for a reason, and that feeling deserves to be acknowledged. It is a lot to hold all at once.",
    ],
    "fear": [
        "That sounds scary, and it makes sense that your body is on alert. You do not have to face that feeling alone in this moment.",
        "There is a lot of fear in that, and it feels understandable. Even naming it shows you are trying to get through something difficult.",
        "That kind of worry can feel overwhelming very quickly. It makes sense that you are feeling shaken by it.",
        "You sound really uneasy, and that reaction fits what you are describing. This seems like a hard moment to sit with.",
        "That fear feels very present, and it is understandable that this would rattle you. I am meeting this with care.",
    ],
    "self-conscious": [
        "That sounds vulnerable, and it makes sense that this would leave you feeling exposed. Your feelings here are valid and very human.",
        "There is a lot of self-doubt in that moment, and it feels understandable. You still deserve kindness while you work through it.",
        "That sounds uncomfortable in a deeply personal way. It makes sense that you would feel sensitive about it.",
        "You seem hard on yourself here, and this clearly matters to you. That does not make you any less worthy of understanding.",
        "That kind of embarrassment or shame can feel very isolating. It is okay that this landed heavily for you.",
    ],
    "disgust": [
        "That sounds really unpleasant, and your discomfort makes sense. Anyone faced with that could feel put off.",
        "There is a strong sense of discomfort in that, and it feels understandable. It is okay to react to something that unsettling.",
        "That seems deeply off-putting, and it makes sense that it stayed with you. Your reaction does not feel exaggerated.",
        "That sounds unsettling in a very immediate way. It is understandable that you would want distance from it.",
        "There is clear discomfort in that response, and it feels warranted. Some things just land as deeply unpleasant.",
    ],
}

LABEL_NORMALIZATION = {
    "loss": "loss",
    "anger": "anger",
    "fear": "fear",
    "self-conscious": "self-conscious",
    "self conscious": "self-conscious",
    "self_conscious": "self-conscious",
    "disgust": "disgust",
}


def _normalize_label(label: str) -> str:
    normalized = str(label).strip().lower()
    if normalized not in LABEL_NORMALIZATION:
        raise ValueError(f"Unsupported emotion label: {label}")
    return LABEL_NORMALIZATION[normalized]


def generate_empathetic_response(text: str, emotion: str) -> str:
    """Return a deterministic, emotion-aware empathetic response."""
    normalized_emotion = _normalize_label(emotion)
    templates = RESPONSE_TEMPLATES[normalized_emotion]
    text_key = hashlib.md5(str(text).strip().encode("utf-8")).hexdigest()
    template_index = int(text_key, 16) % len(templates)
    return templates[template_index]


def build_formatted_text(row: pd.Series) -> str:
    emotion = str(row["level_2"]).strip()
    user_text = str(row["text"]).strip()
    assistant_text = generate_empathetic_response(user_text, emotion)
    return (
        f"Emotion: {emotion}\n"
        f"User: {user_text}\n"
        f"Assistant: {assistant_text}"
    )


def convert_goemotions_to_dialogue(
    input_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """Read a GoEmotions CSV, add `formatted_text`, and save a new CSV."""
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)

    required_columns = {"text", "level_2"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    df["formatted_text"] = df.apply(build_formatted_text, axis=1)
    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    source_path = Path("GoEmotions/goemotions_multilevel_negative.csv")
    target_path = Path("GoEmotions/goemotions_multilevel_negative_dialogue.csv")
    converted_df = convert_goemotions_to_dialogue(source_path, target_path)
    print(
        f"Saved {len(converted_df)} rows with formatted dialogue text to "
        f"{target_path}"
    )
