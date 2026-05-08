from __future__ import annotations

from pathlib import Path
import hashlib
import re

import pandas as pd


DATASET_PATH = Path("GoEmotions/goemotions_multilevel_negative_dialogue.csv")

SUPPORTED_EMOTIONS = {
    "loss": "loss",
    "anger": "anger",
    "fear": "fear",
    "self-conscious": "self-conscious",
    "self conscious": "self-conscious",
    "self_conscious": "self-conscious",
    "disgust": "disgust",
}

FAMILY_WORDS = {
    "mom",
    "mother",
    "dad",
    "father",
    "parents",
    "parent",
    "sister",
    "brother",
    "wife",
    "husband",
    "partner",
    "girlfriend",
    "boyfriend",
    "family",
}

APPEARANCE_WORDS = {
    "hair",
    "face",
    "body",
    "weight",
    "skin",
    "appearance",
    "looks",
    "look",
}

WORK_WORDS = {"work", "job", "boss", "manager", "office", "coworker", "coworkers"}
SCHOOL_WORDS = {"school", "class", "teacher", "college", "exam", "test", "grades"}
ARGUMENT_WORDS = {"argument", "argue", "arguing", "fight", "fighting", "conflict"}
YELLING_WORDS = {"yell", "yelled", "yelling", "shout", "shouted"}
SOCIAL_PRESSURE_WORDS = {
    "haters",
    "hater",
    "people",
    "they",
    "them",
    "everyone",
    "nobody",
    "comment",
    "comments",
}

RESPONSE_PARTS = {
    "loss": {
        "openers": [
            "That carries a quiet kind of hurt",
            "That sounds deeply tender and painful",
            "There is a real ache in that",
            "This feels like a heavy disappointment to sit with",
            "That lands with a lot of sadness",
            "You sound weighed down by this",
            "That sounds like a painful thing to keep holding",
            "There is a lot of tenderness and hurt wrapped up in that",
            "That seems to sit in a very sore place",
            "This sounds quietly heartbreaking",
        ],
        "closers": [
            "You do not have to shrink how hard that feels.",
            "It makes sense that your heart feels tired around this.",
            "Anyone in that spot would need some gentleness.",
            "That kind of hurt can linger more than people realize.",
            "It is okay that this is affecting you.",
            "You deserve some softness while this settles.",
            "It makes sense that this would stay close to the surface.",
            "Anyone carrying that much sadness would need room for it.",
            "That kind of disappointment can settle in deeply.",
            "You are not making too much of something that clearly hurts.",
        ],
    },
    "anger": {
        "openers": [
            "That sounds genuinely frustrating",
            "There is a lot of heat and strain in that",
            "You sound pushed past your limit here",
            "That would get under a lot of people's skin",
            "This comes across as sharp and exhausting",
            "You sound understandably fed up",
            "That feels aggravating in a very real way",
            "You sound tired of being pushed like this",
            "That has the feel of something that kept needling at you",
            "This sounds like the kind of thing that would wear anyone down",
        ],
        "closers": [
            "That kind of pressure can leave anyone on edge.",
            "It makes sense that your nerves feel rubbed raw.",
            "Anyone dealing with that could feel wound up.",
            "That reaction does not sound exaggerated.",
            "It is a lot to carry when things feel this charged.",
            "That sort of tension can build quickly.",
            "It makes sense that this left you feeling so tense.",
            "Anyone being handled that way could feel their patience thinning.",
            "That level of irritation can build before you even notice it.",
            "You are not overreacting to something that feels this abrasive.",
        ],
    },
    "fear": {
        "openers": [
            "That sounds really unsettling",
            "There is a lot of fear sitting underneath that",
            "You sound on edge in a very real way",
            "That carries the feeling of bracing for something hard",
            "This sounds scary to hold by yourself",
            "Your worry feels very present here",
            "That has the feel of a very anxious moment",
            "You sound like you are carrying a lot of dread",
            "That sounds difficult to sit with when your mind is already racing",
            "This feels like the kind of fear that stays close to your skin",
        ],
        "closers": [
            "It makes sense that your body would stay on alert.",
            "Anyone in that moment could feel shaken.",
            "That kind of uncertainty can rattle you quickly.",
            "It is understandable that this feels hard to settle.",
            "That level of worry can feel overwhelming fast.",
            "You do not sound dramatic for reacting this way.",
            "That kind of fear can make everything feel louder for a while.",
            "Anyone facing that uncertainty could feel unsteady.",
            "It makes sense that your nerves are having a hard time easing.",
            "That reaction fits a moment that feels this precarious.",
        ],
    },
    "self-conscious": {
        "openers": [
            "That sounds painfully exposing",
            "There is a lot of vulnerability in that moment",
            "You sound hard on yourself in a way that really hurts",
            "That kind of moment can leave anyone feeling small",
            "This sounds deeply personal and uncomfortable",
            "There is a tender kind of embarrassment in that",
            "That sounds like a moment that would sting afterward",
            "You seem to be carrying a lot of self-consciousness around this",
            "That has the feel of something that would make anyone want to hide a little",
            "This sounds like it touched a very vulnerable spot",
        ],
        "closers": [
            "It makes sense that this would stay with you.",
            "Anyone feeling that exposed could struggle with it.",
            "That does not make you any less worthy of kindness.",
            "It is understandable that this landed so heavily.",
            "That kind of self-doubt can feel very isolating.",
            "You deserve understanding while you sit with that feeling.",
            "It makes sense that this would echo in your mind afterward.",
            "Anyone in a moment like that could feel deeply uncomfortable.",
            "That kind of sting can make you turn inward quickly.",
            "You are still deserving of gentleness while this feels raw.",
        ],
    },
    "disgust": {
        "openers": [
            "That sounds really off-putting",
            "There is a strong sense of discomfort in that",
            "That lands as genuinely unpleasant",
            "This sounds unsettling in an immediate way",
            "You sound deeply put off by it",
            "That kind of reaction makes sense",
            "That seems viscerally unpleasant",
            "This has the feel of something that would make anyone recoil",
            "That sounds hard to brush off once you have seen it",
            "You sound understandably unsettled by how unpleasant that was",
        ],
        "closers": [
            "Some things just feel wrong in a way that sticks.",
            "Anyone near that could recoil from it.",
            "It is understandable that this stayed with you.",
            "That reaction sounds warranted, not overdone.",
            "Some experiences just leave a strong sense of aversion.",
            "It makes sense that you would want distance from that.",
            "That kind of discomfort can linger longer than expected.",
            "It makes sense that this left such a strong reaction behind.",
            "Some things are just hard to be calm around once they hit that way.",
            "Anyone dealing with that could feel immediately put off.",
        ],
    },
}


def normalize_emotion(emotion: str) -> str:
    normalized = str(emotion).strip().lower()
    if normalized not in SUPPORTED_EMOTIONS:
        raise ValueError(f"Unsupported emotion label: {emotion}")
    return SUPPORTED_EMOTIONS[normalized]


def clean_user_text(text: str) -> str:
    cleaned = str(text).replace('"', "")
    return re.sub(r"\s+", " ", cleaned).strip()


def convert_possessive_pronoun(match_text: str) -> str:
    mapping = {
        "my": "your",
        "our": "your",
        "their": "their",
        "his": "his",
        "her": "her",
        "the": "the",
        "this": "this",
        "that": "that",
    }
    words = match_text.split(maxsplit=1)
    if len(words) == 1:
        return match_text
    owner, remainder = words
    return f"{mapping.get(owner, owner)} {remainder}"


def to_possessive(noun_phrase: str) -> str:
    return f"{noun_phrase}'" if noun_phrase.endswith("s") else f"{noun_phrase}'s"


def extract_appearance_phrase(text: str) -> str | None:
    pattern = (
        r"\b(my|our|their|his|her|your|the|this|that)\s+"
        r"(hair|face|body|weight|skin|appearance|looks|look)\b"
    )
    match = re.search(pattern, text.lower())
    if not match:
        return None
    return convert_possessive_pronoun(match.group(0))


def build_context_clause(text: str, emotion: str) -> str:
    lowered = text.lower()
    appearance_phrase = extract_appearance_phrase(text)

    family_match = next((word for word in FAMILY_WORDS if word in lowered.split()), None)
    if family_match and any(word in lowered.split() for word in YELLING_WORDS):
        target = "your family" if family_match == "family" else f"your {family_match}"
        return pick_context(
            [
                f"when you are bracing for {to_possessive(target)} reaction",
                f"when you are waiting for {target} to react",
                f"when you are tense about how {target} might respond",
            ],
            text,
            "family_yelling",
        )

    if family_match:
        target = "your family" if family_match == "family" else f"your {family_match}"
        return pick_context(
            [
                f"when this is happening with {target}",
                f"when {target} are right in the middle of it",
                f"when so much of this sits close to {target}",
            ],
            text,
            "family",
        )

    if any(word in lowered.split() for word in ARGUMENT_WORDS):
        return pick_context(
            [
                "when that argument is still sitting with you",
                "when the tension from that argument has not let up",
                "when that conflict is still close to the surface",
            ],
            text,
            "argument",
        )

    if appearance_phrase:
        return pick_context(
            [
                f"when you are worrying about {appearance_phrase}",
                f"when your attention keeps going back to {appearance_phrase}",
                f"when {appearance_phrase} feels hard to stop thinking about",
            ],
            text,
            "appearance_phrase",
        )

    if any(word in lowered.split() for word in APPEARANCE_WORDS):
        return pick_context(
            [
                "when appearance suddenly feels very exposed",
                "when how things look starts feeling painfully personal",
                "when appearance becomes hard to stop noticing",
            ],
            text,
            "appearance_generic",
        )

    if any(word in lowered.split() for word in WORK_WORDS):
        return pick_context(
            [
                "when work already feels tense",
                "when pressure around work is already building",
                "when the atmosphere around work feels strained",
            ],
            text,
            "work",
        )

    if any(word in lowered.split() for word in SCHOOL_WORDS):
        return pick_context(
            [
                "when school pressure is already in the room",
                "when school already feels heavy around you",
                "when that pressure around school is still there",
            ],
            text,
            "school",
        )

    if "calm down" in lowered or "relax" in lowered:
        return pick_context(
            [
                "when someone is brushing aside what you feel",
                "when your feelings are being waved away",
                "when someone is talking past what is bothering you",
            ],
            text,
            "dismissed",
        )

    if any(word in lowered.split() for word in SOCIAL_PRESSURE_WORDS):
        return pick_context(
            [
                "when other people keep pressing on the same sore spot",
                "when the people around you keep making it worse",
                "when outside pressure keeps piling onto the same feeling",
            ],
            text,
            "social_pressure",
        )

    if any(token in lowered for token in ("scared", "afraid", "fear", "worried", "panic", "anxious")):
        return pick_context(
            [
                "when you are bracing for what might happen next",
                "when your mind is already racing ahead",
                "when the uncertainty will not let you settle",
            ],
            text,
            "fear",
        )

    if any(token in lowered for token in ("gross", "disgusting", "nasty", "filthy", "sick", "vomit", "smell")):
        return pick_context(
            [
                "when something feels this unpleasant up close",
                "when the discomfort is this immediate",
                "when being near it feels that unsettling",
            ],
            text,
            "disgust",
        )

    if any(token in lowered for token in ("lost", "gone", "miss", "missing", "left", "thinning", "disappointed")):
        return pick_context(
            [
                "when something important still feels uncertain",
                "when that sense of loss has not settled yet",
                "when what matters still feels out of place",
            ],
            text,
            "loss",
        )

    fallbacks = {
        "loss": [
            "when something important feels hard to make peace with",
            "when a sense of loss keeps sitting in the background",
            "when this still feels tender and unresolved",
            "when something meaningful still feels out of place",
        ],
        "anger": [
            "when people keep rubbing at the same sore spot",
            "when the tension keeps getting pushed higher",
            "when what happened still feels abrasive",
            "when the situation keeps scraping at your patience",
        ],
        "fear": [
            "when your nerves are already stretched thin",
            "when the uncertainty stays right at the front of your mind",
            "when it feels hard to tell what comes next",
            "when your body still feels on alert",
        ],
        "self-conscious": [
            "when you feel exposed in a way that really stings",
            "when the moment keeps replaying in a tender spot",
            "when it is hard not to turn that feeling back on yourself",
            "when you feel seen in a way that is hard to shake",
        ],
        "disgust": [
            "when something feels deeply wrong up close",
            "when the discomfort is hard to get away from",
            "when what happened leaves that kind of immediate aversion",
            "when the whole thing just feels deeply unpleasant",
        ],
    }
    return pick_context(fallbacks[emotion], text, f"fallback_{emotion}")


def extract_assistant_text(formatted_text: str) -> str:
    marker = "Assistant:"
    if marker not in str(formatted_text):
        return ""
    return str(formatted_text).split(marker, 1)[1].strip()


def normalize_opening(response: str, word_count: int = 6) -> str:
    tokens = re.findall(r"[a-z']+", response.lower())
    return " ".join(tokens[:word_count])


def detect_repetition_flags(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    assistants = df["formatted_text"].fillna("").map(extract_assistant_text)
    assistants = assistants.str.replace(r"\s+", " ", regex=True).str.strip()
    exact_counts = assistants.value_counts()
    opening_counts = assistants.map(normalize_opening).value_counts()
    return exact_counts, opening_counts


def pick(options: list[str], key: str, offset: int) -> str:
    digest = hashlib.md5(f"{key}|{offset}".encode("utf-8")).hexdigest()
    return options[int(digest, 16) % len(options)]


def pick_context(options: list[str], text: str, salt: str) -> str:
    digest = hashlib.md5(f"context|{salt}|{text}".encode("utf-8")).hexdigest()
    return options[int(digest, 16) % len(options)]


def generate_refined_response(
    user_text: str,
    emotion: str,
    row_key: str,
    variant_offset: int = 0,
) -> str:
    normalized_emotion = normalize_emotion(emotion)
    context_clause = build_context_clause(user_text, normalized_emotion)
    parts = RESPONSE_PARTS[normalized_emotion]

    opener = pick(parts["openers"], row_key, variant_offset)
    closer = pick(parts["closers"], row_key, variant_offset + 11)
    pattern_index = int(
        hashlib.md5(f"pattern|{row_key}|{variant_offset}".encode("utf-8")).hexdigest(),
        16,
    ) % 4

    responses = [
        f"{opener}, especially {context_clause}. {closer}",
        f"{opener}, and it makes sense that it feels this sharp {context_clause}. {closer}",
        f"{opener}, particularly {context_clause}. {closer}",
        f"{opener}, with a lot of this landing hard {context_clause}. {closer}",
    ]
    return responses[pattern_index]


def build_formatted_text(emotion: str, user_text: str, assistant_text: str) -> str:
    return (
        f"Emotion: {emotion}\n"
        f"User: {user_text}\n"
        f"Assistant: {assistant_text}"
    )


def refine_dialogue_dataset(csv_path: str | Path = DATASET_PATH) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_columns = {"text", "level_2", "formatted_text"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    exact_counts, opening_counts = detect_repetition_flags(df)
    original_text = df["text"].fillna("").astype(str)

    df["text"] = original_text.map(clean_user_text)

    variant_tracker: dict[str, int] = {}
    new_assistant_texts: list[str] = []

    for index, row in df.iterrows():
        emotion = str(row["level_2"]).strip()
        user_text = str(row["text"]).strip()
        row_key = f"{emotion}|{user_text}|{index}"
        attempt = 0

        while True:
            response = generate_refined_response(user_text, emotion, row_key, attempt)
            duplicate_key = f"{emotion}|{response}"
            if duplicate_key not in variant_tracker or attempt >= 80:
                variant_tracker[duplicate_key] = 1
                break
            attempt += 1

        new_assistant_texts.append(response)

    df["formatted_text"] = [
        build_formatted_text(str(emotion).strip(), user_text, assistant_text)
        for emotion, user_text, assistant_text in zip(
            df["level_2"],
            df["text"],
            new_assistant_texts,
        )
    ]

    df.to_csv(csv_path, index=False)
    repeated_exact = int((exact_counts >= 3).sum())
    repeated_openings = int((opening_counts >= 25).sum())
    print(
        "Detected "
        f"{repeated_exact} highly repeated exact replies and "
        f"{repeated_openings} highly repeated openings before rewrite."
    )
    return df


if __name__ == "__main__":
    refined_df = refine_dialogue_dataset(DATASET_PATH)
    print(f"Refined and saved {len(refined_df)} rows to {DATASET_PATH}")
