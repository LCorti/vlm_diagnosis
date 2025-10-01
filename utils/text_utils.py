import string
from word2num import Word2Num

MC_DATASETS = ["seed", "vqav2"]
SPECIAL_CHARS = [":", "."]


def fix_word_numbers(parser, x):
    try:
        res = parser.parse(x)
        if res is not None:
            return int(res)
        else:
            return x
    except Exception as _:
        return x


def remove_surrounding_whitespaces(resp: str) -> str:
    # Clear preceding and trailing spaces, new lines, etc.
    clean_resp = "".join(resp.splitlines())
    # Remove unicode characters (e.g., zero-width spaces and emojis)
    clean_resp = (clean_resp.encode("ascii", "ignore")).decode("utf-8")
    clean_resp = clean_resp.rstrip().lstrip()
    return clean_resp


def remove_puncuation(resp: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return resp.translate(translator).strip()


def clean_response(
    resp: str, ds_name: str, open_ended: bool = False, mc_options: dict = None
) -> str:
    clean_resp = remove_surrounding_whitespaces(resp)
    if len(clean_resp) == 0:
        return clean_resp

    if open_ended:
        assert ds_name not in MC_DATASETS, (
            "{ds_name} is incompatible with open_ended=True"
        )
        return clean_resp

    # Now we add dataset-specific instructions to clean up model responses
    if ds_name == "seed":
        assert mc_options is not None, "Requires options for multiple-choice questions."
        # If the response is already one of the options {A, B, C, D}, just return it.
        if clean_resp in list(mc_options.keys()):
            return clean_resp

        # If the model responded with the text instead of {A, B, C, D}, get the
        # corresponding letter.
        if clean_resp in list(mc_options.values()):
            ref_letter = next(
                k for k, v in mc_options.items() if v.lower() == clean_resp.lower()
            ).upper()
            return ref_letter

        # If model responds with something as <text> <option>: <answer> <text>,
        # return the corresponding letter from mc_options.
        for k, v in mc_options.items():
            for char in SPECIAL_CHARS:
                str_lookup = f"{k}{char} {v}"
                if str_lookup.lower() in clean_resp.lower():
                    return k.upper()

        # Sometimes, the response is at the very end. Remove punctuation in the string
        # and return the last character to be compared with the ground truth answer.
        return remove_puncuation(clean_resp)[-1].upper()

        # Maybe create an else branch to return None??

    if ds_name == "vqav2":
        # Here, answers are either a single word or a number. Always lowercase and
        # w/o punctuation. Try first converting the model response to a number if it
        # is one spelled out.
        w2n = Word2Num(fuzzy_threshold=100)
        clean_resp = remove_puncuation(clean_resp).lower()  # Always fix format
        # Start looking for possible numbers in the response. If none are found, just
        # return the clean string.
        words = clean_resp.split(" ")
        if len(words) == 1:
            clean_resp = fix_word_numbers(w2n, clean_resp)
            return str(clean_resp)

        for w in words:
            try:
                # If the number is directly contained, return that.
                return str(int(w))
            except Exception as _:
                # If there is no number, check individual words.
                clean_w = fix_word_numbers(w2n, w)
                if isinstance(clean_w, int):
                    return str(clean_w)
        return clean_resp
