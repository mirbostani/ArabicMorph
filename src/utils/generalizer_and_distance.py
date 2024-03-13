from typing import Union, Tuple
import Levenshtein

# breaking down the ur and sf and generalizing the stems accordingly
# from thefuzz import process

# it will get the most right "-" in ur example:
# ki-taba=hu , kitabahu  ----> delete , 2 , 2
def get_nearest_prefix_to_stem(ur, sf) -> Union[Tuple[str, int, int], None]:
    changes = Levenshtein.editops(ur, sf)
    prefix_change = None
    for change in changes:
        type_of_change = change[0]
        index_of_src = change[1]
        index_of_dest = change[2]
        if (type_of_change == "delete" or type_of_change == "replace") and ur[index_of_src] == "-":
            prefix_change = (type_of_change, index_of_src, index_of_dest)
    return prefix_change


# it will get the most left "-" in ur example:
# ki-taba=hu , kitabahu  ----> delete , 8 , 8
def get_nearest_suffix_to_stem(ur, sf) -> Union[Tuple[str, int, int], None]:
    changes = Levenshtein.editops(ur, sf)
    suffix_change = None
    for change in changes:
        type_of_change = change[0]
        index_of_src = change[1]
        index_of_dest = change[2]
        if (type_of_change == "delete" or type_of_change == "replace") and ur[index_of_src] == "=":
            suffix_change = (type_of_change, index_of_src, index_of_dest)
            break
    return suffix_change

#it just input a str and replace any chars expect vowels or "#" or "0" to "C"
def stem_generalizer(stem: str) -> str:
    vowels = ["A", "E", "I", "O", "U", "a", "i", "u"]
    new_stem = ""
    for e in stem:
        if e in vowels:
            new_stem = new_stem + e
        elif e == "#" or e == "0":
            new_stem = new_stem + e
        else:
            new_stem = new_stem + "C"
    return new_stem


def ur_stem_generalizer(stem: str) -> str:
    shamsi = [
        "$", # ش 
        "n", # ن
        "l", # ل
        "T", # ت ط
        "t", # ت ط
        "Z", # ز ض ظ ذ
        "z", # ز ض ظ ذ
        "r", # ر
        "D", # د
        "d", # د
        "S", # س ص ث
        "s"  # س ص ث
        ]

    # e.g. UR: "0Allah0" -> SF: "0'allA0"
    if len(stem) >= 4 and stem.startswith("0Al") and stem[3] in shamsi:
        chars = list(stem)
        chars[3] = "S"
        stem = "".join(chars)
    # e.g. UR: "0raqab=a0" -> SF: "0raqaba0"
    elif len(stem) >= 2 and stem[0] == "0" and stem[1] in shamsi:
        chars = list(stem)
        chars[1] = "S"
        stem = "".join(chars)
    elif len(stem) >= 1 and stem[0] in shamsi:
        chars = list(stem)
        chars[0] = "S"
        stem = "".join(chars)

    vowels = ["A", "E", "I", "O", "U", "a", "i", "u"]
    new_stem = ""
    for e in stem:
        if e in vowels:
            new_stem += e
        elif e == "#" or e == "0":
            new_stem += e
        elif e == "S":
            new_stem += e
        else:
            new_stem += "C"

    return new_stem


def sf_stem_generalizer(stem: str) -> str:
    shamsi = [
        "$", # ش 
        "n", # ن
        "l", # ل
        "T", # ت ط
        "t", # ت ط
        "Z", # ز ض ظ ذ
        "z", # ز ض ظ ذ
        "r", # ر
        "D", # د
        "d", # د
        "S", # س ص ث
        "s"  # س ص ث
        ]

    # e.g. UR: "0Allah0" -> SF: "0'allA0"
    if len(stem) >= 5 and stem.startswith("0'a") and stem[3] == stem[4] and stem[3] in shamsi:
        chars = list(stem)
        chars[3] = "S"
        chars[4] = "S"
        stem = "".join(chars)
    # e.g. UR: "0raqab=a0" -> SF: "0raqaba0"
    elif len(stem) >= 2 and stem[0] == "0" and stem[1] in shamsi:
        chars = list(stem)
        chars[1] = "S"
        stem = "".join(chars)
    # word might include prefix+stem+suffix which makes stem not start with "0"
    elif len(stem) >= 1 and stem[0] in shamsi:
        chars = list(stem)
        chars[0] = "S"
        stem = "".join(chars)

    vowels = ["A", "E", "I", "O", "U", "a", "i", "u"]
    new_stem = ""
    for e in stem:
        if e in vowels:
            new_stem += e
        elif e == "#" or e == "0":
            new_stem += e
        elif e == "S":
            new_stem += e
        else:
            new_stem += "C"

    return new_stem


# it get the sf and ur and beak them to pieces
# for example (examples are not accurate in terms of arabic context of meaning):
# $a-ki-taba=hu=la , $akitabahula --> ($a-ki,taba,hu=la) , ($a-ki,taba,hula)
# it uses the Leveshtine editops to do that it's a little bit tricky in the SF part cause the changes in SF lenght wise
def breakdown_ur_sf(ur, sf) -> Tuple[Tuple[str, str, str], Tuple[str, str, str]]:
    prefix_change = get_nearest_prefix_to_stem(ur, sf)
    suffix_change = get_nearest_suffix_to_stem(ur, sf)
    ur_pre, ur_stem, ur_suf = ("", "", "")
    sf_pre, sf_stem, sf_suf = ("", "", "")

    if prefix_change is None and suffix_change is None:
        ur_stem = ur
        sf_stem = sf
    elif prefix_change is not None and suffix_change is None:
        ur_pre_change = prefix_change[1] + 1
        sf_pre_change = prefix_change[2] + 1 if prefix_change[0] == "replace" else prefix_change[2]
        ur_pre = ur[:ur_pre_change]
        ur_stem = ur[ur_pre_change:]
        sf_pre = sf[:sf_pre_change]
        sf_stem = sf[sf_pre_change:]
    elif suffix_change is not None and prefix_change is None:
        ur_suf_change = suffix_change[1]
        sf_suf_change = suffix_change[2] if suffix_change[0] == "delete" else suffix_change[2]
        ur_stem = ur[:ur_suf_change]
        ur_suf = ur[ur_suf_change:]
        sf_stem = sf[:sf_suf_change]
        sf_suf = sf[sf_suf_change:]

    else:
        ur_pre_change = prefix_change[1] + 1
        sf_pre_change = prefix_change[2] + 1 if prefix_change[0] == "replace" else prefix_change[2]
        ur_suf_change = suffix_change[1]
        sf_suf_change = suffix_change[2] if suffix_change[0] == "delete" else suffix_change[2]

        ur_pre = ur[:ur_pre_change]
        ur_stem = ur[ur_pre_change:ur_suf_change]
        ur_suf = ur[ur_suf_change:]
        sf_pre = sf[:sf_pre_change]
        sf_stem = sf[sf_pre_change:sf_suf_change]
        sf_suf = sf[sf_suf_change:]

    return (ur_pre, ur_stem, ur_suf), (sf_pre, sf_stem, sf_suf)

# this is the beginning function
def UR_SF_generalizer(ur, sf):
    ur, sf = breakdown_ur_sf(ur, sf)
    ur_pre, ur_stem, ur_suf = ur
    sf_pre, sf_stem, sf_suf = sf

    # CHECKPOINT_SHAMSI

    # ur_stem = stem_generalizer(ur_stem)
    # sf_stem = stem_generalizer(sf_stem)

    ur_stem = ur_stem_generalizer(ur_stem)
    sf_stem = sf_stem_generalizer(sf_stem)

    ur = ur_pre + ur_stem + ur_suf
    sf = sf_pre + sf_stem + sf_suf
    return ur, sf


# # 0mitwakkil0 0mitwakkil0
# # ('0CiCCaCCiC0', '0CiCCaCCiC0')
# # x = "0 m i t w a k k i l 0	0 m i t w a k k i l 0"
# # x = "0 ' i l - H a l a w i y y = A t 0	0 ' i l H a l a w i y y A t 0"
# x = "0 t i - x a l l i f = I 0	0 t i x a l l i f i 0"
# x = "0 r a q a b = a 0	0 r a q a b a 0"
# (ur, sf) = [v.replace(" ", "") for v in x.split("\t")]
# print(UR_SF_generalizer(ur, sf))
# print(UR_SF_generalizer(ur, ""))
# print(UR_SF_generalizer(sf, sf))
# print(UR_SF_generalizer("jinEn=at", "jinEnit"))


# # measuring the distance between UR and Rule's UR
# distance = Levenshtein.distance("UR", "Rule's UR")

# # another cool library that can be helpful with measuring distance
# # giving most similar strings to another string
# similar_urs = process.extract(query="UR", choices=["UR0", "UR1", "UR2"],
#                               limit=100)  # it will return the most similar ur form choices query
