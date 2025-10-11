# ai_scoring.py
import json
import os
import re
import difflib
import requests
import time
import pandas as pd
import openai
import numpy as np
import spacy_thai
from pythainlp.tokenize import sent_tokenize
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_words
from pythainlp.tag import pos_tag
from pythainlp.util import normalize
from pythainlp.spell import spell
from sklearn.metrics.pairwise import cosine_similarity

# โหลดโมเดล
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ✅ -----------------------------
# หา path ของไฟล์ json ในโฟลเดอร์ data
BASE_DIR = os.path.dirname(__file__)  # โฟลเดอร์ปัจจุบัน

# ---- thai_loanwords ----
json_path = os.path.join(BASE_DIR, "data", "thai_loanwords_new_update.json")
try:
    with open(json_path, "r", encoding="utf-8") as f:
        thai_loanwords = json.load(f)
    loanwords_whitelist = {
        item["thai_word"] for item in thai_loanwords if "thai_word" in item
    }
except FileNotFoundError:
    print(f"⚠️ ไม่พบไฟล์: {json_path}")
    thai_loanwords = []
    loanwords_whitelist = set()

# ---- common misspellings ----
misspellings_path = os.path.join(BASE_DIR, "data", "update_common_misspellings.json")

try:
    with open(misspellings_path, "r", encoding="utf-8") as f:
        _cm = json.load(f)
    COMMON_MISSPELLINGS = {item["wrong"]: item["right"] for item in _cm}
except FileNotFoundError:
    print(f"⚠️ ไม่พบไฟล์: {misspellings_path} (COMMON_MISSPELLINGS ตั้งเป็น empty dict)")
    COMMON_MISSPELLINGS = {}

API_KEY = '33586c7cf5bfa0029887a9831bf94963' # add Apikey
API_URL = 'https://api.longdo.com/spell-checker/proof'

custom_words = {"ประเทศไทย", "สถาบันการศึกษา", "นานาประการ"}

#คำที่สามารถฉีกคำได้
splitable_phrases = {
    'แม้ว่า', 'ถ้าแม้ว่า', 'แต่ถ้า', 'แต่ทว่า', 'เนื่องจาก', 'ดังนั้น', 'เพราะฉะนั้น','ตกเป็น','เป็นการ',
    'ดีแต่', 'หรือไม่', 'ข้อมูลข่าวสาร', 'ทั่วโลก', 'ยังมี', 'ทำให้เกิด', 'เป็นโทษ', 'ไม่มี', 'ข้อควรระวัง', 'การแสดงความคิดเห็น', 'ผิดกฎหมาย', 'แสดงความคิดเห็น'
}
#คำที่ไม่สามารถฉีกคำได้
strict_not_split_words = {
    'มากมาย', 'ประเทศไทย', 'ออนไลน์', 'ความคิดเห็น', 'ความน่าเชื่อถือ'
}

thai_dict = set(w for w in set(thai_words()).union(custom_words) if (' ' not in w) and w.strip())

# allowed punctuation (เพิ่ม ' และ ")
allowed_punctuations = {'.', ',', '-', '(', ')', '!', '?', '%', '“', '”', '‘', '’', '"', "'", '…', 'ฯ'}

# Allow / Forbid list ไม้ยมก (เพิ่มคำที่ใช้บ่อย)
allow_list = {'ปี', 'อื่น', 'เล็ก', 'ใหญ่', 'มาก', 'หลาย', 'ช้า', 'เร็ว', 'ชัด', 'ดี', 'ผิด'}
forbid_list = {'นา', 'บางคน', 'บางอย่าง', 'บางสิ่ง', 'บางกรณี'}

explanations = [
    "1. ตรวจสอบการฉีกคำ",
    "2. ตรวจสอบคำสะกดผิดด้วย PyThaiNLP (และขอ Longdo ช่วยกรณีสงสัย)",
    "3. ตรวจสอบการใช้เครื่องหมายที่ไม่อนุญาต",
    "4. ตรวจสอบการใช้ไม้ยมก (ๆ) ถูกต้องตามบริบทหรือไม่",
    "5. ตรวจสอบการแยกคำผิด เช่น คำที่ควรติดกัน"
]
# ✅ -----------------------------

def normalize_text(text):
    text = " ".join(text.replace("\n", " ").replace("\r", " ").replace("\t", " ").split())
    return text.replace(" ", "")

def find_keywords_list(text, keywords):
    found = [kw for kw in keywords if kw.replace(" ", "") in text]
    return found

def score_group_1(text):
    text_norm = normalize_text(text)
    media_keywords = ["สื่อสังคมออนไลน์", "สื่อสังคม", "สื่อออนไลน์"]
    usage_keywords = ["เป็นช่องทาง", "ช่องทาง", "เป็นการแพร่กระจาย", "เป็นสื่อ", "สามารถ", "ทำให้", "เป็นการกระจาย", "นั้น"]
    last_keywords = ["แพร่กระจาย", "แพร่กระจายข่าวสาร", "ค้นหา", "รับข้อมูลข่าวสาร", "เผยแพร่", "ติดต่อสื่อสาร", "กระจาย", "รับสาร","รับรู้"]

    found_usage = [kw for kw in usage_keywords if kw.replace(" ", "") in text_norm]
    found_last = [kw for kw in last_keywords if kw.replace(" ", "") in text_norm]
    first_5_words = text.split()[:5]
    first_5_text = "".join(first_5_words)
    found_media_in_first_5 = any(kw in first_5_text for kw in media_keywords)

    score = 1 if (found_media_in_first_5 and found_usage and found_last) else 0
    return score

def score_group_2(text):
    text_norm = normalize_text(text)
    keypoints_1 = ["ไม่ระวัง", "ไม่ระมัดระวัง", "ขาดความรับผิดชอบ", "ควรระมัดระวัง", "ใช้ในทางที่ไม่ดี", "ไม่เหมาะสม", "อย่างระมัดระวัง", "ไตร่ตรอง"]
    keypoints_2 = [
        "โทษ", "ผลเสีย", "ข้อเสีย", "เกิดผลกระทบ", "สิ่งไม่ดี",
        "เสียหาย",
        "การเขียนแสดงความเห็นวิพากษ์วิจารณ์ผู้อื่นในทางเสียหาย",
        "การเขียนแสดงความคิดเห็นวิพากษ์วิจารณ์ผู้อื่นในทางเสียหาย",
        "ตกเป็นเหยื่อของมิจฉาชีพ",
        "ตกเป็นเหยื่อมิจฉาชีพ", "ตกเป็นเหยื่อทางการตลาด"
    ]
    found_1 = find_keywords_list(text_norm, keypoints_1)
    found_2 = find_keywords_list(text_norm, keypoints_2)
    found_illegal = "ผิดกฎหมาย" in text_norm

    score = 1 if (found_1 and found_2) or (found_1 and found_illegal and found_2) else 0
    return score

def score_group_3(text):
    text_norm = normalize_text(text)
    media_keypoint = ["สื่อสังคมออนไลน์", "สื่อสังคม", "สื่อออนไลน์"]
    keypoints = ["รู้เท่าทัน", "รู้ทัน", "ผู้ใช้ต้องรู้เท่าทัน", "รู้ทันสื่อสังคม",
                 "รู้เท่าทันสื่อ", "รู้ทันสื่อ", "สร้างภูมิคุ้มกัน", "ไม่ตกเป็นเหยื่อ", "แก้ปัญหาการตกเป็นเหยื่อ"]

    found_1 = find_keywords_list(text_norm, media_keypoint)
    found_2 = find_keywords_list(text_norm, keypoints)

    score = 1 if (found_1 and found_2) else 0
    return score

def score_group_4(text):
    text_norm = normalize_text(text)
    media_use_keywords = [
        "ใช้สื่อสังคม", "ใช้สื่อออนไลน์", "ใช้สื่อสังคมออนไลน์", "การใช้สื่อ"
    ]
    hidden_intent_keywords = ["เจตนาแอบแฝง"]
    effect_keywords = ["ผลกระทบต่อ", "ผลกระทบ"]
    credibility_keywords = [
        "ความน่าเชื่อถือของข่าวสาร", "ความน่าเชื่อถือของข้อมูลข่าวสาร", "ความน่าเชื่อถือของข้อมูล",
        "มีสติ", "ความน่าเชื่อถือ", "ความเชื่อถือของข้อมูลข่าวสาร", "ข้อมูลข่าวสาร"
    ]
    words = text.split()

    def find_positions(words, keywords):
        positions = []
        joined_text = "".join(words)
        for kw in keywords:
            start = 0
            while True:
                idx = joined_text.find(kw.replace(" ", ""), start)
                if idx == -1:
                    break
                positions.append(len(joined_text[:idx].split()))
                start = idx + len(kw.replace(" ", ""))
        return positions

    media_positions = find_positions(words, media_use_keywords)
    hidden_positions = find_positions(words, hidden_intent_keywords)
    effect_positions = find_positions(words, effect_keywords)
    # ตำแหน่ง media ก่อน hidden หรือ effect (ตามแบบเดิม)
    media_before_hidden = any((0 < h - m <= 5) for m in media_positions for h in hidden_positions)
    media_before_effect = any((0 < e - m <= 5) for m in media_positions for e in effect_positions)

    # ตรวจพบกลุ่ม keyword
    found_hidden_intent = find_keywords_list(text_norm, hidden_intent_keywords)
    found_effect = find_keywords_list(text_norm, effect_keywords)
    found_credibility = find_keywords_list(text_norm, credibility_keywords)

    # ต้องเจอทั้ง hidden_intent ผลกระทบ และ credibility ครบทั้ง 3 อย่าง
    score = 1 if (found_hidden_intent and found_effect and found_credibility) else 0
    return score

def evaluate_mind_score(answer_text):
    score1 = score_group_1(answer_text)
    score2 = score_group_2(answer_text)
    score3 = score_group_3(answer_text)
    score4 = score_group_4(answer_text)
    total_score = score1 + score2 + score3 + score4

    result = {
        "ใจความที่ 1": score1,
        "ใจความที่ 2": score2,
        "ใจความที่ 3": score3,
        "ใจความที่ 4": score4,
        "คะแนนรวมใจความ": total_score
    }

    return result

# -----------------------------
# Helper: ข้ามคำภาษาอังกฤษ/ตัวเลข
# -----------------------------
def is_english_or_number(word: str) -> bool:
    """
    คืน True ถ้า word เป็นภาษาอังกฤษหรือตัวเลข (หรือประกอบด้วยสัญลักษณ์ ASCII บางชนิด)
    """
    w = (word or "").strip()
    if not w:
        return False
    # อนุญาต A-Z a-z 0-9 และ . , ( ) - _ /
    return bool(re.fullmatch(r"[A-Za-z0-9\.\,\-\(\)_/]+", w))

# -----------------------------
# ตรวจ common misspellings (จากข้อความดิบ)
# -----------------------------
def check_common_misspellings_before_tokenize(text: str, misspelling_dict: dict):
    """
    text : ข้อความดิบ (ยังไม่ tokenize)
    misspelling_dict : dict เช่น { "ผิพท์": "พิมพ์", ... }
    คืน list ของ dict ที่มี keys: 'word' (wrong), 'index' (ตำแหน่งในข้อความดิบ), 'right' (คำถูก)
    """
    errors = []
    if not misspelling_dict:
        return errors
    for wrong, right in misspelling_dict.items():
        if wrong in text:
            for m in re.finditer(re.escape(wrong), text):
                errors.append({
                    "word": wrong,
                    "index": m.start(),
                    "right": right
                })
    return errors

# -----------------------------
# ตรวจ loanwords ก่อน tokenize (ใช้กับ tokens)
# -----------------------------
def check_loanword_before_tokenize(tokens, whitelist):
    """
    tokens : list ของ token (ตัดแล้ว)
    whitelist : set/list ของคำทับศัพท์ภาษาไทย (ไทยเขียนไม่ผิด)
    คืน list ของ dict: {'word': token, 'index': position, 'suggestions': [best_match]}
    """
    mistakes = []
    wl_list = list(whitelist) if whitelist else []
    for i, w in enumerate(tokens):
        if not w or is_english_or_number(w):
            continue
        # หา match ใกล้เคียงจาก whitelist
        matches = difflib.get_close_matches(w, wl_list, n=1, cutoff=0.7)
        if matches and w not in whitelist:
            mistakes.append({
                "word": w,
                "index": i,
                "suggestions": [matches[0]]
            })
    return mistakes

#ตรวจการฉีกคำ
def check_linebreak_issue(prev_line_tokens, next_line_tokens, max_words=3):
    last_word = prev_line_tokens[-1]
    first_word = next_line_tokens[0]
    if last_word.endswith('-') or first_word.startswith('-'):
        return False, None, None, None
    for prev_n in range(1, min(max_words, len(prev_line_tokens)) + 1):
        prev_part = ''.join(prev_line_tokens[-prev_n:])
        for next_n in range(1, min(max_words, len(next_line_tokens)) + 1):
            next_part = ''.join(next_line_tokens[:next_n])
            combined = normalize(prev_part + next_part)
            if (
                (' ' not in combined)
                and (combined not in splitable_phrases)
                and (
                    (combined in strict_not_split_words) or (
                        (combined in thai_dict)
                        and (len(word_tokenize(combined, engine='newmm')) == 1)
                    )
                )
            ):
                return True, prev_part, next_part, combined
    return False, None, None, None

#วนตรวจทั้งข้อความทีละบรรทัด
def analyze_linebreak_issues(text):
    lines = text.strip().splitlines()
    issues = []
    for i in range(len(lines) - 1):
        prev_line = lines[i].strip()
        next_line = lines[i + 1].strip()
        prev_tokens = word_tokenize(prev_line)
        next_tokens = word_tokenize(next_line)
        if not prev_tokens or not next_tokens:
            continue
        issue, prev_part, next_part, combined = check_linebreak_issue(prev_tokens, next_tokens)
        if issue:
            issues.append({
                'line_before': prev_line,
                'line_after': next_line,
                'prev_part': prev_part,
                'next_part': next_part,
                'combined': combined,
                'pos_in_text': (i, len(prev_tokens))
            })
    return issues

#รวมข้อความหรือคำที่ถูกตัดข้ามบรรทัด
def merge_linebreak_words(text, linebreak_issues):
    lines = text.splitlines()
    for issue in reversed(linebreak_issues):
        i, _ = issue['pos_in_text']
        lines[i] = lines[i].rstrip() + issue['combined'] + lines[i+1].lstrip()[len(issue['next_part']):]
        lines.pop(i+1)
    return "\n".join(lines)

#ตรวจการสสะกดคำ pythainlp + longdo
def pythainlp_spellcheck(tokens, pos_tags, dict_words=None, ignore_words=None):
    if dict_words is None:
        dict_words = thai_dict
    if ignore_words is None:
        ignore_words = set()
    misspelled = []
    for i, w in enumerate(tokens):
        if not w.strip() or w in dict_words or w in ignore_words or len(w) == 1 or 'ๆ' in w:
            continue
        misspelled.append({
            'word': w,
            'pos': pos_tags[i][1] if i < len(pos_tags) else None,
            'index': i
        })
    return misspelled

def longdo_spellcheck_batch(words):
    results = {}
    if not words:
        return results
    try:
        headers = {'Content-Type': 'application/json'}
        payload = {"key": API_KEY, "text": "\n".join(words)}
        response = requests.post(API_URL, headers=headers, json=payload, timeout=6)
        if response.status_code == 200:
            result = response.json()
            for e in result.get("result", []):
                if e.get("suggestions"):
                    results[e["word"]] = e["suggestions"]
    except Exception as e:
        print(f"Exception calling longdo: {e}")
    return results

#ตรวจการสะกดคำของคำทับศัพท์
def check_loanword_spelling(tokens,loanwords_whitelist):
    mistakes = []
    for tok in tokens:
        # Find close matches with a lower cutoff for loanwords
        matches = difflib.get_close_matches(tok, list(loanwords_whitelist), n=1, cutoff=0.7) # Lowered cutoff
        if matches and tok not in loanwords_whitelist:
            mistakes.append({'found': tok, 'should_be': matches[0]})
    return mistakes

#ตรวจการใช้เครื่องหมายที่ไม่อนุญาต
def find_unallowed_punctuations(text):
    pattern = f"[^{''.join(re.escape(p) for p in allowed_punctuations)}a-zA-Z0-9ก-๙\\s]"
    return set(re.findall(pattern, text))

#ใช้แยกไม้ยมกออกจากคำที่ติดกัน
def separate_maiyamok(text):
    return re.sub(r'(\S+?)ๆ', r'\1 ๆ', text)
#ตรวจการใช้ไม้ยมก
def analyze_maiyamok(tokens, pos_tags):
    results = []
    found_invalid = False
    VALID_POS = {'NCMN', 'NNP', 'VACT', 'VNIR', 'CLFV', 'ADVN', 'ADVI', 'ADVP', 'PRP', 'ADV'}
    for i, token in enumerate(tokens):
        if token == 'ๆ':
            prev_idx = i - 1
            prev_word = tokens[prev_idx] if prev_idx >= 0 else None
            prev_tag = pos_tags[prev_idx][1] if prev_idx >= 0 else None
            if prev_word is None or prev_word == 'ๆ':
                verdict = "❌ ไม้ยมกไม่ควรขึ้นต้นประโยค/คำ"
            elif prev_word in forbid_list:
                verdict = '❌ ไม่ควรใช้ไม้ยมกกับคำนี้'
            elif (prev_tag in VALID_POS) or (prev_word in allow_list):
                verdict = '✅ ถูกต้อง (ใช้ไม้ยมกซ้ำคำได้)'
            else:
                verdict = '❌ ไม่ควรใช้ไม้ยมok นอกจากกับคำนาม/กริยา/วิเศษณ์'
            context = tokens[max(0, i-2):min(len(tokens), i+3)]
            results.append({
                'คำก่อนไม้ยมก': prev_word or '',
                'POS คำก่อน': prev_tag or '',
                'บริบท': ' '.join(context),
                'สถานะ': verdict
            })
            if verdict.startswith('❌'):
                found_invalid = True
    return results, found_invalid

#ตรวจการแยกคำ
def detect_split_errors(tokens, custom_words=None):
    check_dict = set(thai_words()).union(custom_words or [])
    check_dict = {w for w in check_dict if (' ' not in w) and w.strip()}
    errors = []
    for i in range(len(tokens) - 1):
        combined = tokens[i] + tokens[i + 1]
        if (' ' not in combined) and (combined in check_dict) and (combined not in splitable_phrases):
            errors.append({
                "split_pair": (tokens[i], tokens[i+1]),
                "suggested": combined
            })
    return errors

def evaluate_text(text):
    # -----------------------------
    # จัดการตัดบรรทัด
    # -----------------------------
    linebreak_issues = analyze_linebreak_issues(text)
    corrected_text = merge_linebreak_words(text, linebreak_issues)

    # tokenize
    tokens = word_tokenize(corrected_text, engine='newmm', keep_whitespace=False)
    pos_tags = pos_tag(tokens, corpus='orchid')

    # ✅ 1) ตรวจ spelling ด้วย PyThaiNLP
    pythai_errors = pythainlp_spellcheck(tokens, pos_tags, dict_words=thai_dict, ignore_words=custom_words)

    # ✅ 2) ตรวจ Longdo (batch)
    wrong_words = [e['word'] for e in pythai_errors]
    longdo_results = longdo_spellcheck_batch(wrong_words)
    longdo_errors = [
        {**e, 'suggestions': longdo_results.get(e['word'], [])}
        for e in pythai_errors if e['word'] in longdo_results
    ]

    # ✅ 3) ตรวจ common misspellings จากข้อความดิบ
    json_misspells = check_common_misspellings_before_tokenize(corrected_text, COMMON_MISSPELLINGS)

    # ✅ 4) ตรวจ loanwords
    loanword_errors = check_loanword_before_tokenize(tokens, loanwords_whitelist)

    # ✅ รวม spelling errors ทั้งหมด
    all_spelling_errors = longdo_errors + [
        {
            "word": e["word"],
            "pos": None,
            "index": e["index"],
            "suggestions": [e["right"]],
        }
        for e in json_misspells
    ] + loanword_errors

    # ✅ ตรวจ punctuation, maiyamok, split word
    punct_errors = find_unallowed_punctuations(corrected_text)
    maiyamok_results, has_wrong_maiyamok = analyze_maiyamok(tokens, pos_tags)
    split_errors = detect_split_errors(tokens, custom_words=custom_words)

    # ✅ รวมผล errors
    error_counts = {
        "spelling": len(all_spelling_errors),
        "linebreak": len(linebreak_issues),
        "split": len(split_errors),
        "punct": len(punct_errors),
        "maiyamok": sum(1 for r in maiyamok_results if r['สถานะ'].startswith('❌'))
    }

    # ✅ สร้าง reasons
    reasons = []
    if error_counts["linebreak"]:
        details = [f"{issue['prev_part']} + {issue['next_part']} → {issue['combined']}" for issue in linebreak_issues]
        reasons.append("พบการฉีกคำข้ามบรรทัด: " + "; ".join(details))
    if error_counts["split"]:
        details = [f"{e['split_pair'][0]} + {e['split_pair'][1]} → {e['suggested']}" for e in split_errors]
        reasons.append("พบการแยกคำผิด: " + "; ".join(details))
    if error_counts["spelling"]:
        error_words = []
        for e in all_spelling_errors:
            suggestions = e.get('suggestions', [])
            safe_suggestions = [str(s) for s in suggestions if s]
            suggestion_text = ', '.join(safe_suggestions) if safe_suggestions else 'ไม่มีคำแนะนำ'
            error_words.append(f"{e.get('word', '?')} (แนะนำ: {suggestion_text})")

        reasons.append(f"ตรวจเจอคำสะกดผิดหรือทับศัพท์ผิด: {', '.join(error_words)}")
    if error_counts["punct"]:
        reasons.append(f"ใช้เครื่องหมายที่ไม่อนุญาต: {', '.join(punct_errors)}")
    if error_counts["maiyamok"]:
        wrong_desc = [x for x in maiyamok_results if x['สถานะ'].startswith('❌')]
        texts = [f"{x['คำก่อนไม้ยมก']}: {x['สถานะ']}" for x in wrong_desc]
        reasons.append("ใช้ไม้ยมกผิด: " + '; '.join(texts))
    if not reasons:
        reasons.append("ไม่มีปัญหา")

    # ✅ การให้คะแนน
    if sum(error_counts.values()) == 0:
        score = 1.0
    elif sum(c > 0 for c in error_counts.values()) == 1 and max(error_counts.values()) >= 2:
        score = 0.0
    elif sum(c > 0 for c in error_counts.values()) == 1:
        score = 0.5
    else:
        score = 0.0

    return {
        'linebreak_issues': linebreak_issues,
        'spelling_errors': all_spelling_errors,
        'loanword_spell_errors': loanword_errors,
        'punctuation_errors': list(punct_errors),
        'maiyamok_results': maiyamok_results,
        'split_errors': split_errors,
        'reasons': reasons,
        'score': score
    }


# ==========================
# S2---ฟังก์ชันตรวจเรียงลำดับ/เชื่อมโยงความคิด
# ==========================
ignore_list = ["สื่อ", "สื่อออนไลน์", "สื่อสังคม", "สื่อสังคมออนไลน์",
               "ออนไลน์", "ออนไลท์", "\n", "ด้วยการ", "จนทำให้", "การใช้",
               "ต่อสังคม", "ในทาง", "การทำ", "อย่างไม่", "สังคม",
               "ยังไม่", "ได้อย่าง", "เราควร", "ใช้ใน", "เราจึง", "เข้าได้",
               "ทางที่", "ใช้ในการ", "ให้แก่", "เป็นช่องทาง", "ในการ", "ถูกหลอก"]

specific_terms = ["ผิดกฎหมาย", "โฆษณา"]

ignore_single_char = ["สิ", "สี่", "สัญญา", "ผิดก", "หริ", "รู", "ภูมิ",
                      "เจ", "คา", "เป้", "เสีย", "หาย", "ผิด", "ที",
                      "สี" , "ริ" , "ข่อ" , "ออนไลน์", "โท"]

def preprocess_text(text):
    return "".join(text.split())

def is_thai_word(word):
    return bool(re.fullmatch(r'[\u0E00-\u0E7F]+', word))

def check_thai_text_integrity(text, ignore_single_char=None):
    if ignore_single_char is None:
        ignore_single_char = []
    text_clean = preprocess_text(text)
    words = word_tokenize(text_clean, engine='newmm')
    thai_words_only = [w for w in words if is_thai_word(w)]
    single_char_words = []
    for i, w in enumerate(thai_words_only):
        if len(w) == 1:
            preceding = thai_words_only[i-1] if i > 0 else None
            if preceding in ignore_single_char:
                continue
            single_char_words.append({"word": w, "preceding": preceding})
    special_violations = []
    return single_char_words, special_violations

def semantic_similarity_lines(text, threshold=0.3):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    embeddings = model.encode(lines)
    failed_lines = []
    for i, emb in enumerate(embeddings):
        sims = [util.cos_sim(emb, embeddings[j]).item() for j in range(len(embeddings)) if j != i]
        max_sim = max(sims) if sims else 1.0
        if max_sim < threshold:
            failed_lines.append({
                "line_index": i,
                "line_text": lines[i],
                "max_similarity": round(max_sim, 3)
            })
    return failed_lines

def ngrams(tokens, n):
    return ["".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def find_repeated_ngrams(tokens, min_len=2, ignore_list=None):
    if ignore_list is None:
        ignore_list = []
    repeated = {}
    max_n = len(tokens)
    for n in range(min_len, max_n+1):
        ngs = ngrams(tokens, n)
        counts = {}
        for ng in ngs:
            if any(ignore in ng for ignore in ignore_list):
                continue
            counts[ng] = counts.get(ng, 0) + 1
        for ng, c in counts.items():
            if c > 1:
                repeated[ng] = c
    longest_only = {}
    for ng, count in repeated.items():
        if not any((ng in other and len(other) > len(ng)) for other in repeated):
            longest_only[ng] = count
    repeated_words_count = sum(count - 1 for count in longest_only.values())
    return {"repeated_ngrams": longest_only, "count": repeated_words_count}

def find_specific_terms(text, specific_terms):
    found = {}
    total_count = 0
    for term in specific_terms:
        count = text.count(term)
        if count > 1:
            found[term] = count
            total_count += count - 1
    return {"specific_found": found, "count": total_count}

def evaluate_student_answer(student_text, ignore_list=None, specific_terms=None, ignore_single_char=None, similarity_threshold=0.3):
    if ignore_list is None:
        ignore_list = []
    if specific_terms is None:
        specific_terms = []

    single_char_words, special_violations = check_thai_text_integrity(student_text, ignore_single_char)
    missing_content = {
        "single_char_words": single_char_words,
        "special_violations": special_violations
    }

    s_clean = student_text.replace("\n", "")
    tokens = [t for t in word_tokenize(s_clean, keep_whitespace=False) if t.strip()]
    repeated_ngrams_result = find_repeated_ngrams(tokens, min_len=2, ignore_list=ignore_list)
    specific_found_result = find_specific_terms(s_clean, specific_terms)

    duplicate_content = {
        "repeated_ngrams": repeated_ngrams_result["repeated_ngrams"],
        "specific_found": specific_found_result["specific_found"]
    }

    failed_similarity = semantic_similarity_lines(student_text, threshold=similarity_threshold)
    semantic_issue = failed_similarity

    score = 2
    if single_char_words:
        score -= 1
    if special_violations:
        score -= 1
    score -= repeated_ngrams_result["count"]
    score -= specific_found_result["count"]
    if failed_similarity:
        score -= 1
    score = max(0, score)

    return {
        "เนื้อความขาด": missing_content,
        "เนื้อความซ้ำ": duplicate_content,
        "เนื้อความไม่สัมพันธ์กัน": semantic_issue,
        "คะแนนรวม": score
    }


#---------S3 ความถูกต้องตามหลักการเขียนย่อความ-------------
# ---------- ตั้งค่า ----------
TNER_URL = 'https://api.aiforthai.in.th/tner'
#AIFORTHAI_URL = "https://api.aiforthai.in.th/qaiapp"


# ---------- โหลด Dataset ----------
examples_df = pd.read_csv(r'D:\\project1\\example_dialect (3).csv')
pronouns_df = pd.read_csv(r'D:\\project1\\personal_pronoun (1).csv')

example_phrases = examples_df['local_word'].dropna().tolist()
pronouns_1 = pronouns_df['personal pronoun 1'].dropna().tolist()
pronouns_2 = pronouns_df['personal pronoun 2'].dropna().tolist()
pronouns_1_2 = pronouns_1 + pronouns_2

# ---------- บทความอ้างอิง ----------
reference_text = """
สื่อสังคม (Social Media) หรือที่คนทั่วไปเรียกว่า สื่อออนไลน์ หรือ สื่อสังคม ออนไลน์ นั้น เป็นสื่อหรือช่องทางที่แพร่กระจายข้อมูลข่าวสารในรูปแบบต่างๆ ได้อย่างรวดเร็วไปยังผู้คนที่อยู่ทั่วทุกมุมโลกที่สัญญาณโทรศัพท์เข้าถึง เช่น การนําเสนอข้อดีนานาประการของสินค้าชั้นนํา สินค้าพื้นเมืองให้เข้าถึงผู้ซื้อได้
ทั่วโลก การนําเสนอข้อเท็จจริงของข่าวสารอย่างตรงไปตรงมา การเผยแพร่ งานเขียนคุณภาพบนโลกออนไลน์แทนการเข้าสํานักพิมพ์ เป็นต้น จึงกล่าวได้ว่า เราสามารถใช้สื่อสังคมออนไลน์ค้นหาและรับข้อมูลข่าวสารที่มีประโยชน์ได้เป็นอย่างดี
  อย่างไรก็ตาม หากใช้สื่อสังคมออนไลน์อย่างไม่ระมัดระวัง หรือขาดความรับผืดชอบต่อสังคมส่วนรวม ไม่ว่าจะเป็นการเขียนแสดงความคิดเห็นวิพากษ์วิจารณ์ผู้อื่นในทางเสียหาย การนำเสนอผลงานที่มีเนื้อหาล่อแหลมหรือชักจูงผู้รับสารไปในทางไม่เหมาะสม หรือการสร้างกลุ่มเฉพาะที่ขัดต่อศีลธรรมอันดีของสังคมตลอดจนใช้เป็นช่องทางในการกระทำผิดกฎหมายทั้งการพนัน การขายของ
ผิดกฎหมาย เป็นต้น การใช้สื่อสังคมออนไลน์ในลักษณะดังกล่าวจึงเป็นการใช้ที่เป็นโทษแก่สังคม
	ปัจจุบันผู้คนจํานวนไม่น้อยนิยมใช้สื่อสังคมออนไลน์เป็นช่องทางในการทํา การตลาดทั้งในทางธุรกิจ สังคม และการเมือง จนได้ผลดีแบบก้าวกระโดด ทั้งนี้ เพราะสามารถเข้าถึงกลุ่มคนทุกเพศ ทุกวัย และทุกสาขาอาชีพโดยไม่มีข้อจํากัดเรื่อง เวลาและสถานที่ กลุ่มต่างๆ ดังกล่าวจึงหันมาใช้สื่อสังคมออนไลน์เพื่อสร้างกระแสให้ เกิดความนิยมชมชอบในกิจการของตน ด้วยการโฆษณาชวนเชื่อทุกรูปแบบจนลูกค้า เกิดความหลงใหลข้อมูลข่าวสาร จนตกเป็นเหยื่ออย่างไม่รู้ตัว เราจึงควรแก้ปัญหา การตกเป็นเหยื่อทางการตลาดของกลุ่มมิจฉาชีพด้วยการเร่งสร้างภูมิคุ้มกันรู้ทันสื่อไม่ตกเป็นเหยื่อทางการตลาดโดยเร็ว
	แม้ว่าจะมีการใช้สื่อสังคมออนไลน์ในทางสร้างสรรค์สิ่งที่ดีให้แก่สังคม ตัวอย่างเช่น การเตือนภัยให้แก่คนในสังคมได้อย่างรวดเร็ว การส่งต่อข้อมูลข่าวสาร เพื่อระดมความช่วยเหลือให้แก่ผู้ที่กําลังเดือดร้อน เป็นต้น แต่หลายครั้งคนในสังคมก็ อาจรู้สึกไม่มั่นใจเมื่อพบว่าตนเองถูกหลอกลวงจากคนบางกลุ่มที่ใช้สื่อสังคมออนไลน์
เป็นพื้นที่แสวงหาผลประโยชน์ส่วนตัว จนทําให้เกิดความเข้าใจผิดและสร้างความ เสื่อมเสียให้แก่ผู้อื่น ดังนั้นการใช้สื่อสังคมออนไลน์ด้วยเจตนาแอบแฝงจึงมีผลกระทบต่อความน่าเชื่อถือของข้อมูลข่าวสารโดยตรง
"""

# ---------- ฟังก์ชันตรวจหลักการย่อความ ----------
def call_tner(text):
    headers = {'Apikey': API_KEY}
    data = {'text': text}
    try:
        resp = requests.post(TNER_URL, headers=headers, data=data, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"TNER API error: {e}")
    return None

def check_summary_similarity(student_answer, reference_text, threshold=0.8):
    embeddings = model.encode([student_answer, reference_text])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim >= threshold, sim

def check_examples(student_answer, example_phrases):
    return not any(phrase in student_answer for phrase in example_phrases)

def check_pronouns(student_answer, pronouns_list):
    words = word_tokenize(student_answer, engine='newmm')
    return not any(p in words for p in pronouns_list)

def check_abbreviations(student_answer):
    pattern = r'\b(?:[ก-ฮA-Za-z]\.){2,}'
    if re.search(pattern, student_answer):
        return False
    tner_result = call_tner(student_answer)
    if tner_result:
        for item in tner_result.get('entities', []):
            if item['type'] in ['ABB_DES', 'ABB_TTL', 'ABB_ORG', 'ABB_LOC', 'ABB']:
                return False
    return True

def check_title(student_answer, forbidden_title="การใช้สื่อสังคมออนไลน์"):
    return forbidden_title not in student_answer

def validate_student_answer(student_answer):
    sim_pass, sim_score = check_summary_similarity(student_answer, reference_text)
    results = {
        "การย่อความผิดไปจากตัวบทอ่าน": sim_pass,
        "similarity_score": round(sim_score, 3),
        "การยกตัวอย่าง": check_examples(student_answer, example_phrases),
        "การใช้คำสรรพนามบุรษที่ 1 หรือ 2": check_pronouns(student_answer, pronouns_1_2),
        "การใช้อักษรย่อหรือคำย่อ": check_abbreviations(student_answer),
        "การเขียนชื่อเรื่อง": check_title(student_answer),
    }
    errors = [k for k, v in results.items() if k != "similarity_score" and not v]
    score = 1 if len(errors) == 0 else 0
    return score, errors, results

#----------S5 การใช้คำ/ถ้อยคำสำนวน-------------

# โหลดโมเดล mask-filling
fill_mask = pipeline("fill-mask", model="xlm-roberta-base", tokenizer="xlm-roberta-base")

# API Key
API_KEY = "pHeDDSTgNpK4jLxoHXDQsdt3b9LC5yRL"

def call_tner(text):
    url = "https://api.aiforthai.in.th/tner"
    headers = {"Apikey": API_KEY}
    try:
        response = requests.post(url, headers=headers, data={"text": text}, timeout=10)
        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                print("❌ Response ไม่ใช่ JSON:", response.text[:200])
                return {}
        else:
            print(f"❌ API Error {response.status_code}: {response.text[:200]}")
            return {}
    except requests.exceptions.RequestException as e:
        print("❌ Request error:", e)
        return {}

def find_words_by_pos(tner_result, pos_tags):
    words = []
    tokens = tner_result.get("words", [])
    pos = tner_result.get("POS", [])
    for idx, (w, p) in enumerate(zip(tokens, pos)):
        if p in pos_tags:
            words.append((idx, w))
    return words

def check_word_with_fill_mask(sentence, target_word):
    masked_sentence = sentence.replace(target_word, fill_mask.tokenizer.mask_token, 1)
    preds = fill_mask(masked_sentence)
    predicted_tokens = [p["token_str"].strip() for p in preds]
    return True, predicted_tokens

def normalize_word(w):
    return w.replace("\n", "").replace("\r", "").replace(" ", "").lower()

# โหลด dataset
file_path = r"D:\\project1\speak_words(in).csv"
spoken_words_dataset = pd.read_csv(file_path)["word"].dropna().astype(str).str.strip()
spoken_words_dataset = [w for w in spoken_words_dataset if w]  # ลบ empty string

notinlan_dataset = pd.read_csv(r"D:\\project1\\notinlan_words.csv")["notinlan"].dropna().astype(str).str.strip()
notinlan_dataset = [w for w in notinlan_dataset if w]

local_words_context = pd.read_csv(r"D:\\project1\sample_local_dialect(1)(in).csv")["local_word"].dropna().astype(str).str.strip()
local_words_context = [w for w in local_words_context if w]

spoken_words_set = set(normalize_word(w) for w in spoken_words_dataset)
notinlan_set = set(normalize_word(w) for w in notinlan_dataset)
local_dialect_set = set(normalize_word(w) for w in local_words_context)

# keyword_dict
keyword_dict = {
      "conjunctions": ["จน", "แม้มี", "แม้ว่า", "ถ้าใช้", "จึงมี", "และเรา", "ดังนั้น", "รับผิดชอบ", "ระวัง", "หากใชสือ", "ทั้งนี้",
                       "หากใช้", "แต่ปัจจุบัน", "อย่างไรก็ตาม", "อย่างไม่", "หรือ", "ถึงแม้", "ยังมี", "การ", "เพราะ", "เช่น การเตือน"],

      "prepositions": ["ในการ", "ในทางที่ดี", "ให้กับ", "ด้วย", "ของสังคม", "ของข้อมูล", "ต่อสังคม", "ยัง", "อย่างรวดเร็ว",
                       "ก็จะ", "เมื่อพบว่า", "แก่สังคม", "แก่ผู้อื่น", "ลักษณะดังกล่าว", "จากคนบางกลุ่ม", "กิจการของตน", "สื่อ",
                       "ทั่วโลก", "ทางด้าน", "ต่อง"],

      "classifiers": ["หลายครั้ง"]
}

def evaluate_student_text(student_text, keyword_dict,
                          spoken_words_set,
                          notinlan_set,
                          local_dialect_set,
                          full_score=1.0,
                          deduct_per_word=0.5):
    errors = {k: [] for k in (list(keyword_dict.keys()) + ["slang", "dialect", "invalid_word"])}
    total_wrong = 0
    penalty = 0.0

    # 1) ตรวจ POS + fill-mask
    tner_result = call_tner(student_text)
    words_list = [w.strip() for w in re.split(r'\s+', student_text.strip()) if w.strip()]
    pos_categories = {
        "conjunctions": ["CNJ"],
        "prepositions": ["P"],
        "classifiers": ["CL"]
    }

    for key, pos_list in pos_categories.items():
        words = find_words_by_pos(tner_result, pos_list)
        for idx, word in words:
            prev_word = words_list[idx-1] if 0 <= idx-1 < len(words_list) else ""
            next_word = words_list[idx+1] if 0 <= idx+1 < len(words_list) else ""

            is_valid, preds = check_word_with_fill_mask(student_text, word)

            clean_word = normalize_word(word)
            clean_prev = normalize_word(prev_word)
            clean_next = normalize_word(next_word)

            matched_keyword = None
            for kw in keyword_dict.get(key, []):
                if kw.startswith(clean_prev + clean_word + clean_next) or clean_word in kw:
                    matched_keyword = kw
                    break

            is_wrong = False
            if preds:
                if not matched_keyword and (clean_word not in preds):
                    is_wrong = True

            if is_wrong:
                total_wrong += 1

            errors[key].append({
                "word": clean_word,
                "predicted": preds,
                "prev_word": prev_word,
                "next_word": next_word,
                "matched_keyword": matched_keyword,
                "is_wrong": is_wrong
            })

    # 2) ตรวจคำภาษาพูด / ไม่มีในภาษา / คำถิ่น
    clean_text = normalize_word(student_text)

    for w in [w for w in spoken_words_set if w in clean_text]:
        errors["slang"].append({"word": w, "is_wrong": True})
        penalty += deduct_per_word

    for w in [w for w in notinlan_set if w in clean_text]:
        errors["invalid_word"].append({"word": w, "is_wrong": True})
        penalty += deduct_per_word

    for w in [w for w in local_dialect_set if w in clean_text]:
        errors["dialect"].append({"word": w, "is_wrong": True})
        penalty += deduct_per_word

    # 3) คำนวณคะแนน
    score = full_score - 0.5 * total_wrong - penalty
    score = max(min(score, full_score), 0.0)

    return {
        "errors": errors,
        "score": round(score, 2)
    }

#----------S6 การใช้ประโยค------------
# ---------------- Typhoon API ----------------
client = openai.OpenAI(
    api_key="sk-3u6WAA0DwMjJoJ2xDDxFy2ecuZDKTjUF1mCOCXAJKSlR3Xqq",
    base_url="https://api.opentyphoon.ai/v1"
)

# ---------------- ฟังก์ชัน Typhoon ----------------
def ask_typhoon(question, document):
    response = client.chat.completions.create(
        model="typhoon-v2.1-12b-instruct",
        messages=[
            {"role": "system", "content": "คุณคือผู้เชี่ยวชาญด้านภาษาไทย"},
            {"role": "user", "content": f"{question} จากประโยค:\n{document}"}
        ],
        temperature=0,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

nlp = spacy_thai.load()

# ---------------- ฟังก์ชัน Extract SVO ----------------
def extract_svo_spacythai(sentence, subject_keywords=None, object_keywords=None):
    if subject_keywords is None:
        subject_keywords = []
    if object_keywords is None:
        object_keywords = []

    doc = nlp(sentence)
    subject_list, verb_list, object_list = [], [], []

    for token in doc:
        if token.dep_ in ["nsubj", "nsubjpass", "FIXN", "CCONJ", "SCONJ"]:
            subject_list.append(token.text)
        elif token.dep_ == "ROOT" or token.pos_ == "VERB":
            verb_list.append(token.text)
        elif token.dep_ in ["obj", "dobj", "iobj"]:
            object_list.append(token.text)

    # ---------------- เช็ค special case ----------------
    first_token = doc[0] if len(doc) > 0 else None
    subject_text = ", ".join(subject_list) if subject_list else "(ไม่พบ)"
    if subject_text == "(ไม่พบ)":
        if (first_token and first_token.pos_ in ["VERB", "SCONJ", "AUX"]) or any(kw in sentence for kw in subject_keywords):
            subject_text = "(ไม่พบแต่ไม่ถือว่าผิด)"

    object_text = ", ".join(object_list) if object_list else "(ไม่พบ)"
    if object_text == "(ไม่พบ)":
        if any(kw in sentence for kw in object_keywords):
            object_text = "(ไม่พบแต่ไม่ถือว่าผิด)"

    # กรณีพิเศษ: ถ้ามี verb แต่กรรมไม่พบ → ถือว่าอนุโลม
    if subject_text == "(ไม่พบแต่ไม่ถือว่าผิด)" and verb_list and object_text == "(ไม่พบ)":
        object_text = "(ไม่พบแต่ไม่ถือว่าผิด)"

    return {
        "sentence_text": sentence,
        "subject": subject_text,
        "verb": ", ".join(verb_list) if verb_list else "(ไม่พบ)",
        "object": object_text
    }

# ---------------- ฟังก์ชันถาม Typhoon Q2 ----------------
def ask_typhoon_q2_retry(system_prompt, question, document, wait_sec=3, max_attempts=5):
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model="typhoon-v2.1-12b-instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{question}\n{document}"}
                ],
                temperature=0,
                max_tokens=1000
            )
            ans = response.choices[0].message.content.strip()
            if ans:
                return ans
            else:
                print(f"⚠️ ไม่มีผลลัพธ์จาก Typhoon Q2 (retry {attempt})")
        except Exception as e:
            print(f"⚠️ เกิดข้อผิดพลาด Q2: {e} (retry {attempt})")
        time.sleep(wait_sec)
    return "(ไม่มีคำตอบจาก Typhoon Q2)"

# ---------------- ฟังก์ชันหลัก S6 ----------------
def evaluate_sentence_usage(student_text):
    """
    ตรวจการใช้ประโยค (S6)
    - Q1: ตรวจ SVO ด้วย spaCy-Thai
    - Q2: ตรวจว่ามีประโยคไม่สื่อความหมาย (Typhoon)
    """
    sentences = sent_tokenize(student_text)
    subject_keywords = ["สื่อสังคม", "สื่อออนไลน์", "สื่อสังคมออนไลน์",
                        "สื่อ", "การ", "ตลอดจน", "ไม่ว่าจะเป็น", "ในชีวิตประจำวัน", "ทุกวัย",
                        "และ", "หรือ", "อีกทั้งยัง", "อย่างไรก็ตาม", "แต่"]

    object_keywords = ["ข้อมูลข่าวสาร", "ข้อมูล", "ข่าวสาร", "แก่สังคม", "จำนวนมาก",
                    "สื่อสังคม", "สื่อออนไลน์", "สื่อสังคมออนไลน์", "ทุกวัย"]

    svo_result = [extract_svo_spacythai(sent, subject_keywords, object_keywords) for sent in sentences]

    q2 = "จากประโยคที่ให้หาประโยคที่ไม่มีความหมาย ถ้าไม่มีตอบว่า ไม่มีประโยคที่ไม่สื่อความหมาย"
    system_q2 = "คุณคือผู้เชี่ยวชาญด้านภาษาไทย ห้ามคิดคำเอง"
    ans2 = ask_typhoon_q2_retry(system_q2, q2, student_text)

    score = 1.0  # เต็ม 1 คะแนน (ตาม rubric)

    # Q1: หัก 0.5 ต่อ S/V/O ที่ไม่พบ
    for item in svo_result:
        for key in ['subject', 'verb', 'object']:
            if "(ไม่พบ)" in item[key]:
                score -= 0.5

    # Q2: ถ้ามีประโยคไม่สื่อความหมาย → หัก 0.5
    if not ans2.strip().startswith("ไม่มี"):
        score -= 0.5

    score = max(score, 0)

    return {
        "svo_analysis": svo_result,
        "q2_result": ans2,
        "score": round(score, 2)
    }

#------------------S7 คำบอกข้อคิดเห็น --------------------
def evaluate_agreement_with_reference(answer: str, reference_text: str, threshold: float = 0.6) -> dict:
    """
    ตรวจคำบอกข้อคิดเห็น (เห็นด้วย/ไม่เห็นด้วย) + cosine similarity กับ reference_text
    ใช้ตรวจ essay_analysis
    """
    found = None
    if "ไม่เห็นด้วย" in answer:
        found = "ไม่เห็นด้วย"
    elif "เห็นด้วย" in answer:
        found = "เห็นด้วย"

    emb_answer = model.encode(answer, convert_to_tensor=True)
    emb_ref = model.encode(reference_text, convert_to_tensor=True)
    cosine_score = float(util.cos_sim(emb_answer, emb_ref)[0][0].item())

    if not found and cosine_score < threshold:
        return {
            "cosine_similarity": round(cosine_score, 3),
            "found_word": "ไม่พบ",
            "score": 0,
            "message": "ไม่มีคำบอกข้อคิดเห็น และ cosine < threshold, ไม่ตรวจทั้งข้อ"
        }

    return {
        "cosine_similarity": round(cosine_score, 3),
        "found_word": found if found else "ไม่พบ",
        "score": 1 if found else 0,
        "message": "ตรวจผ่าน"
    }

#------------------S8 เหตุผลสนับสนุนและคำบอกข้อคิดเห็น --------------------

def load_local_words_s8(file_path):
    """โหลดคำท้องถิ่น (local_word) สำหรับ S8"""
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    if "local_word" not in df.columns:
        raise ValueError("ไม่พบคอลัมน์ 'local_word' ในไฟล์")
    return [str(x).strip() for x in df["local_word"].dropna().tolist()]

local_words_s8 = load_local_words_s8(r"D:\\project1\\example_dialect (3)(in)(1).csv")

def normalize_text(words):
    text = str(words).lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def check_has_example(student_answer, local_words):
    student_text = normalize_text(student_answer)
    for w in local_words:
        w_norm = normalize_text(w)
        if w_norm in student_text:
            return True, w_norm
    return False, None

def detect_opinion(student_answer):
    """ตรวจคำบอกข้อคิดเห็น: 'ไม่เห็นด้วย' ก่อน 'เห็นด้วย'"""
    text = normalize_text(student_answer)
    if "ไม่เห็นด้วย" in text:
        return "ไม่เห็นด้วย", 1
    elif "เห็นด้วย" in text:
        return "เห็นด้วย", 1
    else:
        return "ไม่มีคำบอกข้อคิดเห็น", 0

def evaluate_student_answer8(student_answer, articles, main_ideas, local_words):
    # แปลงเป็น str เผื่อ NaN หรือ float
    student_answer = str(student_answer)

    # ตรวจคำบอกข้อคิดเห็น
    opinion, score_opinion = detect_opinion(student_answer)

    # -------------------------
    # embedding SBERT
    # -------------------------
    article_embeddings = model.encode(articles, convert_to_tensor=True)
    student_embedding = model.encode(student_answer, convert_to_tensor=True)
    main_idea_embeddings = model.encode(main_ideas, convert_to_tensor=True)

    cosine_scores_articles = util.cos_sim(student_embedding, article_embeddings)
    cosine_scores_main_ideas = util.cos_sim(student_embedding, main_idea_embeddings)

    # -------------------------
    # ตรวจคัดลอกบทความ
    # -------------------------
    similarity_threshold_article = 0.87
    copied_article = any(score.item() >= similarity_threshold_article for score in cosine_scores_articles[0])

    # -------------------------
    # ตรวจใจความสำคัญ
    # -------------------------
    similarity_threshold_main_idea = 0.55
    covered_main_idea_flags = [score.item() >= similarity_threshold_main_idea for score in cosine_scores_main_ideas[0]]
    has_main_idea = any(covered_main_idea_flags)

    # -------------------------
    # ตรวจการยกตัวอย่างจาก local_word
    # -------------------------
    has_example, found_example_word = check_has_example(student_answer, local_words)

    # -------------------------
    # ใช้ SSense ตรวจ sentiment
    # -------------------------
    url_ssense = "https://api.aiforthai.in.th/ssense"
    params = {"text": student_answer}
    headers_ssense = {"Apikey": "zyHC3BNtLiesIuTj2UMlQd8DhrVXBxzM"}
    try:
        response_sentiment = requests.get(url_ssense, headers=headers_ssense, params=params, timeout=10)
        sentiment_result = response_sentiment.json()
    except:
        sentiment_result = {"polarity-pos": False, "polarity-neg": True, "score": 50}

    # -------------------------
    # นับจำนวนบรรทัด
    # -------------------------
    line_count = student_answer.count("\n") + 1

    # -------------------------
    # ตรวจ keyword ให้คะแนน
    # -------------------------
    keyword_words_2 = ["เพราะ"]
    has_keyword_2 = any(k in student_answer for k in keyword_words_2)

    # ✅ ดึงคำหลัง "เพราะ"
    after_keyword_2 = "0"
    match = re.search(r"เพราะ\s*(.*)", student_answer)
    if match:
        text_after = match.group(1).strip()
        words = text_after.split()
        after_keyword_2 = " ".join(words[:10]) if words else "0"

    keyword_words_0 = ["เป็นสื่อหรือช่องทางที่แพร่กระจายข้อมูลข่าวสารในรูปแบบต่างๆ", "ตกเป็นเหยื่อล่อ"]
    has_keyword_0 = any(k in student_answer for k in keyword_words_0)

    keyword_words_2_donthave = ["จำเป็น", "สิ่งที่ดี", "ข้อดี", "ป้องกัน"]
    has_keyword_2_donthave = any(k in student_answer for k in keyword_words_2_donthave)

    keyword_words_2_donthave_de = ["โกง", "ฟ้อง", "ไม่น่าเชื่อถือ", "ป้องกัน", "เครื่องมือ",
                                  "โดนชักจูง", "ถูกกระทำ", "เป็นภัย", "มีความผิด",
                                  "ทางที่เหมาะสม","หลอกเอาเงิน", "คนที่ทำไม่ดี",
                                  "ของปลอม", "ไม่มีความรับผิดชอบ", "การเตือนเรื่องภัย"]
    has_keyword_2_donthave_de = any(k in student_answer for k in keyword_words_2_donthave_de)

    keyword_words_4 = ["เจตนาแอบแฝง", "เจตนา", "แอบแฝง", "มิจฉาชีพ", "ผ่อนคลาย",
                      "ผิดกฎหมาย",  "พนัน", "หลอก", "จำคุก", "โฆษณาชวนเชื่อ",
                      "ถูกหมายเรียก", "ติดโรค", "เจาะข้อมูล"]
    has_keyword_4 = any(k in student_answer for k in keyword_words_4)

    keyword_words_6 = ["การใช้สื่อสังคมออนไลน์ในทางที่ผิด", "โดนมิจฉาชีพหลอก", "ตกเป็นเหยื่อของสังคมออนไลน์",
                       "การพนันหรือขายของผิดกฏหมาย", "การตลาดทั้งในธุรกิจ สังคม และการเมือง",
                       "สินค้าพื้นเมือง", "การโฆษณาสินค้า", "ขัดต่อศีลธรรมอันดีของสังคม",
                       "การเตือนภัยให้แก่คนในสังคม" , "เขียนวิจารณ์", "รู้เท่าทันสื่อออนไลน์",
                       "กลลวงมิจฉาชีพ", "วิจารณ์ในทางที่เสียหาย", "รับชมข่าวสาร",
                       "ประกอบอาชีพ", "การส่งข้อความ"]
    has_keyword_6 = any(k in student_answer for k in keyword_words_6)

    keyword_words_8 = ["ไลฟ์สด" , "ดูหนัง", "การศึกษา", "เปิดเพลงฟัง", "แพลตฟอร์มออนไลน์",
                      "ไลฟ์", "เพลง", "โรงงาน", "ท่องเที่ยว", "การสั่งของจากสื่อ", "โอนเงิน", "พัฒนาตนเอง"]
    has_keyword_8 = any(k in student_answer for k in keyword_words_8)

    # -------------------------
    # ให้คะแนนตามเงื่อนไข
    # -------------------------
    polarity_pos = sentiment_result.get("polarity-pos", False)
    polarity_neg = sentiment_result.get("polarity-neg", True)

    if sentiment_result.get("polarity", "") == "positive" or (polarity_pos and not polarity_neg):
        sentiment = "pos"
    else:
        sentiment = "neg"

    score_total = 0
    if (
        (opinion == "ไม่มีคำบอกข้อคิดเห็น" and has_keyword_2 and after_keyword_2 == "0" and not has_main_idea)
        or (opinion == "ไม่มีคำบอกข้อคิดเห็น" and not has_keyword_2 and after_keyword_2 == "0"  and has_main_idea)
        or (opinion == "ไม่มีคำบอกข้อคิดเห็น" and has_keyword_2 and after_keyword_2 == "0" and not has_main_idea and not has_example)
        or (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"]  and not has_main_idea and not has_keyword_4 and not has_example and  found_example_word is None and not has_keyword_8 and not has_keyword_6 and not has_keyword_2_donthave_de and not has_keyword_2_donthave)
        or (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"] and after_keyword_2 == "0" and not has_keyword_4 and not has_example and  found_example_word is None and not has_keyword_8 and not has_keyword_6 and not has_keyword_2_donthave_de and not has_keyword_2_donthave)
        or (has_keyword_0)
    ):
        score_total = 0

    elif ((opinion == "ไม่เห็นด้วย" and has_keyword_2_donthave) or (opinion == "เห็นด้วย" and has_keyword_2_donthave_de)) and has_main_idea and sentiment == "neg" and not copied_article:
        score_total = 2

    elif (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"] and has_example and not has_keyword_6 and has_keyword_2 and has_keyword_8):
        score_total = 8

    elif (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"] and (copied_article or has_example) and has_main_idea and not has_keyword_8 and has_keyword_6):
        score_total = 6

    elif (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"] and (not found_example_word or not has_example or has_keyword_4)):
        score_total = 4

    elif ((opinion == "ไม่มีคำบอกข้อคิดเห็น" and not copied_article and sentiment == "neg" and has_main_idea) or (opinion == "ไม่มีคำบอกข้อคิดเห็น" and not copied_article and sentiment == "neg" and has_main_idea and not has_keyword_2)):
        score_total = 2

    elif ((opinion in ["เห็นด้วย", "ไม่เห็นด้วย"] and not copied_article and not has_main_idea and not found_example_word and not has_example)):
        score_total = 2

    elif ((opinion == "ไม่มีคำบอกข้อคิดเห็น" and copied_article) or (opinion == "ไม่มีคำบอกข้อคิดเห็น" and not has_main_idea) or (opinion in ["เห็นด้วย", "ไม่เห็นด้วย"] and not has_main_idea)):
        score_total = 0

    return {
        "opinion": opinion,
        "score_opinion": score_opinion,
        "has_example": has_example,
        "found_example_word": found_example_word,
        "sentiment": sentiment_result,
        "copied_article": copied_article,
        "has_main_idea": has_main_idea,
        "has_keyword_2": has_keyword_2,
        "after_keyword_2": after_keyword_2,
        "has_keyword_2_donthave": has_keyword_2_donthave,
        "has_keyword_2_donthave_de": has_keyword_2_donthave_de,
        "has_keyword_4": has_keyword_4,
        "has_keyword_6": has_keyword_6,
        "has_keyword_8": has_keyword_8,
        "line_count": line_count,
        "score_total": score_total
    }


#------------------S9 การเรียงลำดับ --------------------
ignore_list_s9 = ["สื่อ", "สื่อออนไลน์", "สื่อสังคม", "สื่อสังคมออนไลน์",
               "ออนไลน์", "ออนไลท์", "\n", "ในบ้าน", "ในรูปแบบ",
               "การใช้", "เด็กชายA", "ที่ดี", "ให้แก่", "หากเรา",
               "ได้อย่าง", "ในการ", "เราก็", "อย่างรวดเร็ว", "ในทาง",
               "ได้ด้วย", "ก็มี", "ที่ไม่", "หรือจะ", "ในปัจจุบัน", "สามารถดู",
               "ก็ไม่", "เช่นการ", "สามารถใช้", "ใช้เพื่อ", "ควรใช้",
               "ของตน", "ที่", "ไม่ควร", "จึง", "ควร", "เรา", "ใช้อย่าง", "ชาชีพ",
               "นั้นสามารถ", "อีกมากมาย", "นั้นใช้", "การนำเสนอ", "ก็จะ", "ถ้าใช้",
               "อยากจะ", "ก็ทำได้", "สังคมเป็น", "ช่วยให้", "เข้าถึงได้", "หรือ",
               "อย่างมาก", "ในทุกวันนี้", "ต่างๆ", "เพราะ", "อาจ", "เข้ามาช่วยทำงานในด้าน" ,
               "ช่วย","ใน", "ตรงไปตรงมา", "เกิดประโยชน์", "แก่สังคม", "จะทำให้", "สือ",
               "มีการ", "ไม่รู้","ความ", "ไม่น้อย", "คือ", "ไหน", "แก่", "ตนเอง",
               "ซื้อของ", "ตรวจสอบว่า","ได้ง่าย", "ค้นคว้า", "ข้อมูล", "กลุ่มแชท",
               "การเล่น", "ด้วยเช่นกัน", "ให้กับ", "และ", "มีทั้ง", "เช่น", "คน",
               "ปรโยชน์", "อะไรได้", "สมัยนี้มี", "สังคมส่วนรวม", "ตามโฆษณา", "ไม่มี",
               "ให้ดี", "ทำอะไร", "เค้า", "ซี", "แอป", "ผิดการ",
               "มีด้าน", "จะได้", "ได้ไม่ต้อง", "โดยไม่", "เป็นสิ่ง", "ไม่ดี",
               "คุยกับเพื่อน", "บางกลุ่ม", "โทษต่อ", "อย่างระมัดระวัง",
               "คุย", "เพื่อน", "โพช", "เช็กให้"]

specific_terms_s9 = []

ignore_single_char_s9 = ["สิ", "สี่", "สัญญา", "ผิดก", "หริ", "รู", "ภูมิ",
                      "เจ", "คา", "เป้", "เสีย", "หาย", "ผิด", "ที",
                      "สี" , "ริ" , "ข่อ" , "ออนไลน์", "โท", "ต่อ", "ใส",
                      "ข่า", "แอ", "ยุค", "หลาย", "ฮิต", "ทู",
                      "บุ", "ยี่", "มาก", "ทำได้", "ตน", "เขา",
                      "ควร", "ดัก", "กรู", "กลุ่ม", "เด็ก", "โคล", "มั่ว", "คน",
                      "อย่า", "รู้", "รี", "โพ", "เฉย", "เยอ", "ดี", "ติ",
                      "ลื่อ", "ดี", "เทคโนโลยี", "กุ", "ผู้อื่น", "เสียหาย",
                      "สิ้น", "ค้น", "คอ", "หลอ"]  # คำเดี่ยวที่ไม่ถือว่าผิด


def evaluate_ordering_and_coherence(student_text,
                                    ignore_list_s9=None,
                                    specific_terms_s9=None,
                                    ignore_single_char_s9=None,
                                    similarity_threshold=0.3):
    if ignore_list_s9 is None:
        ignore_list_s9 = []
    if specific_terms_s9 is None:
        specific_terms_s9 = []
    if ignore_single_char_s9 is None:
        ignore_single_char_s9 = []

    # ตรวจคำเดี่ยว
    single_char_words, special_violations = check_thai_text_integrity(student_text, ignore_single_char_s9)
    missing_content = {
        "single_char_words": single_char_words,
        "special_violations": special_violations
    }

    # ตรวจคำซ้ำ
    s_clean = student_text.replace("\n", "")
    tokens = [t for t in word_tokenize(s_clean, keep_whitespace=False) if t.strip()]
    repeated_ngrams_result = find_repeated_ngrams(tokens, min_len=2, ignore_list=ignore_list_s9)
    specific_found_result = find_specific_terms(s_clean, specific_terms_s9)

    duplicate_content = {
        "repeated_ngrams": repeated_ngrams_result["repeated_ngrams"],
        "specific_found": specific_found_result["specific_found"]
    }

    # ตรวจความสัมพันธ์ระหว่างบรรทัด
    failed_similarity = semantic_similarity_lines(student_text, threshold=similarity_threshold)

    # คำนวณคะแนน
    score = 3
    if single_char_words:
        score -= 1
    if special_violations:
        score -= 1
    score -= repeated_ngrams_result["count"]
    score -= specific_found_result["count"]
    if failed_similarity:
        score -= 1
    score = max(0, score)

    return {
        "เนื้อความขาด": missing_content,
        "เนื้อความซ้ำ": duplicate_content,
        "เนื้อความไม่สัมพันธ์กัน": failed_similarity,
        "คะแนนรวม": score
    }



#------------------S10 ความถูกต้องตามหลักการเขียนแสดงความคิดเห็น --------------------
TNER_API_KEY = 'pHeDDSTgNpK4jLxoHXDQsdt3b9LC5yRL'
CYBERBULLY_API_KEY = 'pHeDDSTgNpK4jLxoHXDQsdt3b9LC5yRL'

personal_pronoun_1 = {"หนู", "ข้า", "กู"}
personal_pronoun_2 = {"คุณ", "แก", "เธอ", "ตัวเอง", "เอ็ง", "มึง"}
all_personal_pronouns = personal_pronoun_1.union(personal_pronoun_2)

def check_named_entities(text):
    url = "https://api.aiforthai.in.th/tner"
    headers = {"Apikey": TNER_API_KEY}
    data = {"text": text}
    try:
        response = requests.post(url, headers=headers, data=data, timeout=8)
        if response.status_code == 200:
            ner_result = response.json()
            bad_tags = {'ABB_DES', 'ABB_TTL', 'ABB_ORG', 'ABB_LOC', 'ABB'}
            bad_entities = [ent['word'] for ent in ner_result.get("entities", []) if ent['tag'] in bad_tags]
            if bad_entities:
                return True, bad_entities
    except:
        pass
    return False, []

def check_cyberbully(text):
    url = "https://api.aiforthai.in.th/cyberbully"
    headers = {"Apikey": CYBERBULLY_API_KEY}
    data = {"text": text}
    try:
        response = requests.post(url, headers=headers, data=data, timeout=8)
        if response.status_code == 200:
            result = response.json()
            if result.get("bully", "no") == "yes":
                bully_words = result.get("bully_words") or result.get("bully_phrases") or [text]
                return True, bully_words
    except:
        pass
    return False, []

def check_personal_pronouns(text):
    tokens = word_tokenize(text, engine="newmm")
    found_pronouns = [token for token in tokens if token in all_personal_pronouns]
    if found_pronouns:
        return True, found_pronouns
    return False, []

def evaluate_comment_validity(text):
    """
    ตรวจความถูกต้องของการแสดงความคิดเห็น
    - ห้ามมีชื่อเฉพาะ
    - ห้ามใช้คำ bully
    - ห้ามใช้สรรพนามบุรุษที่ 1/2
    """
    mistakes = []
    mistake_count = 0

    ne_flag, ne_words = check_named_entities(text)
    if ne_flag:
        mistake_count += 1
        mistakes.append(f"มีชื่อเฉพาะ/ตัวย่อ: {', '.join(ne_words)}")

    bully_flag, bully_words = check_cyberbully(text)
    if bully_flag:
        mistake_count += 1
        mistakes.append(f"ข้อความลักษณะ Cyberbully: {', '.join(bully_words)}")

    pronoun_flag, pronouns = check_personal_pronouns(text)
    if pronoun_flag:
        mistake_count += 1
        mistakes.append(f"ใช้สรรพนามบุรุษที่ 1 หรือ 2: {', '.join(pronouns)}")

    if mistake_count == 0:
        score = 2
    elif mistake_count == 1:
        score = 1
    else:
        score = 0

    return {
        "score": score,
        "details": mistakes if mistakes else ["ไม่มีข้อผิดพลาด"]
    }

#------------------S11 การสะกดคำ (ข้อ 2) --------------------
with open(r'D:\\project1\\thai_loanwords_new_update(1) (3).json', 'r', encoding='utf-8') as f:
     loanwords_data = json.load(f)
     loanwords_whitelist = set(item['thai_word'] for item in loanwords_data)

with open(r'D:\\project1\\update_common_misspellings (1) (1).json', 'r', encoding='utf-8') as f:
    raw_data = f.read()

# แทนที่ NaN ด้วย null
raw_data = raw_data.replace('NaN', 'null')
data = json.loads(raw_data)
# สร้าง dict
COMMON_MISSPELLINGS = {item['wrong']: item.get('right') for item in data}


with open(r"D:\\project1\splitable_phrases (1).json", "r", encoding="utf-8") as f:
    splitable_phrases = set(json.load(f))

API_KEY = '33586c7cf5bfa0029887a9981bf94963'
API_URL = 'https://api.longdo.com/spell-checker/proof'

custom_words = {"ประเทศไทย", "สถาบันการศึกษา", "นานาประการ"}
strict_not_split_words = {'มากมาย', 'ประเทศไทย', 'ออนไลน์', 'ความคิดเห็น', 'ความน่าเชื่อถือ'}
thai_dict = set(w for w in set(thai_words()).union(custom_words) if (' ' not in w) and w.strip())

allowed_punctuations = ['.', ',', '-', '(', ')', '!', '?', '%', '“', '”', '‘', '’', '"', "'", '…', 'ฯ']
allow_list = {'ปี', 'อื่น', 'เล็ก', 'ใหญ่', 'มาก', 'หลาย', 'ช้า', 'เร็ว', 'ชัด', 'ดี', 'ผิด' ,'เสีย', 'หาย','สวย','มั่ว','ง่าย'}
forbid_list = {'นา', 'บางคน', 'บางอย่าง', 'บางสิ่ง', 'บางกรณี'}
splitable_pairs = {("ไป", "มา"),("ได้","ส่วน"),("ดี","แต่")}
allowed_phrases = ["ฯลฯ"]

# ---------------------------
# Helper Functions
# ---------------------------

def is_english_or_number(word: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9().,\-_/]+", word.strip()))

def longdo_spellcheck_batch(words):
    results = {}
    headers = {"Content-Type": "application/json"}
    for word in words:
        try:
            payload = {"text": word, "api": API_KEY}
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                data = response.json()
                suggestions = []
                if "words" in data and data["words"]:
                    for w in data["words"]:
                        if "candidates" in w:
                            suggestions.extend(c["text"] for c in w["candidates"])
                results[word] = suggestions
            else:
                results[word] = []
        except:
            results[word] = []
    return results

def pythainlp_spellcheck(tokens, pos_tags, dict_words, ignore_words):
    mistakes = []
    for i, token in enumerate(tokens):
        if not token or is_english_or_number(token):
            continue
        if token in ignore_words:
            continue
        if token in dict_words:
            continue
        suggestions = spell(token)
        if not suggestions:
            mistakes.append({'word': token, 'pos': pos_tags[i][1] if i < len(pos_tags) else None,
                             'index': i, 'suggestions': suggestions})
    return mistakes

def check_common_misspellings_before_tokenize(text, misspelling_dict):
    errors = []
    for wrong, right in misspelling_dict.items():
        if wrong in text:
            for m in re.finditer(re.escape(wrong), text):
                errors.append({"word": wrong, "index": m.start(), "right": right})
    return errors

def check_loanword_before_tokenize(words, whitelist):
    mistakes = []
    for i, w in enumerate(words):
        if is_english_or_number(w):
            continue
        matches = difflib.get_close_matches(w, list(whitelist), n=1, cutoff=0.7)
        if matches and w not in whitelist:
            mistakes.append({'word': w, 'index': i, 'suggestions': [matches[0]]})
    return mistakes

def find_unallowed_punctuations(text):
    for phrase in allowed_phrases:
        text = text.replace(phrase, "")
    pattern = f"[^{''.join(re.escape(p) for p in allowed_punctuations)}a-zA-Z0-9ก-๙\\s]"
    return list(set(re.findall(pattern, text)))

def analyze_maiyamok(tokens, pos_tags):
    results = []
    found_invalid = False
    VALID_POS = {'NCMN', 'NNP', 'VACT', 'VNIR', 'CLFV', 'ADVN', 'ADVI', 'ADVP', 'PRP', 'ADV'}
    for i, token in enumerate(tokens):
        if token == 'ๆ':
            prev_idx = i - 1
            prev_word = tokens[prev_idx] if prev_idx >= 0 else None
            prev_tag = pos_tags[prev_idx][1] if prev_idx >= 0 else None
            if prev_word is None or prev_word == 'ๆ':
                verdict = "❌ ไม้ยมกไม่ควรขึ้นต้นประโยค/คำ"
            elif prev_word in forbid_list:
                verdict = '❌ ไม่ควรใช้ไม้ยมกกับคำนี้'
            elif (prev_tag in VALID_POS) or (prev_word in allow_list):
                verdict = '✅ ถูกต้อง (ใช้ไม้ยมกซ้ำคำได้)'
            else:
                verdict = '❌ ไม่ควรใช้ไม้ยมกนอกจากกับคำนาม/กริยา/วิเศษณ์'
            context = tokens[max(0, i-2):min(len(tokens), i+3)]
            results.append({'คำก่อนไม้ยมก': prev_word or '',
                            'POS คำก่อน': prev_tag or '',
                            'บริบท': ' '.join(context),
                            'สถานะ': verdict})
            if verdict.startswith('❌'): found_invalid = True
    return results, found_invalid

def detect_split_errors(tokens, custom_words=None, splitable_phrases=None):
    check_dict = set(thai_words()).union(custom_words or [])
    check_dict = {w for w in check_dict if (' ' not in w) and w.strip()}
    splitable_phrases = splitable_phrases or set()
    errors = []
    for i in range(len(tokens) - 1):
        combined = tokens[i] + tokens[i + 1]
        if (combined in check_dict) and ((tokens[i], tokens[i+1]) not in splitable_pairs):
            errors.append({"split_pair": (tokens[i], tokens[i+1]), "suggested": combined})
    return errors

# ---------------------------
# ฟังก์ชันหลัก S11
# ---------------------------
def evaluate_text_s11(text):
    if not text or not text.strip():
        return {'score': 0.0, 'reasons': ["ไม่มีคำตอบ"], 'total_error_count': 0}

    corrected_text = normalize(text.replace("\n", ""))

    json_misspells = check_common_misspellings_before_tokenize(corrected_text, COMMON_MISSPELLINGS)
    tokens = [t for t in word_tokenize(corrected_text, engine='newmm', keep_whitespace=False)
              if not is_english_or_number(t)]
    pos_tags = pos_tag(tokens, corpus='orchid')

    pythai_errors = pythainlp_spellcheck(tokens, pos_tags, dict_words=thai_dict, ignore_words=custom_words)

    raw_words = re.findall(r'[ก-๙]+', corrected_text)
    loanword_errors = check_loanword_before_tokenize(raw_words, loanwords_whitelist) \
                    + check_loanword_before_tokenize(tokens, loanwords_whitelist)

    longdo_results = longdo_spellcheck_batch([e['word'] for e in pythai_errors])
    longdo_errors = [{**e, 'suggestions': [str(s) for s in longdo_results.get(e['word'], []) if s]}
                     for e in pythai_errors]

    all_spelling_errors = longdo_errors + \
        [{"word": e["word"], "pos": None, "index": e["index"], "suggestions": [str(e["right"])]} for e in json_misspells] + \
        [{"word": e["word"], "pos": None, "index": e["index"],
          "suggestions": [str(s) for s in e.get("suggestions", []) if s]} for e in loanword_errors]

    punct_errors = find_unallowed_punctuations(corrected_text)
    maiyamok_results, _ = analyze_maiyamok(tokens, pos_tags)
    split_errors = detect_split_errors(tokens, custom_words=custom_words)

    total_errors = len({e["word"] for e in all_spelling_errors}) \
                 + len(split_errors) \
                 + len(punct_errors) \
                 + sum(1 for r in maiyamok_results if r['สถานะ'].startswith('❌'))

    if total_errors == 0: score = 2.0
    elif total_errors == 1: score = 1.5
    elif total_errors == 2: score = 1.0
    elif total_errors == 3: score = 0.5
    else: score = 0.0

    return {
        'spelling_errors': all_spelling_errors,
        'loanword_errors': loanword_errors,
        'punctuation_errors': punct_errors,
        'maiyamok_results': maiyamok_results,
        'split_errors': split_errors,
        'score': score,
        'total_error_count': total_errors
    }


#------------------S12 การใช้คำ/ถ้อยคำสำนวน (ข้อ 2) --------------------

# Dataset สำหรับ S12 (ข้อ 2)
spoken_words_dataset_s12 = pd.read_csv(r"D:\\project1\dataset_speak_word(in).csv")["word"].dropna().astype(str).str.strip()
spoken_words_dataset_s12 = [w for w in spoken_words_dataset_s12 if w]  # ลบ empty string

notinlan_dataset_s12 = pd.read_csv(r"D:\\project1\dataset_notinlan_words(in).csv")["notinlan"].dropna().astype(str).str.strip()
notinlan_dataset_s12 = [w for w in notinlan_dataset_s12 if w]

local_words_context_s12 = pd.read_csv(r"D:\\project1\S12_sample_local_dialect(1)(in)(in).csv")["local_word"].dropna().astype(str).str.strip()
local_words_context_s12 = [w for w in local_words_context_s12 if w]

spoken_words_set_s12 = set(normalize_word(w) for w in spoken_words_dataset_s12)
notinlan_set_s12 = set(normalize_word(w) for w in notinlan_dataset_s12)
local_dialect_set_s12 = set(normalize_word(w) for w in local_words_context_s12)

# keyword_dict สำหรับ S12 (ข้อ 2)
keyword_dict_s12 = {
      "conjunctions": ["จน", "แม้มี", "แม้ว่า", "ถ้าใช้", "จึงมี", "และเรา", "ดังนั้น", "รับผิดชอบ", "ระวัง", "วันต่อมา", "เช่น การเตือนภัย",
                       "หากใช้", "แต่ปัจจุบัน", "อย่างไรก็ตาม", "อย่างไม่", "หรือ", "ถึงแม้", "ยังมี", "การ", "เพราะ",
                       "ในทั้งนี้", "เพื่อเป็น", "เพื่อความ", "แม้แต่", "ตลอดจน", "ถ้าหาก", "เพราะฉะนั้น", "ส่วน", "บุคคล", "เข้าถึง",
                       "ข้อมูลข่าวสาร", "ทำให้", "ไม่งั้น", "โทษแก่สังคม", "หลายคน", "แก้ปัญหา",
                       "สะดวก", "สามารถ", "ระดม", "การสร้าง", "จะเป็น", "ไปเจอ", "จำเป็น", "ค้าขาย",
                       "ตัวอย่าง", "หรือติ"],

      "prepositions": ["ในการ", "ในทางที่ดี", "ให้กับ", "ด้วย", "ของสังคม", "ของข้อมูล", "ต่อสังคม", "ยัง", "อย่างรวดเร็ว",
                       "ก็จะ", "เมื่อพบว่า", "แก่สังคม", "แก่ผู้อื่น", "ลักษณะดังกล่าว", "จากคนบางกลุ่ม", "กิจการของตน", "สื่อ",
                       "ทั่วโลก", "ทางด้าน", "ต่อง", "รวมถึง", "เกี่ยวกับ", "พอ", "โรงงาน", "ฉะนั้น", "เป็นโทษ",
                       "ช่วยเหลือ", "เดือดร้อน", "มีการ", "อีกด้วย", "ชวนเชื่อ", "ก่อนหลงเชื่อ", "ไม่ตรง",
                       "แต่หลายคน", "ชีวิตประจำวัน", "สะดวก", "ข้อมูลข่าวสาร", "ติดต่อผ่าน", "สำหรับ",
                       "เรื่องต่างๆ", "เรียกได้เลย", "เข้าถึง", "กันเป็นจำนวนมาก", "การขาย", "การสื่อสาร",
                       "ทุกคน", "ส่วนใหญ่", "ตัวเอง", "ทางการ", "แทนการ"],

      "classifiers": ["หลายครั้ง", "บางคน", "ทุกวัน", "เหล่านี้", "อย่างมาก", "สำคัญ", "ไม่ดี", "อาจจะ",
                      "ไม่ว่าจะเป็น", "เสียใจมาก", "เข้าถึง", "ได้เลย"]
}

def evaluate_student_text_s12(student_text, keyword_dict_s12 ,
                              spoken_words_set_s12,
                              notinlan_set_s12,
                              local_dialect_set_s12,
                              full_score=2.0,
                              deduct_per_word=0.5):
    """
    ตรวจคำ/ถ้อยคำสำนวน (S12 สำหรับข้อ 2)
    """
    errors = {k: [] for k in (list(keyword_dict.keys()) + ["slang", "dialect", "invalid_word"])}
    total_wrong = 0
    penalty = 0.0

    # 1) ตรวจ POS + fill-mask
    tner_result = call_tner(student_text)
    words_list = [w.strip() for w in re.split(r'\s+', student_text.strip()) if w.strip()]
    pos_categories = {
        "conjunctions": ["CNJ"],
        "prepositions": ["P"],
        "classifiers": ["CL"]
    }

    for key, pos_list in pos_categories.items():
        words = find_words_by_pos(tner_result, pos_list)
        for idx, word in words:
            prev_word = words_list[idx-1] if 0 <= idx-1 < len(words_list) else ""
            next_word = words_list[idx+1] if 0 <= idx+1 < len(words_list) else ""

            _, preds = check_word_with_fill_mask(student_text, word)

            clean_word = normalize_word(word)
            clean_prev = normalize_word(prev_word)
            clean_next = normalize_word(next_word)

            matched_keyword = None
            for kw in keyword_dict.get(key, []):
                if kw.startswith(clean_prev + clean_word + clean_next) or clean_word in kw:
                    matched_keyword = kw
                    break

            is_wrong = False
            if preds:
                if not matched_keyword and (clean_word not in preds):
                    is_wrong = True

            if is_wrong:
                total_wrong += 1

            errors[key].append({
                "word": clean_word,
                "predicted": preds,
                "prev_word": prev_word,
                "next_word": next_word,
                "matched_keyword": matched_keyword,
                "is_wrong": is_wrong
            })

    # 2) ตรวจคำภาษาพูด / ไม่มีในภาษา / คำถิ่น
    clean_text = normalize_word(student_text)

    for w in [w for w in spoken_words_set if w in clean_text]:
        errors["slang"].append({"word": w, "is_wrong": True})
        penalty += deduct_per_word

    for w in [w for w in notinlan_set if w in clean_text]:
        errors["invalid_word"].append({"word": w, "is_wrong": True})
        penalty += deduct_per_word

    for w in [w for w in local_dialect_set if w in clean_text]:
        errors["dialect"].append({"word": w, "is_wrong": True})
        penalty += deduct_per_word

    # 3) คำนวณคะแนน
    score = full_score - 0.5 * total_wrong - penalty
    score = max(min(score, full_score), 0.0)

    return {
        "errors": errors,
        "score": round(score, 2)
    }

# ---------------- S13 การใช้ประโยค ----------------
def evaluate_reasoning_usage(student_text,
                             subject_keywords=None,
                             object_keywords=None,
                             full_score=2.0,
                             deduct_per_word=0.5):
    """
    S13 - การใช้เหตุผลประกอบการแสดงความคิดเห็น
    Q1: ตรวจ SVO
    Q2: ตรวจว่ามีประโยคที่ไม่สื่อความหมาย
    """
    # กำหนดค่า default ถ้าไม่ได้ส่งเข้ามา
    if subject_keywords is None:
        subject_keywords = [
            "สื่อสังคม", "สื่อออนไลน์", "สื่อสังคมออนไลน์", "สื่อ", "การ",
            "ที่", "และ", "เห็นด้วย", "ไม่เห็นด้วย", "ต่าง ๆ", "ไม่",
            "ทุกวัย", "ทุกเพศ", "เป็นต้น"
        ]
    if object_keywords is None:
        object_keywords = [
            "ข้อมูลข่าวสาร", "ข้อมูล", "ข่าวสาร",
            "สื่อสังคม", "สื่อออนไลน์", "สื่อสังคมออนไลน์",
            "เห็นด้วย", "ไม่เห็นด้วย", "มีกันหมด", "ทุกเพศทุกวัย"
        ]

    # ---------------- Q1 ----------------
    sentences = sent_tokenize(student_text)
    svo_result = [extract_svo_spacythai(sent, subject_keywords, object_keywords) for sent in sentences]

    # ---------------- Q2 ----------------
    q2 = "จากประโยคที่ให้หาประโยคที่ไม่มีความหมาย ถ้าไม่มีตอบว่า ไม่มีประโยคที่ไม่สื่อความหมาย"
    system_q2 = "คุณคือผู้เชี่ยวชาญด้านภาษาไทย ห้ามคิดคำเอง"
    ans2 = ask_typhoon_q2_retry(system_q2, q2, student_text)

    # ---------------- คะแนน ----------------
    score = full_score

    for item in svo_result:
        for key in ['subject', 'verb', 'object']:
            if "(ไม่พบ)" in item[key]:
                score -= deduct_per_word

    if not ans2.strip().startswith("ไม่มี"):
        score -= deduct_per_word

    score = max(score, 0)

    return {
        "svo_analysis": svo_result,
        "q2_result": ans2,
        "score": round(score, 2)
    }



# ✅ -----------------------------
# ฟังก์ชันหลัก เรียกจาก FastAPI
# ✅ -----------------------------
def convert_numpy_to_python(obj):
    """แปลงค่า numpy หรือ set ให้ JSON-safe"""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(x) for x in obj]
    elif isinstance(obj, set):
        return [convert_numpy_to_python(x) for x in obj]  # set → list
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

# ==========================
# ฟังก์ชันหลัก: ตรวจใจความ + สะกดคำ + เรียงลำดับ/เชื่อมโยง + การใช้ประโยค
# ==========================
def evaluate_single_answer(answer_text, essay_analysis):
    # ---------------------------
    # ✅ ข้อที่ 1 : ตรวจจาก answer_text
    # ---------------------------

    # 1) ใจความสำคัญ
    student_emb = model.encode(answer_text, convert_to_tensor=True)
    core_sentences = [
        "สื่อสังคมหรือสื่อออนไลน์หรือสื่อสังคมออนไลน์เป็นช่องทางที่ใช้ในการเผยแพร่หรือค้นหาหรือรับข้อมูลข่าวสาร",
        "การใช้สื่อสังคมหรือสื่อออนไลน์หรือสื่อสังคมออนไลน์อย่างไม่ระมัดระวังหรือขาดความรับผิดชอบจะเกิดโทษหรือผลเสียหรือข้อเสียหรือผลกระทบหรือสิ่งไม่ดี",
        "ผู้ใช้ต้องรู้ทันหรือรู้เท่าทันสื่อสังคมออนไลน์",
        "การใช้สื่อสังคมหรือสื่อออนไลน์หรือสื่อสังคมออนไลน์ด้วยเจตนาแอบแฝงมีผลกระทบต่อความน่าเชื่อถือของข้อมูลข่าวสาร"
    ]
    core_embs = model.encode(core_sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(student_emb, core_embs)[0]
    best_score = float(cosine_scores.max().item())

    mind_score = evaluate_mind_score(answer_text)
    mind_total = int(mind_score.get("คะแนนรวมใจความ", 0))

    # 🔹 เงื่อนไข (1): ถ้าใจความสำคัญเป็น 0 หรือ cosine >= 0.9 ⇒ S1–S6 = 0 ทั้งหมด
    if mind_total == 0 or best_score >= 0.9:
        mind_score = {
            "cosine_similarity": round(best_score, 3), 
            **mind_score, "คะแนนรวมใจความ": mind_total, 
            "bert_score": round(best_score, 3),
            "message": "ใจความเป็น 0 หรือ cosine >= 0.9 → S1–S6 = 0 ทั้งหมด"
        }
        ordering1_score, ordering1_details = 0, {}
        summary1_score, summary1_details = 0, {}
        spelling_score, spelling_res = 0, {}
        score_s5, s5_result = 0, {}
        score_s6, s6_result = 0, {}
    else:
        # 2) เรียงลำดับความคิด (S2)
        ordering1_result = evaluate_student_answer(
            answer_text,
            ignore_list=ignore_list,
            specific_terms=specific_terms,
            ignore_single_char=ignore_single_char,
            similarity_threshold=0.3
        )
        ordering1_score = int(ordering1_result.get("คะแนนรวม", 0))
        ordering1_details = convert_numpy_to_python(ordering1_result)

        # 3) ความถูกต้องตามหลักการย่อความ
        summary1_score, summary1_err, summary1_details = validate_student_answer(answer_text)
        summary1_score = int(summary1_score)
        summary1_details = convert_numpy_to_python(summary1_details)

        # 4) การสะกดคำ
        spelling_res = evaluate_text(answer_text)
        spelling_score = float(spelling_res.get("score", 0))

        # 5) การใช้ถ้อยคำ/สำนวน (S5)
        s5_result = evaluate_student_text(
            answer_text,
            keyword_dict,
            spoken_words_set,
            notinlan_set,
            local_dialect_set
        )
        score_s5 = float(s5_result.get("score", 0))

        # 6) การใช้ประโยค (S6)
        s6_result = evaluate_sentence_usage(answer_text)
        score_s6 = float(s6_result.get("score", 0))

    total_score1 = mind_total + ordering1_score + summary1_score + spelling_score + score_s5 + score_s6


    # ---------------------------
    # ✅ ข้อที่ 2 : ตรวจจาก essay_analysis
    # ---------------------------

    # 1) คำบอกข้อคิดเห็น (S7)
    agreement_result = evaluate_agreement_with_reference(essay_analysis, reference_text)
    agreement_score = int(agreement_result.get("score", 0))

    # 2) เหตุผลสนับสนุน (S8)
    s8_result = evaluate_student_answer8(essay_analysis, reference_text, core_sentences, local_words_s8)
    s8_score = int(s8_result.get("score_total", 0))

    # 🔹 เงื่อนไข (2.1): ถ้าไม่มีคำบอกข้อคิดเห็น และ เหตุผลสนับสนุน = 0 → ข้อที่ 2 = 0 ทั้งข้อ
    # 🔹 เงื่อนไข (2.2): ตรวจจำนวนบรรทัด essay_analysis ถ้ามี ≤ 2 → ตรวจแค่ S7–S8, ที่เหลือเป็น 0
    line_count = essay_analysis.count("\n") + 1
    if (s8_score == 0) or (line_count >= 1 and line_count <= 2):
        ordering2_score = 0
        ordering2_details = {}
        comment_validity_score = 0
        comment_validity_details = {}
        score_s11 = 0
        s11_result = {}
        score_s12 = 0
        s12_result = {}
        score_s13 = 0
        s13_result = {}
    
    else:
        # 3) เรียงลำดับความคิด
        ordering2_result = evaluate_ordering_and_coherence(
            essay_analysis,
            ignore_list_s9=ignore_list_s9,
            specific_terms_s9=specific_terms_s9,
            ignore_single_char_s9=ignore_single_char_s9,
            similarity_threshold=0.3
        )
        ordering2_score = int(ordering2_result.get("คะแนนรวม", 0))
        ordering2_details = convert_numpy_to_python(ordering2_result)

        # 4) ความถูกต้องตามหลักการแสดงความคิดเห็น
        comment_validity_result = evaluate_comment_validity(essay_analysis)
        comment_validity_score = float(comment_validity_result.get("score", 0))
        comment_validity_details = convert_numpy_to_python(comment_validity_result)

        # 5) การสะกดคำ 
        s11_result = evaluate_text_s11(essay_analysis)
        score_s11 = float(s11_result.get("score", 0))

        # 6) การใช้คำ/ถ้อยคำสำนวน (S12)
        s12_result = evaluate_student_text_s12(
            essay_analysis,
            keyword_dict_s12,
            spoken_words_set_s12,
            notinlan_set_s12,
            local_dialect_set_s12
        )
        score_s12 = float(s12_result.get("score", 0))

        # 7) การใช้ประโยค (S13)
        s13_result = evaluate_reasoning_usage(essay_analysis)
        score_s13 = float(s13_result.get("score", 0))

    # ✅ รวมคะแนน
    total_score2 = agreement_score + s8_score + ordering2_score + comment_validity_score + score_s11 + score_s12 + score_s13
    total_all = total_score1 + total_score2

    # ---------------------------
    # ✅ คืนค่า JSON-safe
    # ---------------------------
    return convert_numpy_to_python({
        "ข้อที่ 1 - ใจความสำคัญ": {"cosine_similarity": round(best_score, 3), **mind_score, "คะแนนรวมใจความ": mind_total, "bert_score": round(best_score, 3)},
        "ข้อที่ 1 - การเรียงลำดับและเชื่อมโยงความคิด": {"score": ordering1_score, "details": ordering1_details},
        "ข้อที่ 1 - ความถูกต้องตามหลักการเขียนย่อความ": {"score": summary1_score, **summary1_details},
        "ข้อที่ 1 - การสะกดคำ": {"score": spelling_score, "details": spelling_res},
        "ข้อที่ 1 - การใช้คำ/ถ้อยคำสำนวน": {"score": score_s5, "details": s5_result},
        "ข้อที่ 1 - การใช้ประโยค": {"score": score_s6, "details": s6_result},

        "ข้อที่ 2 - คำบอกข้อคิดเห็น": agreement_result,
        "ข้อที่ 2 - เหตุผลสนับสนุน": s8_result,
        "ข้อที่ 2 - การเรียงลำดับและเชื่อมโยงความคิด": {"score": ordering2_score, "details": ordering2_details},
        "ข้อที่ 2 - ความถูกต้องตามหลักการแสดงความคิดเห็น": {"score": comment_validity_score, "details": comment_validity_details},
        "ข้อที่ 2 - การสะกดคำ/การใช้ภาษา": s11_result,
        "ข้อที่ 2 - การใช้คำ/ถ้อยคำสำนวน": {"score": score_s12, "details": s12_result},
        "ข้อที่ 2 - การใช้ประโยค": {"score": score_s13, "details": s13_result},

        "คะแนนรวมข้อที่ 1": total_score1,
        "คะแนนรวมข้อที่ 2": total_score2,
        "คะแนนรวมทั้งหมด": total_all
    })