# ai_scoring.py
import json
import os
import re
import difflib
import requests
from sentence_transformers import SentenceTransformer, util
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_words
from pythainlp.tag import pos_tag
from pythainlp.util import normalize

# โหลดโมเดล
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ✅ -----------------------------
# หา path ของไฟล์ json ในโฟลเดอร์ data
BASE_DIR = os.path.dirname(__file__)  # โฟลเดอร์ปัจจุบัน
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
        "คะแนนรวมใจความ ": total_score
    }

    return result

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
    # วิเคราะห์
    linebreak_issues = analyze_linebreak_issues(text)
    corrected_text = merge_linebreak_words(text, linebreak_issues)
    tokens = word_tokenize(corrected_text, engine='newmm', keep_whitespace=False)
    pos_tags = pos_tag(tokens, corpus='orchid')

    # ✅ ตรวจคำทับศัพท์ (ใช้ loanwords_whitelist ที่ประกาศ global)
    loanword_spell_errors = check_loanword_spelling(tokens, loanwords_whitelist)

    # ตรวจสะกด
    pythai_errors = pythainlp_spellcheck(tokens, pos_tags, dict_words=thai_dict, ignore_words=custom_words)
    wrong_words = [e['word'] for e in pythai_errors]
    longdo_results = longdo_spellcheck_batch(wrong_words)
    spelling_errors_legit = [
        {**e, 'suggestions': longdo_results.get(e['word'], [])}
        for e in pythai_errors if e['word'] in longdo_results
    ]

    # อื่น ๆ
    punct_errors = find_unallowed_punctuations(text)
    maiyamok_results, has_wrong_maiyamok = analyze_maiyamok(tokens, pos_tags)
    split_errors = detect_split_errors(tokens, custom_words=custom_words)

    # ==== นับจำนวนข้อผิดพลาดแต่ละประเภท ====
    error_counts = {
        "spelling": len(spelling_errors_legit) + len(loanword_spell_errors),
        "linebreak": len(linebreak_issues),
        "split": len(split_errors),
        "punct": len(punct_errors),
        "maiyamok": sum(1 for r in maiyamok_results if r['สถานะ'].startswith('❌'))
    }
    n_issue_types = sum(1 for c in error_counts.values() if c > 0)
    multi_in_single_type = any(c >= 2 for c in error_counts.values())

    # ==== สร้าง reasons ====
    reasons = []
    if error_counts["linebreak"]:
        details = [f"{issue['prev_part']} + {issue['next_part']} → {issue['combined']}" for issue in linebreak_issues]
        reasons.append("พบการฉีกคำข้ามบรรทัด: " + "; ".join(details))
    if error_counts["split"]:
        details = [f"{e['split_pair'][0]} + {e['split_pair'][1]} → {e['suggested']}" for e in split_errors]
        reasons.append("พบการแยกคำผิด: " + "; ".join(details))
    if error_counts["spelling"]:
        error_words = [e['word'] for e in spelling_errors_legit]
        error_desc = [f"{e['found']} (ควรเป็น {e['should_be']})" for e in loanword_spell_errors]
        reasons.append(f"ตรวจเจอคำสะกดผิดหรือทับศัพท์ผิด: {', '.join(error_words + error_desc)}")
    if error_counts["punct"]:
        reasons.append(f"ใช้เครื่องหมายที่ไม่อนุญาต: {', '.join(punct_errors)}")
    if error_counts["maiyamok"]:
        wrong_desc = [x for x in maiyamok_results if x['สถานะ'].startswith('❌')]
        texts = [f"{x['คำก่อนไม้ยมก']}: {x['สถานะ']}" for x in wrong_desc]
        reasons.append("ใช้ไม้ยมกผิด: " + '; '.join(texts))
    if not reasons:
        reasons.append("ไม่มีปัญหา")

    # ==== เกณฑ์การให้คะแนน ====
    if sum(error_counts.values()) == 0:
        score = 1.0
    elif n_issue_types == 1 and multi_in_single_type:
        score = 0.0
    elif n_issue_types == 1:
        score = 0.5
    else:
        score = 0.0

    return {
        'linebreak_issues': linebreak_issues,
        'spelling_errors': spelling_errors_legit,
        'loanword_spell_errors': loanword_spell_errors,
        'punctuation_errors': list(punct_errors),
        'maiyamok_results': maiyamok_results,
        'split_errors': split_errors,
        'reasons': reasons,
        'score': score
    }

# ✅ -----------------------------
# ฟังก์ชันหลัก เรียกจาก FastAPI
# ✅ -----------------------------
def evaluate_single_answer(answer_text):
    student_emb = model.encode(answer_text, convert_to_tensor=True)
    core_sentences = [
        "สื่อสังคมหรือสื่อออนไลน์หรือสื่อสังคมออนไลน์เป็นช่องทางที่ใช้ในการเผยแพร่หรือค้นหาหรือรับข้อมูลข่าวสาร",
        "การใช้สื่อสังคมหรือสื่อออนไลน์หรือสื่อสังคมออนไลน์อย่างไม่ระมัดระวังหรือขาดความรับผิดชอบจะเกิดโทษหรือผลเสียหรือข้อเสียหรือผลกระทบหรือสิ่งไม่ดี",
        "ผู้ใช้ต้องรู้ทันหรือรู้เท่าทันสื่อสังคมออนไลน์",
        "การใช้สื่อสังคมหรือสื่อออนไลน์หรือสื่อสังคมออนไลน์ด้วยเจตนาแอบแฝงมีผลกระทบต่อความน่าเชื่อถือของข้อมูลข่าวสาร"
    ]
    core_embs = model.encode(core_sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(student_emb, core_embs)[0]
    best_score = cosine_scores.max().item()

    # ถ้า similarity ต่ำกว่าเกณฑ์ ไม่ตรวจใจความ
    if best_score < 0.6:
        return {
            "cosine_similarity": best_score,
            "คะแนนใจความสำคัญ": {"ใจความที่ 1": 0, "ใจความที่ 2": 0,
                                "ใจความที่ 3": 0, "ใจความที่ 4": 0,
                                "คะแนนรวมใจความ ": 0},
            "คะแนนการสะกดคำ": 0.0,
            "เงื่อนไขที่ผิด": ["Cosine similarity < 0.6 ไม่ตรวจใจความสำคัญ"],
            "คะแนนรวมทั้งหมด": 0.0
        }

    # ตรวจใจความสำคัญ
    mind_score = evaluate_mind_score(answer_text)
    mind_total = mind_score["คะแนนรวมใจความ "]

    if mind_total == 0:
        spelling_score = 0.0
        combined_score = 0.0
        spelling_reason = ["ไม่ได้คะแนนใจความสำคัญ (ใจความ = 0) จึงไม่ตรวจการสะกดคำ"]
    else:
        res = evaluate_text(str(answer_text))
        spelling_score = res["score"]
        combined_score = mind_total + spelling_score
        spelling_reason = res["reasons"]

    return {
        "cosine_similarity": best_score,
        "คะแนนใจความสำคัญ (4 คะแนน)": mind_score,
        "คะแนนการสะกดคำ (1 คะแนน)": spelling_score,
        "เงื่อนไขที่ผิด": spelling_reason,
        "คะแนนรวมทั้งหมด": combined_score
    }