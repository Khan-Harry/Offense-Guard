package com.offenseguard.keyboard;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Static Lexicon and Detection Dictionaries:
 * Contains blacklist sets, multi-word phrases, and zoological context markers.
 */
public class WordLists {

    // ─── Roman Urdu / English blacklist (matched after toLowerCase) ───────────
    public static final Set<String> ROMAN_BLACKLIST = new HashSet<>(Arrays.asList(
        // کمینہ
        "kameena", "kameene", "kameeni", "kameeno",
        "kamina", "kamine", "kaminey", "kamini",
        "kmina", "kmeena",

        // حرامی
        "harami", "haramee", "haramy", "hrami", "harammi",
        "haramzada", "haramzaada", "haramzade", "haramzadey", "haramzadi",
        "haramkhor",

        // چوتیہ
        "chutiya", "chootiya", "chutia", "chootia", "chotia",
        "chootiye", "chutiye", "chutiyas", "chutiyapa",
        "chutmarika", "choot",

        // گانڈو
        "gaandu", "gandu", "gandoo", "gaando",
        "gaand", "gand", "gandou",

        // بہنچود
        "behenchod", "bhenchod", "behanchod", "benachod",
        "bhen chod", "bhnchd", "bc", "b.c",

        // مادرچود
        "madarchod", "maderchod", "madarchaud",
        "madar chod", "mc",

        // لنڈ / لوڑا
        "lund", "lun", "land", "landa",
        "loray", "lorhay", "lora", "lawda",

        // کتا / کتیا
        "kutta", "kuttay", "kutte", "kutti", "kutia",
        "kuta", "kutey",

        // گدھا
        "gadha", "gadhe", "gadhi", "gadhey",
        "gadday",

        // رنڈی / گشتی
        "randi", "randy", "rande", "randwa", "randia",
        "gashti", "gashty", "gashte",

        // کنجر / دلال
        "kanjar", "kanjri", "kanjari",
        "dalla", "dalley", "dala",

        // بکواس
        "bakwas", "bakwaas", "bakawaas", "bakwasss",

        // پاگل
        "pagal", "paagal", "pagla", "pagli",

        // الو
        "ullo", "ullu", "ulu", "ulloo",

        // بدتمیز
        "badtameez", "badtamiz", "bad tameez", "bad tamiz",

        // احمق / بیوقوف
        "ahmaq", "ahmak",
        "bewaqoof", "bewakoof", "bewakuf", "bewkoof",
        "bawakoof", "baywakoof",

        // جاہل
        "jahil", "jaahil", "jaheel",

        // بھڑوا / چنال
        "bharwa", "bharway", "bhadwa", "bhadway",
        "chinal", "channl",

        // سور
        "suar", "suwar", "soar",

        // لعنت
        "lanat", "laanat", "laannat",

        // گھٹیا
        "ghatiya", "ghateya", "ghatiha",

        // سالا / سالی
        "sala", "saala", "saale", "sale",
        "sali", "saali",

        // حرام خور
        "haramkhor", "haram khor", "haramkhori",

        // گھنٹا
        "ghanta", "ghantaa", "gante",

        // کمبخت
        "kambakht", "kambakth",

        // نالائق
        "nalayiq", "nalayak", "nalaik", "nalyak",

        // بدمعاش
        "badmaash", "badmash", "badmaas",

        // لفنگا
        "lafanga", "lafange", "lafangi",

        // بے غیرت
        "beghairat", "be ghairat", "beghayrat",
        "besharam", "be sharam",
        "behaya", "be haya",

        // بکوُاس abbreviations
        "bkl", "bsdk", "bund",

        // English swear words
        "fuck", "fucking", "fucked", "fucker", "fucks", "fck", "f*ck", "fu*k",
        "shit", "shitting", "shitty", "shits", "sh*t",
        "bitch", "bitches", "b*tch",
        "bastard", "bastards",
        "asshole", "asshat", "ass",
        "cunt", "cunts",
        "dick", "dickhead", "dicks",
        "pussy", "pussies",
        "cock", "cocks",
        "whore", "whores",
        "slut", "sluts",
        "faggot", "fag",
        "idiot", "idiots",
        "stupid", "stupidity",
        "crap", "crappy",
        "dumbass", "dumb",
        "jackass", "jerk",
        "bullshit", "bs",
        "shithead", "motherfucker", "motherfucking",
        "fuckyou", "fuckoff", "fucku",
        "wtf", "stfu"
    ));

    // ─── Urdu Script blacklist (WITHOUT tashkeel) ──────────────────────────
    public static final Set<String> URDU_BLACKLIST = new HashSet<>(Arrays.asList(
        "کتا", "کتیا", "کتے", "کتی",
        "سور", "خنزیر",
        "گدھا", "گدھے", "گدھی",
        "الو",
        "حرامی", "حرامزادہ", "حرامکور",
        "کمینہ", "کمینے", "کمینی",
        "چوتیہ", "چوت", "چت",
        "گانڈو", "گانڈ", "گاند",
        "لنڈ", "لن", "لوڑا",
        "بھوسڑی", "بھوسڑا",
        "مادرچود", "بہنچود",
        "رنڈی", "رنڈوا",
        "گشتی", "چنال", "چنڈال",
        "دلا", "دلال", "کنجر", "کنجری", "کنجروں",
        "بھڑوا", "بھڑوے",
        "فاحشہ", "زانی", "زانیہ",
        "بدکردار", "بدکار",
        "بکواس", "لعنت",
        "خبیث", "ذلیل",
        "پاگل",
        "جاہل", "بدتمیز", "گھٹیا",
        "بدمعاش", "لفنگا", "لفنگے",
        "بے غیرت", "بے شرم", "بے حیا",
        "حرام خور",
        "سالا", "سالی",
        "نالائق", "احمق", "بیوقوف",
        "کمبخت", "بدبخت",
        "شیطان",
        "مردود", "ملعون", "لعین",
        "گنجا", "گدی",
        "چوڑا", "چوڑی",
        "ذلت"
    ));

    // ─── Multi-word Urdu & Roman Urdu phrases (substring search) ─────────
    public static final String[] URDU_PHRASES = {
        "سور کا بچہ", "الو کا پٹھا",
        "بہن چود", "بہنچود",
        "ماں چود", "ماں کی",
        "باپ کا", "ماں کا",
        "حرام زادہ", "حرام کا",
        "بے غیرت", "بے شرم", "بے حیا",
        "حرام خور",
        "کمینہ آدمی", "ذلیل آدمی", "نالائق آدمی",
        "کتے کی اولاد", "گدھے کی اولاد",
        "جانور ہو", "جانور ہے", "جانور ھے", "جانور جیسی", "جانور جیسا", "جانوروں جیسی", "جانوروں جیسا",
        // Roman Urdu possessive slurs & idioms (always offensive)
        "kutta k bachay", "kuttay k bachay", "kutta kay bachay", "kuttay kay bachay",
        "kutta ki aulad", "kutte ki aulad", "kuttay ki aulad",
        "gadha k bachay", "gadhe k bachay", "gadha ki aulad", "gadhe ki aulad",
        "ullu ka patha", "ullu ka pattha", "teri maa", "teri behan",
        "janwar ho", "janwar hai", "janwar he", "janwar jesi", "janwar jesa", "janwaro jesi", "janwaron jesi", "harkaten janwar"
    };

    // ─── Zoological / Biological Factual Context Markers (Prevent False Positives) ───
    public static final List<String> ZOOLOGICAL_MARKERS = Arrays.asList(
        "ایک جانور", "ek janwar", "aik janwar", "وفادار جانور", "wafadar janwar", "wafadar",
        "پالتو", "palto", "paaltu", "animal", "pet", "pets",
        "جنگلی", "jungli", "چوپایہ", "دودھ", "breed", "nasl", "نسل", "چرند", "پرند", "بیمار"
    );

    // Contextual animal words that are safe in zoological narrative
    public static final Set<String> CONTEXTUAL_ANIMALS = new HashSet<>(Arrays.asList(
        "کتا", "کتیا", "کتے", "کتی", "سور", "خنزیر", "گدھا", "گدھے", "گدھی", "الو", "بندر",
        "kutta", "kuttay", "kutte", "kutti", "kutia", "kuta", "kutey",
        "gadha", "gadhe", "gadhi", "gadhey", "suar", "suwar", "ullo", "ullu", "ulu", "bandar"
    ));
}
