"""
OffenseGuard Bilingual Lexicon
Contains curated wordlists, animal classification, human target pronouns,
praise markers, and explicit slurs for Roman Urdu and Urdu script.
"""

CATEGORIES = {
    0: "Non-offensive (Neutral)",
    1: "Hate Speech",
    2: "Abusive/Profanity",
    3: "Offensive"
}

# 0. Common Pakistani Personal Names & Political Figures (should never be flagged standalone):
COMMON_NAMES = {
    'imran', 'nawaz', 'ali', 'ahmed', 'ahmad', 'muhammad', 'mohammad', 'usman', 'bilal',
    'hamza', 'fatima', 'ayesha', 'zainab', 'khan', 'hassan', 'hasan', 'hussain', 'husain',
    'asif', 'shahid', 'babar', 'rizwan', 'shaheen', 'maryam', 'bilawal', 'zardari',
    'tariq', 'omer', 'umar', 'saad', 'rashid', 'farhan', 'kamran', 'salman', 'amna',
    'hina', 'sara', 'sana', 'iqbal', 'quaid', 'waqas', 'arslan', 'haris', 'adnan',
    'irfan', 'kashif', 'fahad', 'junaid', 'zubair', 'faisal', 'abid', 'nadeem',
    'rehan', 'daniyal', 'danish', 'mohsin', 'shahrukh', 'atif', 'mehwish', 'saba',
    'mahira', 'iqra', 'nimra', 'sadia', 'samina', 'bushra', 'sidra', 'marium',
    'عمران', 'نواز', 'علی', 'احمد', 'محمد', 'عثمان', 'بلال', 'حمزہ', 'فاطمہ', 'عائشہ',
    'زینب', 'خان', 'حسن', 'حسین', 'آصف', 'شاہد', 'بابر', 'رضوان', 'شاہین', 'مریم',
    'بلاول', 'زرداری', 'طارق', 'عمر', 'سعد', 'راشد', 'فرحان', 'کامران', 'سلمان', 'آمنہ',
    'حنا', 'سارہ', 'ثنا', 'اقبال', 'وقاص', 'ارسلان', 'حارث', 'عدنان', 'عرفان', 'کاشف',
    'فہد', 'جنید', 'زبیر', 'فیصل', 'عابد', 'ندیم', 'ریحان', 'دانیال', 'دانش', 'محسن',
    'عاطف', 'صبا', 'اقرا', 'نمرہ', 'سعدیہ', 'ثمینہ', 'بشریٰ', 'سدرہ'
}

# 1. Noble animals / heroic praise metaphors (positive in Urdu culture):
NOBLE_PRAISE_ANIMALS = {
    'sher', 'shair', 'cheeta', 'cheetay', 'shaheen', 'baaz', 'hiran', 'bulbul', 'ghoda', 'ghode',
    'شیر', 'چیتا', 'شاہین', 'باز', 'ہرن', 'بلبل', 'گھوڑا', 'گھوڑے'
}

# 2. Pejorative animal slurs (when directed at humans or actions, used as insults):
PEJORATIVE_ANIMALS = {
    'kutta', 'kutte', 'kutto', 'kutti', 'kuttay', 'kuto',
    'gadha', 'gadhe', 'gadho', 'ghadha', 'ghadhe', 'khota', 'khote',
    'suar', 'suwar', 'ullu', 'bandar', 'chirkut', 'kanjar', 'bakra', 'bakre',
    'janwar', 'janwaron', 'janwaro', 'janwaroon', 'janwer',
    'haivan', 'haiwan', 'darinda', 'darinde', 'darinday',
    'کتا', 'کتے', 'کتوں', 'کتیا', 'کتیاں',
    'گدھا', 'گدھے', 'گدھوں', 'کھوتا', 'کھوتے',
    'سور', 'خنزیر', 'الو', 'بندر', 'بکرا', 'بکرے',
    'جانور', 'جانوروں', 'حیوان', 'درندہ', 'درندے'
}

# 2b. Pejorative dehumanizing animal insult phrases:
PEJORATIVE_ANIMAL_PHRASES = [
    'janwar ho', 'janwar hai', 'janwar he', 'janwar hain', 'janwar hay', 'janwar tha',
    'janwar jesa', 'janwar jesi', 'janwar jese', 'janwaro jesa', 'janwaro jesi', 'janwaro jese',
    'janwaron jesa', 'janwaron jesi', 'janwaron jese', 'janwaroon jesi', 'janwaron ki tarah', 'janwar ki tarah',
    'harkaten janwar', 'harkatein janwar', 'harkat janwar', 'harkat janwaron',
    'kutta ho', 'kutta hai', 'kutta he', 'kutte ho', 'kutte hai', 'kuttay ho', 'kuttay hai',
    'gadha ho', 'gadha hai', 'gadha he', 'gadhe ho', 'gadhe hai',
    'suar ho', 'suar hai', 'suar he',
    'haivan ho', 'haivan hai', 'darinda hai', 'darinda ho',
    'kuttay jesa', 'kuttay jesi', 'kutton jesi', 'kutton ki tarah',
    'gadhe jesa', 'gadhe jesi', 'gadho jesi', 'gadhe ki tarah',
    'جانور ہو', 'جانور ہے', 'جانور ھے', 'جانور ہیں', 'جانور جیسا', 'جانور جیسی', 'جانور جیسے',
    'جانوروں جیسا', 'جانوروں جیسی', 'جانوروں جیسے', 'جانوروں کی طرح', 'جانور کی طرح',
    'حرکتیں جانور', 'حرکت جانور', 'کتا ہے', 'کتا ہو', 'کتے ہو', 'گدھا ہے', 'گدھا ہو',
    'سور ہے', 'سور ہو', 'حیوان ہے', 'حیوان ہو', 'درندہ ہے'
]

# Actual biological animal species (for zoological context verification):
ACTUAL_BIOLOGICAL_ANIMALS = {
    'kutta', 'kutti', 'kutte', 'kuttay', 'billi', 'billa', 'billiyan',
    'sher', 'cheeta', 'hathi', 'ghoda', 'ghode', 'gadha', 'gadhe', 'khota',
    'suar', 'suwar', 'bakra', 'bakri', 'bakray', 'gay', 'gaaye', 'bhains',
    'oonth', 'bail', 'lomri', 'bheriya', 'bandar', 'hiran', 'bulbul', 'shaheen', 'chirya', 'machhli',
    'کتا', 'بلی', 'شیر', 'چیتا', 'ہاتھی', 'گھوڑا', 'گدھا', 'سور', 'بکرا', 'بکری', 'گائے', 'بھینس', 'اونٹ'
}

ALL_ANIMAL_WORDS = NOBLE_PRAISE_ANIMALS.union(PEJORATIVE_ANIMALS).union(ACTUAL_BIOLOGICAL_ANIMALS).union({
    'billio', 'murga', 'murgi', 'dog', 'dogs', 'cat', 'cats', 'horse', 'donkey', 'animal', 'animals',
    'بیل', 'بھینس', 'اونٹ', 'ہاتھی', 'لومڑی', 'بھیڑیا',
    'بلی', 'بلیاں', 'مرغ', 'مرغا', 'مرغی', 'جانور', 'جانوروں', 'حیوان', 'حیوانات',
    'چڑیا', 'مچھلی', 'پرندہ', 'پرندے'
})

# 3. Explicit Zoological / Factual context indicators (definitively biological context):
ZOOLOGICAL_FACTUAL_MARKERS = [
    'ek janwar', 'aik janwar', 'palto', 'paaltu', 'paltoo', 'wafadar janwar',
    'jungli janwar', 'jungle', 'jangal', 'chopaya', 'nasl', 'nasal', 'breed',
    'animal', 'pet', 'pets', 'veterinary', 'bhonkta', 'doodh',
    'ایک جانور', 'وفادار جانور', 'پالتو جانور', 'پالتو',
    'جنگلی جانور', 'جنگل', 'چرند', 'پرند', 'چوپایہ', 'نسل', 'پالا', 'پالتے', 'بھونکتا',
    'کاٹتا', 'گوشت خور', 'گھاس', 'دودھ دینے والا', 'چڑیا گھر'
]

# 4. Praise / Compliment / Positive context descriptors:
PRAISE_DESCRIPTORS = [
    'bahadur', 'diler', 'shandar', 'shaandaar', 'zindadil', 'behtreen',
    'zabardast', 'kamal', 'kamaal', 'pyara', 'pyari', 'wafadar', 'madadgar',
    'masoom', 'shareef', 'fakhr', 'proud', 'hero', 'champ', 'superstar',
    'acha', 'accha', 'achi', 'acchi', 'good', 'great', 'brave', 'strong',
    'loyal', 'cute', 'sweet', 'kind', 'honest', 'larte', 'larta', 'larhte',
    'larhta', 'hifazat', 'bachaya', 'pyar', 'muhabbat', 'pasand', 'khoobsurat', 'khubsurat',
    'بہادر', 'دلیر', 'شاندار', 'زندہ دل', 'بہترین', 'زبردست', 'کمال', 'پیارا', 'پیاری',
    'وفادار', 'مددگار', 'معصوم', 'شریف', 'فخر', 'اچھا', 'اچھی', 'اچھے', 'خوبصورت',
    'محبت', 'پیار', 'پسند', 'حفاظت', 'بچایا'
]

# 5. Explicit abuse / slurs (ALWAYS offensive regardless of animal names):
EXPLICIT_ABUSE = [
    # Shamelessness / dishonour insults
    'beghairat', 'be-ghairat', 'baghairat', 'beghyrat', 'beghairat', 'begharat',
    'besharm', 'be-sharm', 'bewaqoof', 'be-waqoof', 'bewaquf', 'ahmaq',
    'pagal', 'paagal', 'diwana', 'diwanay', 'ullu ka patha', 'ullu ka pattha',
    # Classic heavy slurs
    'kameena', 'kameene', 'kameenay', 'kameeni', 'harami', 'haramkhor',
    'haramzada', 'haramzadi', 'haramzade', 'haram',
    'jahil', 'jahalat', 'chutiya', 'chutiye', 'chutya', 'madarchod', 'mc',
    'behenchod', 'bhenchod', 'bc', 'randi', 'gandu', 'dalal', 'kanjar',
    'bhadwa', 'bhadway', 'laanti', 'lanat', 'aisi ki taisi', 'maa ki aankh',
    'teri maa', 'teri behan', 'tatti', 'choot', 'lodu', 'lund', 'bhosdike',
    'bhosdi', 'gashti', 'kutti ka', 'kanjri', 'dallay',
    'maderchod', 'madar', 'sister fucker', 'saala', 'saali', 'sala', 'sali',
    'kutte ki aulad', 'kutte k bachay', 'kuttay k bachay', 'kutte kay bachay', 'kuttay kay bachay',
    'kutta k bachay', 'kutta ki aulad', 'kutta kay bachay',
    'gadhe ki aulad', 'gadhe k bachay', 'gadha k bachay', 'gadha ki aulad',
    'namoosi', 'namoos', 'zaleel', 'zaleeli',
    'neech', 'neecha', 'paleed', 'gandagi', 'faaltu', 'nikamma', 'nikaamma',
    'badmaash', 'badmash', 'badtameez', 'badtamiz', 'gustakh', 'ghatiya',
    'khabees', 'khabbees', 'lafanga', 'awara', 'charsi', 'nashebaaz', 'nashebaz',
    'chor', 'chorni', 'dhokebaaz', 'dhoka', 'faraibi', 'munafiq', 'munaafiq',
    'fasiq', 'zalim', 'zaalim', 'murdaad', 'murtad', 'kafir', 'kaafir',
    # Urdu Script explicit abuse
    'بے غیرت', 'بے شرم', 'احمق', 'بیوقوف', 'پاگل', 'دیوانہ', 'الو کا پٹھا',
    'کمینہ', 'کمینے', 'کمینی', 'حرامی', 'حرامخور', 'حرام خور', 'حرامزادہ',
    'جاہل', 'چوتیہ', 'مادرچود', 'بہنچود', 'بہن چود', 'رنڈی', 'گانڈو', 'دلا',
    'کنجر', 'کنجری', 'بھڑوا', 'لعنت', 'تیری ماں', 'تیری بہن', 'لنڈ', 'بھوسڑی',
    'گشتی', 'کتے کی اولاد', 'گدھے کی اولاد', 'ذلیل', 'نیچ', 'پلید', 'بدتمیز',
    'بدقماش', 'بدماش', 'گھٹیا', 'خبیث', 'لفنگا', 'اوارہ', 'منافق', 'ظالم'
]

HUMAN_TARGET_PRONOUNS = {
    'tum', 'tumhe', 'tumko', 'tumhara', 'tumhari', 'tumhare',
    'tu', 'tujhe', 'tujhko', 'teri', 'tera', 'tere',
    'aap', 'aapko', 'aapka', 'aapki', 'aapke', 'ap',
    'you', 'your', 'yours',
    'insan', 'insaan', 'banda', 'banday', 'aadmi', 'shakhs', 'aurat',
    'woh', 'wo', 'yeh', 'ye', 'isay', 'usay', 'inko', 'unko', 'is', 'us',
    'bhai', 'behan', 'dost', 'yaar', 'beta', 'beti', 'baap',
    'تم', 'تمہیں', 'تم کو', 'تمکو', 'تمہارا', 'تمہاری', 'تمہارے',
    'تو', 'تجھے', 'تجھکو', 'تیرا', 'تیری', 'تیرے',
    'آپ', 'آپکو', 'آپکا', 'آپکی', 'آپکے',
    'وہ', 'یہ', 'اسے', 'انہیں', 'انکو', 'اس', 'ان',
    'انسان', 'بندہ', 'بندے', 'آدمی', 'شخص', 'عورت', 'لڑکا', 'لڑکی', 'بھائی', 'بہن', 'دوست'
}.union(COMMON_NAMES)

STOP_WORDS = {
    'hai', 'hain', 'he', 'hay', 'hein', 'ho', 'hun', 'hoon', 'tha', 'the', 'thi', 'thin',
    'mein', 'main', 'kya', 'aur', 'yeh', 'ye', 'woh', 'wo', 'kar', 'raha', 'rahe', 'rahi',
    'ko', 'se', 'par', 'pe', 'ki', 'ka', 'ke', 'kay', 'k', 'ne', 'ik', 'ek', 'aik',
    'boht', 'bahut', 'bhi', 'to', 'toh', 'hi', 'kuch', 'koi', 'sab', 'jo', 'jis',
    'tum', 'tumhe', 'tumko', 'tumhara', 'tumhari', 'tumhare', 'tu', 'tujhe', 'tujhko',
    'teri', 'tera', 'tere', 'aap', 'aapko', 'aapka', 'aapki', 'aapke', 'ap',
    'mera', 'meri', 'mere', 'hum', 'humein', 'humko', 'hamara', 'hamari', 'hamare',
    'is', 'us', 'in', 'un', 'isko', 'usko', 'inko', 'unko', 'isay', 'usay',
    'harkat', 'harkaten', 'harkatein', 'tarah', 'jesa', 'jesi', 'jese', 'jaisi', 'jaisa', 'jaise',
    'walo', 'wale', 'wali', 'wala', 'baat', 'baatein', 'karo', 'karta', 'karte', 'karti',
    'ہے', 'ہیں', 'ھے', 'ہو', 'ہوں', 'تھا', 'تھے', 'تھی', 'تھیں',
    'میں', 'کیا', 'اور', 'یہ', 'وہ', 'کر', 'رہا', 'رہے', 'رہی', 'کو', 'سے', 'پر', 'کی', 'کا', 'کے', 'نے', 'ایک',
    'بہت', 'بھی', 'تو', 'ہی', 'کچھ', 'کوئی', 'سب', 'جو', 'جس',
    'تم', 'تمہیں', 'تم کو', 'تمہارا', 'تمہاری', 'تمہارے', 'تو', 'تجھے', 'تیرا', 'تیری', 'تیرے',
    'آپ', 'آپ کا', 'آپ کی', 'آپ کے', 'میرا', 'میری', 'میرے', 'ہم', 'ہمارا', 'ہماری', 'ہمارے',
    'حرکت', 'حرکتیں', 'طرح', 'جیسا', 'جیسی', 'جیسے'
}
