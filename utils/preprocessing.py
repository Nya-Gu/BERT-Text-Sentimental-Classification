import re
import pandas as pd

def Proofreading(text):
    if pd.isna(text):
        return ""
    
    text = text.strip()
    text = text.replace('g', 'ㅎ')
    text = text.replace('z', 'ㅋ')
    text = text.replace('ooo', 'oo')

    text = text.replace('ㅜ', 'ㅠ')
    text = text.replace('ㅠㅡㅠ', 'ㅠ')
    text = text.replace('ㅠㅡ', 'ㅠ')
    text = text.replace('ㅡㅠ', 'ㅠ')

    text = text.replace('-_-', 'ㅡㅡ')
    text = text.replace('ㅡ.ㅡ', 'ㅡㅡ')

    text = text.replace('ㅡㅅ', 'ㅡㅅㅡ')
    text = text.replace('ㅅㅡ', 'ㅡㅅㅡ')
    text = text.replace('ㅡㅇㅡ', 'ㅡㅅㅡ')

    text = text.replace('에효', '에휴')
    text = text.replace('어휴', '에휴')
    text = text.replace('아호', '에휴')

    text = text.replace('ㅅㅂ', '씨발')
    text = text.replace('ㅆㅂ', '씨발')
    text = text.replace('ㅣ발', '씨발')
    text = text.replace('시발', '씨발')

    text = text.replace('ㅇㅏ','아')
    text = text.replace('ㅏ','아')
    text = text.replace('ㅎㅏ','하')
    text = text.replace('오ㅐ','왜')
    text = text.replace('ㄱㅐ','개')
    text = text.replace('욱겨','웃겨')
    text = text.replace('ㅣ친','미친')
    text = text.replace('ㅆㄹㄱ','쓰레기')
    text = text.replace('ㅆㄺ','쓰레기')
    text = text.replace('ㅔ','에')
    text = text.replace('ㅐ','애')

    # 글자 사이, 문장 시작과 끝에 위치한 모음 ㅣ를 제거 ex) 가ㅣ나 -> 가나, ㅣ가 -> 가, 나ㅣ -> 나
    text = re.sub(r'(?<=[가-힣])ㅣ(?=[가-힣])', '', text)
    text = re.sub(r'^ㅣ(?=[가-힣])', '', text)
    text = re.sub(r'(?<=[가-힣])ㅣ$', '', text)

    # 글자 사이, 문장 시작과 끝에 위치한 모음 ㅡ를 제거 ex) 가ㅡ나 -> 가나, ㅡ가 -> 가, 나ㅡ -> 나
    text = re.sub(r'(?<=[가-힣])ㅡ(?=[가-힣])', '', text)
    text = re.sub(r'^ㅡ(?=[가-힣])', '', text)
    text = re.sub(r'(?<=[가-힣])ㅡ$', '', text)

    text = text.replace('ㆍ', '')

    text = text.lower()
    text = text.replace('bomb','[neg_ex]')
    text = text.replace('shit','[neg_ex]')
    text = text.replace('bad','[neg_ex]')
    text = text.replace('bored','[neg_ex]')
    text = text.replace('ㅎood', 'good')
    
    return text.strip()




HTML_RE     = re.compile(r"<[^>]*>")
EMAIL_RE    = re.compile(r"\S+@\S+")
HASH_MEN_RE = re.compile(r"#\w+|@\w+")
URL_RE      = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
PHONE_RE    = re.compile(r"\b01[016789]-\d{3,4}-\d{4}\b")
DATE_RE     = re.compile(r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b")
TIME_RE     = re.compile(r"\b(?:오전|오후)?\s?\d{1,2}:\d{2}\b")
SPACE_RE    = re.compile(r"\s+")
ESSENCE_RE  = re.compile(r"[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?~\-_^*;♥]")

# 텍스트 전처리 파이프라인 클래스 구성
class TextPreprocessingPipeline:
    """
    텍스트 전처리 파이프라인 클래스
    - 기본 전처리와 학습 데이터 기반 고급 전처리를 통합 관리
    - 재사용 가능하고 확장 가능한 구조
    """

    def __init__(self):
        self.is_fitted = False
        self.vocab_info = {}
        self.label_patterns = {}

    def basic_preprocess(self, texts):
        """기본 전처리 (clean_text + normalize 기능)"""
        processed_texts = []
        for text in texts:
            # 기본 텍스트 정리
            cleaned = self._clean_text(text)
            # 정규화
            normalized = self._normalize_text(cleaned)
            processed_texts.append(normalized)
        return processed_texts

    def _clean_text(self, text):
        """기존 clean_text 함수 내용"""
        if pd.isna(text):
            return ""

        text = str(text).lower().strip()                            # 영어 소문자화 및 양 끝 공백 제거
        text = re.sub(r"([가-힣])\1{4,}", r"\1\1\1\1", text)     # 과도한 한글 반복 축소: 4번 이상 반복 -> 4번으로
        text = re.sub(r"([ㄱ-ㅎㅏ-ㅣ])\1{4,}", r"\1\1\1\1", text) # 과도한 자/모음 반복 축소: 4번 이상 반복 -> 4번으로
        
        text = SPACE_RE.sub(" ", text)                              # 연속 공백 제거

        return text.strip()

    def _normalize_text(self, text):
        """텍스트 정규화 함수"""
        # 여러 개의 구두점을 최대 3개로 정규화
        text = re.sub(r"[.]{2,}", ".", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        text = re.sub(r"[,]{2,}", ",", text)
        text = re.sub(r"[;]{2,}", ";", text)
        
        # 구두점 주변 공백 정리
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = re.sub(r"([.,!?])\s+", r"\1 ", text)
        
        # 감정 기호 주변 공백 추가
        text = text.replace('ㅋㅋ', 'ㅋㅋ ')
        text = text.replace('ㅎㅎ', 'ㅎㅎ ')
        text = text.replace('ㅠㅠ', 'ㅠㅠ ')
        text = text.replace('ㅡㅡ', 'ㅡㅡ ')
        text = text.replace('ㅡㅅㅡ', 'ㅡㅅㅡ ')

        # 특수문자 정리
        text = HTML_RE.sub(" ", text)       # HTML 태그 제거
        text = EMAIL_RE.sub(" ", text)      # 이메일 주소 제거
        text = HASH_MEN_RE.sub(" ", text)   # 해시 및 멘션 제거
        text = URL_RE.sub(" ", text)        # URL 제거
        text = PHONE_RE.sub(" ", text)      # 전화번호 제거
        text = DATE_RE.sub(" ", text)       # 날짜 제거
        text = TIME_RE.sub(" ", text)       # 시간 제거
        text = ESSENCE_RE.sub(" ", text)    # 불필요한 문자 제거

        text = SPACE_RE.sub(" ", text)      # 연속 공백 제거

        return text.strip()

    def fit(self, texts, labels=None):
        """학습 데이터로부터 전처리 정보 학습"""
        print("학습 데이터 기반 전처리 정보 수집 중...")

        # NOTE: 학습 데이터를 분석하여 전처리에 필요한 정보를 수집하고 저장하는 코드를 직접 구현해서 추가해보세요!
        # 예시:
        # - 도메인 특화 패턴 분석 (영화 리뷰 특성)
        # - 빈도 기반 노이즈 패턴 식별
        # - 라벨별 텍스트 특성 분석
        # - 어휘 사전 구축
        # - 정규화 규칙 최적화

        self.is_fitted = True
        print("✓ 전처리 파이프라인 학습 완료")

    def transform(self, texts):
        """전처리 적용"""
        # if not self.is_fitted:
        #     print(
        #         "Warning: 파이프라인이 학습되지 않았습니다. 기본 전처리만 적용합니다."
        #     )

        return self.basic_preprocess(texts)

    def fit_transform(self, texts, labels=None):
        """학습과 변환을 동시에 수행"""
        self.fit(texts, labels)
        return self.transform(texts)


class TextPreprocessingPipeline_Base:
    """
    텍스트 전처리 파이프라인 클래스
    - 기본 전처리와 학습 데이터 기반 고급 전처리를 통합 관리
    - 재사용 가능하고 확장 가능한 구조
    """

    def __init__(self):
        self.is_fitted = False
        self.vocab_info = {}
        self.label_patterns = {}

    def basic_preprocess(self, texts):
        """기본 전처리 (clean_text + normalize 기능)"""
        processed_texts = []
        for text in texts:
            # 기본 텍스트 정리
            cleaned = self._clean_text(text)
            # 정규화
            normalized = self._normalize_text(cleaned)
            processed_texts.append(normalized)
        return processed_texts

    def _clean_text(self, text):
        """기존 clean_text 함수 내용"""
        if pd.isna(text):
            return ""

        text = str(text).strip()
        text = re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", "", text)
        text = re.sub(r"([ㅋㅎ])\1{2,}", r"\1\1", text)
        text = re.sub(r"([ㅠㅜㅡ])\1{2,}", r"\1\1", text)
        text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
        text = re.sub(r"[^\w\s가-힣.,!?ㅋㅎㅠㅜㅡ~\-]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _normalize_text(self, text):
        """텍스트 정규화 함수"""
        # 소문자 변환
        text = text.lower()

        # 구두점 정규화
        text = re.sub(r"[.]{2,}", ".", text)
        text = re.sub(r"[!]{2,}", "!", text)
        text = re.sub(r"[?]{2,}", "?", text)
        text = re.sub(r"[,]{2,}", ",", text)
        text = re.sub(r"\s+([.,!?])", r"\1", text)
        text = re.sub(r"([.,!?])\s+", r"\1 ", text)

        # 특수문자 정리
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def fit(self, texts, labels=None):
        """학습 데이터로부터 전처리 정보 학습"""
        print("학습 데이터 기반 전처리 정보 수집 중...")

        # NOTE: 학습 데이터를 분석하여 전처리에 필요한 정보를 수집하고 저장하는 코드를 직접 구현해서 추가해보세요!
        # 예시:
        # - 도메인 특화 패턴 분석 (영화 리뷰 특성)
        # - 빈도 기반 노이즈 패턴 식별
        # - 라벨별 텍스트 특성 분석
        # - 어휘 사전 구축
        # - 정규화 규칙 최적화

        self.is_fitted = True
        print("✓ 전처리 파이프라인 학습 완료")

    def transform(self, texts):
        """전처리 적용"""
        if not self.is_fitted:
            print(
                "Warning: 파이프라인이 학습되지 않았습니다. 기본 전처리만 적용합니다."
            )

        return self.basic_preprocess(texts)

    def fit_transform(self, texts, labels=None):
        """학습과 변환을 동시에 수행"""
        self.fit(texts, labels)
        return self.transform(texts)