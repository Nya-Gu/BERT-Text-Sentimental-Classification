import torch
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    # """
    # 리뷰 텍스트 데이셋 클래스
    # - BERT 모델 훈련/추론을 위한 PyTorch Dataset 구현
    # - 텍스트 토크나이징 및 텐서 변환 처리
    # """

    def __init__(self, texts, labels, tokenizer, max_length, ID=None):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.ID         = ID

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 텍스트 토크나이징 및 패딩
        encoding = self.tokenizer(str(self.texts.iloc[idx]),
                                  truncation=True,
                                  padding="max_length",
                                  max_length=self.max_length,
                                  return_tensors="pt",
                                  )

        # 기본 아이템 구성 (input_ids, attention_mask)
        item = {
                "input_ids":        encoding["input_ids"].flatten(),
                "attention_mask":   encoding["attention_mask"].flatten(),
                }

        # labels가 None이 아닌 경우에만 추가 (train/valid용)
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        if self.ID is not None:
            item["ID"] = torch.tensor(self.ID.iloc[idx], dtype=torch.long)

        return item
