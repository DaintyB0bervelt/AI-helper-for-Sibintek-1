import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import classification_report

# 1. Загрузка и предобработка данных
df = pd.read_excel(r'C:\Users\selvl\Desktop\ITOG\Hackaton\dataset.xlsx')

# Заполнение пустых значений в 'Solution' пустой строкой
df['Solution'] = df['Solution'].fillna('')

# 2. Объединение редких классов
threshold = 5  # Классы с количеством образцов меньше 5 считаются редкими

# Используем исходные текстовые метки из 'Label'
class_counts = df['label'].value_counts()
rare_classes = class_counts[class_counts < threshold].index.tolist()

# Объединяем редкие классы в 'Other'
df['label_combined'] = df['label'].apply(
    lambda x: 'Other' if x in rare_classes else x
)

# Преобразуем метки в числовой формат
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label_combined'])

# Получаем список классов и их количество
classes = label_encoder.classes_
num_classes = len(classes)
print(f"Количество классов после объединения: {num_classes}")
print(f"Список классов: {classes}")

# 3. Разделение данных на обучающую и тестовую выборки
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=80,
    stratify=df['label_encoded']
)

# 4. Создание датасета
class TopicDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 5. Загрузка модели и токенизатора
model_name = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_classes
)

# 6. Создание датасетов для обучения и тестирования
train_dataset = TopicDataset(
    train_df['Topic'],
    train_df['label_encoded'],
    tokenizer
)
test_dataset = TopicDataset(
    test_df['Topic'],
    test_df['label_encoded'],
    tokenizer
)

# 7. Настройка параметров обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy='epoch',  # Заменено 'evaluation_strategy' на 'eval_strategy'
    logging_dir='./logs',
    logging_steps=10,
)

# 8. Обучение модели
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    # tokenizer=tokenizer,  # Удалено, так как параметр устарел
)

trainer.train()

# 9. Оценка модели
from sklearn.metrics import classification_report

# Получаем список всех меток
labels = list(range(len(classes)))

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)

print(classification_report(
    test_df['label_encoded'],
    preds,
    labels=labels,
    target_names=classes,
    zero_division=0
))

# 10. Сохранение модели и токенизатора
model.save_pretrained('saved_model')
tokenizer.save_pretrained('saved_model')
