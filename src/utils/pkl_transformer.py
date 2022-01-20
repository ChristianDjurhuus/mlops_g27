import pickle
import torch
from transformers import AutoModelForSequenceClassification
from src.models.model import ImdbTransformer
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader

pt_file = r'C:\Users\andre\Documents\GitHub\mlops_g27\models\model.ckpt'

classification = {0:'Negative', 1:'Positive'}
sample_data = {
    "text": [
        "This is a good movie, my family are all like it!",
        "I don't like this film, The story is too slow.",
    ],
    "label": [1, 0],
}

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(sample_data):
    return tokenizer(sample_data['text'], padding="max_length", truncation=True)

tokenized_sample = Dataset.from_dict(sample_data).map(tokenize_function, batched=True)
tokenized_sample = tokenized_sample.remove_columns(["text"])
tokenized_sample = tokenized_sample.rename_column("label", "labels")
tokenized_sample.set_format("torch")
dataloader = DataLoader(tokenized_sample)
model = ImdbTransformer(
        model_name="bert-base-cased",
        learning_rate=0.01,
        batch_size=24,
    )
#model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

model.load_state_dict(torch.load(pt_file, map_location=torch.device('cpu'))['state_dict'])
model.eval()
with torch.no_grad():
    for batch in dataloader:
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        print(classification[predictions.data[0].item()])
#output = model(**tokenized_sample)
#print(output)

#with open('..\..\models\model.pkl', 'wb') as file:
#    pickle.dump(model, file)